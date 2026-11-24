import types
import logging
import traceback

import grpc
import torch
import numpy as np

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.trainer_tools import process_sample, _get_input_tensor_for_sample
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType
from weightslab.components.global_monitoring import weightslab_rlock, try_acquire_rlock, _GRPC_LOCK_TIMEOUT_S


logger = logging.getLogger(__name__)


class ModelService:
    """
    Model-centric operations: samples, weights, activations, architecture ops.
    """

    def __init__(self, ctx):
        self._ctx = ctx

    # -------------------------------------------------------------------------
    # Sample retrieval (images / segmentation / recon)
    # -------------------------------------------------------------------------
    def GetSamples(self, request, context):
        import concurrent.futures

        self._ctx.ensure_components()

        components = self._ctx.components

        # Dynamically find the loader for the requested origin
        from weightslab.backend.ledgers import get_dataloaders
        loader_names = get_dataloaders()

        ds = None
        for loader_name in loader_names:
            loader = components.get(loader_name)
            if loader is None:
                continue
            tracked_ds = getattr(loader, "tracked_dataset", None)
            if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                if tracked_ds._dataset_split == request.origin:
                    ds = loader
                    break

        if ds is None:
            logger.warning(f"No loader found for origin '{request.origin}'")
            return pb2.BatchSampleResponse()

        dataset = getattr(ds, "tracked_dataset", ds)
        response = pb2.BatchSampleResponse()

        do_resize = request.HasField("resize_width") and request.HasField("resize_height")
        resize_dims = (request.resize_width, request.resize_height) if do_resize else None
        task_type = getattr(
            dataset,
            "task_type",
            getattr(components.get("model"), "task_type", "classification"),
        )

        with concurrent.futures.ThreadPoolExecutor(thread_name_prefix="get_samples_worker") as executor:
            fut_map = {
                executor.submit(
                    process_sample,
                    sid,
                    dataset,
                    do_resize,
                    resize_dims,
                    types.SimpleNamespace(
                        **{
                            "tasks": getattr(components.get("model"), "tasks", None),
                            "task_type": getattr(
                                components.get("model"),
                                "task_type",
                                getattr(dataset, "task_type", "classification"),
                            ),
                            "num_classes": getattr(components.get("model"), "num_classes", None),
                        }
                    ),
                ): sid
                for sid in request.sample_ids
            }
            results = {}
            for future in concurrent.futures.as_completed(fut_map):
                sid, transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes = future.result()
                results[sid] = (transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes)

        # build response preserving input order
        for sid in request.sample_ids:
            transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes = results.get(
                sid, (None, None, -1, b"", b"")
            )
            if transformed_bytes is None or raw_bytes is None:
                continue

            if task_type == "segmentation":
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=cls_label,
                    data=transformed_bytes,
                    raw_data=raw_bytes,
                    mask=mask_bytes,
                    prediction=pred_bytes or b"",
                )
            elif pred_bytes and len(pred_bytes) > 0:
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=-1,
                    data=transformed_bytes,
                    raw_data=raw_bytes,
                    mask=b"",
                    prediction=pred_bytes,
                )
            else:
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=cls_label,
                    data=transformed_bytes,
                    raw_data=raw_bytes,
                    mask=b"",
                    prediction=b"",
                )
            response.samples.append(sample_response)

        return response

    # -------------------------------------------------------------------------
    # Weights inspection
    # -------------------------------------------------------------------------
    def GetWeights(self, request, context):
        self._ctx.ensure_components()

        components = self._ctx.components

        answer = pb2.WeightsResponse(success=True, error_message="")

        neuron_id = request.neuron_id
        layer = None

        try:
            model = components.get("model")
            if model is None:
                answer.success = False
                answer.error_messages = "No model registered"
                return answer
            layer = model.get_layer_by_id(neuron_id.layer_id)
        except Exception as e:
            answer.success = False
            answer.error_messages = str(e)
            return answer

        answer.neuron_id.CopyFrom(request.neuron_id)
        answer.layer_name = layer.__class__.__name__
        answer.incoming = layer.in_neurons
        answer.outgoing = layer.out_neurons
        if "Conv2d" in layer.__class__.__name__:
            answer.layer_type = "Conv2d"
            answer.kernel_size = layer.kernel_size[0]
        elif "Linear" in layer.__class__.__name__:
            answer.layer_type = "Linear"

        if neuron_id.neuron_id >= layer.out_neurons:
            answer.success = False
            answer.error_messages = f"Neuron {neuron_id.neuron_id} outside bounds."
            return answer

        if neuron_id.neuron_id < 0:
            weights = layer.weight.data.cpu().detach().numpy().flatten()
        else:
            weights = layer.weight[neuron_id.neuron_id].data.cpu().detach().numpy().flatten()
        answer.weights.extend(weights)

        return answer

    # -------------------------------------------------------------------------
    # Activations
    # -------------------------------------------------------------------------
    def GetActivations(self, request, context):
        self._ctx.ensure_components()

        components = self._ctx.components

        empty_resp = pb2.ActivationResponse(layer_type="", neurons_count=0)

        try:
            model = components.get("model")
            if model is None:
                return empty_resp
            last_layer = model.layers[-1]
            last_layer_id = int(last_layer.get_module_id())
            if int(request.layer_id) == last_layer_id:
                return empty_resp

            # Dynamically find the loader for the requested origin
            from weightslab.backend.ledgers import get_dataloaders
            loader_names = get_dataloaders()

            ds = None
            for loader_name in loader_names:
                loader = components.get(loader_name)
                if loader is None:
                    continue
                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    if tracked_ds._dataset_split == request.origin:
                        ds = getattr(loader, "tracked_dataset", loader)
                        break

            if ds is None:
                logger.warning(f"No dataset found for origin '{request.origin}'")
                return empty_resp

            requested_sample_id = str(request.sample_id)
            if requested_sample_id == "":
                raise ValueError(f"No sample id {request.sample_id} for {request.origin}")

            sid = requested_sample_id
            if hasattr(ds, "get_index_from_sample_id"):
                try:
                    ds.get_index_from_sample_id(requested_sample_id)
                except Exception:
                    try:
                        ds.get_index_from_sample_id(int(requested_sample_id))
                        sid = int(requested_sample_id)
                    except Exception as exc:
                        raise ValueError(f"No sample id {request.sample_id} for {request.origin}") from exc
            x = _get_input_tensor_for_sample(ds, sid, getattr(model, "device", "cpu"))

            with torch.no_grad():
                intermediaries = {}
                handles = []

                try:
                    def make_hook(module):
                        def hook(mod, inp, out):
                            try:
                                mid = None
                                if hasattr(mod, "get_module_id"):
                                    try:
                                        mid = mod.get_module_id()
                                    except Exception:
                                        mid = None
                                if mid is None:
                                    return
                                key = mid
                                try:
                                    intermediaries[key] = out.detach().cpu()
                                except Exception:
                                    try:
                                        intermediaries[key] = out[0].detach().cpu()
                                    except Exception:
                                        intermediaries[key] = None
                            except Exception:
                                pass

                        return hook

                    for layer in model.layers:
                        try:
                            h = layer.register_forward_hook(make_hook(layer))
                            handles.append(h)
                        except Exception:
                            pass

                    try:
                        _ = model(x)
                    except Exception:
                        pass
                finally:
                    for h in handles:
                        try:
                            h.remove()
                        except Exception:
                            pass

            if intermediaries[request.layer_id] is None:
                raise ValueError(f"No intermediary layer {request.layer_id}")

            layer = model.get_layer_by_id(request.layer_id)
            layer_type = layer.__class__.__name__
            amap = intermediaries[request.layer_id].squeeze(0).detach().cpu().numpy()
            resp = pb2.ActivationResponse(layer_type=layer_type)

            C, H, W = 1, 1, 1
            if amap.ndim == 3:
                C, H, W = amap.shape
            elif amap.ndim == 1:
                C = amap.shape[0]

            resp.neurons_count = C
            for c in range(C):
                vals = amap[c].astype(np.float32).reshape(-1).tolist()
                if not isinstance(vals, list):
                    vals = [vals]
                resp.activations.append(
                    pb2.ActivationMap(neuron_id=c, values=vals, H=H, W=W)
                )
            return resp
        except (ValueError, Exception) as e:
            logger.error(f"Error in GetActivations: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")

        return empty_resp

    # -------------------------------------------------------------------------
    # Weight manipulation (architecture operations)
    # -------------------------------------------------------------------------
    @staticmethod
    def _abort_or_fail(context, message: str) -> None:
        """Abort the gRPC call when a context is present, no-op otherwise.

        Lets in-process callers (e.g. the agent) invoke ManipulateWeights
        directly with ``context=None`` without crashing on a lock timeout.
        """
        if context is not None:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, message)

    def ManipulateWeights(self, request, context):
        self._ctx.ensure_components()

        components = self._ctx.components

        answer = pb2.WeightsOperationResponse(success=False, message="Unknown error")
        weight_operations = request.weight_operation

        if weight_operations.op_type == pb2.WeightOperationType.ADD_NEURONS:
            op_type = ArchitectureNeuronsOpType.ADD
        elif weight_operations.op_type == pb2.WeightOperationType.REMOVE_NEURONS:
            op_type = ArchitectureNeuronsOpType.PRUNE
        elif weight_operations.op_type == pb2.WeightOperationType.FREEZE:
            op_type = ArchitectureNeuronsOpType.FREEZE
        elif weight_operations.op_type == pb2.WeightOperationType.REINITIALIZE:
            op_type = ArchitectureNeuronsOpType.RESET
        else:
            op_type = None

        model = components.get("model")

        if model is None or op_type is None:
            return pb2.WeightsOperationResponse(
                success=False,
                message="Model not found or invalid op_type",
            )

        if len(weight_operations.neuron_ids) == 0:
            layer_id = weight_operations.layer_id
            neuron_id = []

            if not try_acquire_rlock():
                logger.error("[ManipulateWeights] weightslab_rlock timed out after %.0fs", _GRPC_LOCK_TIMEOUT_S)
                self._abort_or_fail(context, f"Training lock not acquired within {_GRPC_LOCK_TIMEOUT_S:.0f}s")
                return pb2.WeightsOperationResponse(success=False, message="Lock timeout")
            try:
                model.apply_architecture_op(
                    op_type=op_type,
                    layer_id=layer_id,
                    neuron_indices=neuron_id,
                )
            finally:
                weightslab_rlock.release()

        else:
            for neuron_details in weight_operations.neuron_ids:
                layer_id = neuron_details.layer_id
                neuron_id = neuron_details.neuron_id

                if not try_acquire_rlock():
                    logger.error("[ManipulateWeights] weightslab_rlock timed out after %.0fs", _GRPC_LOCK_TIMEOUT_S)
                    self._abort_or_fail(context, f"Training lock not acquired within {_GRPC_LOCK_TIMEOUT_S:.0f}s")
                    return pb2.WeightsOperationResponse(success=False, message="Lock timeout")
                try:
                    model.apply_architecture_op(
                        op_type=op_type,
                        layer_id=layer_id,
                        neuron_indices=neuron_id,
                    )
                finally:
                    weightslab_rlock.release()

        answer = pb2.WeightsOperationResponse(
            success=True,
            message=f"{weight_operations.op_type} - {weight_operations.neuron_ids}",
        )

        return answer

    # -------------------------------------------------------------------------
    # Modeling tab (architecture graph, representations, layer edits)
    # -------------------------------------------------------------------------
    def _modeling_targets(self):
        """(model, {loader_name: loader}) from the ledger, or (None, {})."""
        self._ctx.ensure_components()
        components = self._ctx.components
        model = components.get("model")
        loaders = {}
        try:
            from weightslab.backend.ledgers import get_dataloaders
            for name in get_dataloaders():
                loader = components.get(name)
                if loader is not None:
                    loaders[name] = loader
        except Exception as exc:
            logger.debug(f"[modeling] no dataloaders: {exc}")
        try:
            from weightslab.components.model_representation import resolve_inner
            if resolve_inner(model) is None:
                model = None
        except Exception:
            model = None
        return model, loaders

    @staticmethod
    def _graph_node_pb(node: dict):
        fields = {k: v for k, v in node.items() if k in pb2.ModelGraphNode.DESCRIPTOR.fields_by_name}
        return pb2.ModelGraphNode(**fields)

    @staticmethod
    def _summary_pb(summary: dict):
        fields = {k: v for k, v in summary.items()
                  if k in pb2.ModelNodeActivationSummary.DESCRIPTOR.fields_by_name}
        return pb2.ModelNodeActivationSummary(**fields)

    def GetModelArchitecture(self, request, context):
        from weightslab.components.model_representation import MANAGER
        model, loaders = self._modeling_targets()
        if model is None:
            return pb2.ModelArchitectureResponse(success=False, message="No model registered")
        try:
            arch = MANAGER.architecture(model, loaders, refresh=bool(request.refresh))
        except Exception as exc:
            logger.error(f"[modeling] GetModelArchitecture failed: {exc}", exc_info=True)
            return pb2.ModelArchitectureResponse(success=False, message=str(exc))
        response = pb2.ModelArchitectureResponse(
            success=bool(arch.get("success")),
            message=arch.get("message", ""),
            model_name=arch.get("model_name", ""),
            model_age=int(arch.get("model_age", 0)),
            task_type=arch.get("task_type", ""),
            architecture_version=int(arch.get("architecture_version", 0)),
            dependencies_available=bool(arch.get("dependencies_available")),
            traced=bool(arch.get("traced")),
            trace_message=arch.get("trace_message", ""),
            total_parameters=int(arch.get("total_parameters", 0)),
            trainable_parameters=int(arch.get("trainable_parameters", 0)),
        )
        response.nodes.extend(self._graph_node_pb(n) for n in arch.get("nodes", []))
        response.edges.extend(
            pb2.ModelGraphEdge(source=e["source"], target=e["target"], shape=e.get("shape") or [])
            for e in arch.get("edges", []))
        return response

    def StartModelRepresentation(self, request, context):
        from weightslab.components.model_representation import MANAGER
        model, loaders = self._modeling_targets()
        if model is None:
            return pb2.ModelRepresentationStatus(state="error", message="No model registered")
        try:
            status = MANAGER.start(
                model, loaders,
                sample_ids=list(request.sample_ids), origins=list(request.origins),
                tag=request.tag, tag_origin=request.origin, max_samples=int(request.max_samples))
        except Exception as exc:
            logger.error(f"[modeling] StartModelRepresentation failed: {exc}", exc_info=True)
            return pb2.ModelRepresentationStatus(state="error", message=str(exc))
        return pb2.ModelRepresentationStatus(**status)

    def GetModelRepresentation(self, request, context):
        from weightslab.components.model_representation import MANAGER
        if request.clear:
            MANAGER.clear()
            return pb2.ModelRepresentationResponse(
                status=pb2.ModelRepresentationStatus(state="none"))
        session = MANAGER.get(request.session_id)
        if session is None:
            return pb2.ModelRepresentationResponse(
                status=pb2.ModelRepresentationStatus(state="none"))
        response = pb2.ModelRepresentationResponse(
            status=pb2.ModelRepresentationStatus(**session.status(MANAGER.is_stale(session))))
        response.nodes.extend(self._summary_pb(s) for s in MANAGER.summaries(session) if len(s) > 1)
        return response

    def GetModelLayerDetails(self, request, context):
        from weightslab.components.model_representation import MANAGER, layer_details
        model, loaders = self._modeling_targets()
        if model is None:
            return pb2.ModelLayerDetailsResponse(success=False, message="No model registered")
        try:
            details = layer_details(
                model, loaders, request.node_id, request.session_id,
                max_channels=int(request.max_channels), max_samples=int(request.max_samples),
                manager=MANAGER,
                skip_previews=bool(request.skip_previews),
                sample_id=request.sample_id, origin=request.origin,
                tile_size=int(request.tile_size))
        except Exception as exc:
            logger.error(f"[modeling] GetModelLayerDetails failed: {exc}", exc_info=True)
            return pb2.ModelLayerDetailsResponse(success=False, message=str(exc))
        if not details.get("success"):
            return pb2.ModelLayerDetailsResponse(success=False, message=details.get("message", ""))
        response = pb2.ModelLayerDetailsResponse(
            success=True,
            node=self._graph_node_pb(details["node"]),
            weight_histogram=details["weight_histogram"],
            weight_hist_min=details["weight_hist_min"],
            weight_hist_max=details["weight_hist_max"],
            session_id=details["session_id"],
            neuron_axis=details["neuron_axis"],
            health_source=details.get("health_source", ""),
            model_age=int(details.get("model_age", 0)),
            children=details["children"],
        )
        response.neurons.extend(pb2.ModelNeuronStat(**n) for n in details["neurons"])
        response.neuron_groups.extend(pb2.ModelNeuronGroup(**g) for g in details.get("neuron_groups", []))
        response.previews.extend(pb2.ModelTensorPreview(**p) for p in details["previews"])
        if details.get("activation") and len(details["activation"]) > 1:
            response.activation.CopyFrom(self._summary_pb(details["activation"]))
        return response

    def EditModelLayer(self, request, context):
        from weightslab.components.model_representation import MANAGER, edit_layer
        model, loaders = self._modeling_targets()
        if model is None:
            return pb2.ModelLayerEditResponse(success=False, message="No model registered")
        if not try_acquire_rlock():
            self._abort_or_fail(context, f"Training lock not acquired within {_GRPC_LOCK_TIMEOUT_S:.0f}s")
            return pb2.ModelLayerEditResponse(success=False, message="Lock timeout")
        try:
            result = edit_layer(
                model, request.node_id, request.op,
                neuron_indices=list(request.neuron_indices),
                count=int(request.count or 1),
                ratio=float(request.ratio) if request.ratio else 0.1,
                manager=MANAGER)
        except ValueError as exc:
            return pb2.ModelLayerEditResponse(success=False, message=str(exc))
        except Exception as exc:
            logger.error(f"[modeling] EditModelLayer failed: {exc}", exc_info=True)
            return pb2.ModelLayerEditResponse(success=False, message=f"{type(exc).__name__}: {exc}")
        finally:
            weightslab_rlock.release()
        response = pb2.ModelLayerEditResponse(
            success=True, message=result.get("message", ""),
            architecture_version=int(MANAGER.version))
        try:
            arch = MANAGER.architecture(model, loaders)
            node = next((n for n in arch["nodes"] if n["id"] == request.node_id), None)
            if node is not None:
                response.node.CopyFrom(self._graph_node_pb(node))
            response.architecture_version = int(arch.get("architecture_version", MANAGER.version))
        except Exception as exc:
            logger.debug(f"[modeling] could not refresh the edited node: {exc}")
        return response

    # ---- Compare A vs B ------------------------------------------------------
    @staticmethod
    def _comparison_status_pb(status: dict):
        fields = dict(status)
        sides = {k: pb2.ModelComparisonSideStatus(**fields.pop(k)) for k in ("a", "b") if k in fields}
        return pb2.ModelComparisonStatus(**fields, **sides)

    @staticmethod
    def _node_diff_pb(diff: dict):
        fields = {k: v for k, v in diff.items() if k in pb2.ModelNodeDiff.DESCRIPTOR.fields_by_name}
        return pb2.ModelNodeDiff(**fields)

    @staticmethod
    def _comparison_side(side) -> dict:
        return {"tag": side.tag, "sample_ids": list(side.sample_ids), "origins": list(side.origins),
                "label": side.label, "origin": side.origin}

    def StartModelComparison(self, request, context):
        from weightslab.components.model_comparison import COMPARE
        model, loaders = self._modeling_targets()
        if model is None:
            return pb2.ModelComparisonStatus(state="error", message="No model registered")
        try:
            status = COMPARE.start(
                model, loaders, self._comparison_side(request.a), self._comparison_side(request.b),
                max_samples=int(request.max_samples))
        except Exception as exc:
            logger.error(f"[modeling] StartModelComparison failed: {exc}", exc_info=True)
            return pb2.ModelComparisonStatus(state="error", message=str(exc))
        return self._comparison_status_pb(status)

    def GetModelComparison(self, request, context):
        from weightslab.components.model_comparison import COMPARE
        if request.clear:
            COMPARE.clear()
            return pb2.ModelComparisonResponse(status=pb2.ModelComparisonStatus(state="none"))
        session = COMPARE.get(request.comparison_id)
        if session is None:
            return pb2.ModelComparisonResponse(status=pb2.ModelComparisonStatus(state="none"))
        response = pb2.ModelComparisonResponse(
            status=self._comparison_status_pb(session.status(COMPARE.is_stale(session))))
        try:
            response.nodes.extend(self._node_diff_pb(d) for d in COMPARE.diffs(session))
        except Exception as exc:
            logger.debug(f"[modeling] comparison diffs failed: {exc}", exc_info=True)
        return response

    def GetModelLayerComparison(self, request, context):
        from weightslab.components.model_comparison import COMPARE
        try:
            result = COMPARE.layer(request.comparison_id, request.node_id)
        except Exception as exc:
            logger.error(f"[modeling] GetModelLayerComparison failed: {exc}", exc_info=True)
            return pb2.ModelLayerComparisonResponse(success=False, message=str(exc))
        response = pb2.ModelLayerComparisonResponse(
            success=bool(result.get("success")), message=result.get("message", ""),
            mode=result.get("mode", ""), neuron_axis=result.get("neuron_axis", ""))
        session = result.get("session")
        if session is not None:
            response.status.CopyFrom(self._comparison_status_pb(session.status(COMPARE.is_stale(session))))
        if result.get("summary"):
            response.summary.CopyFrom(self._node_diff_pb(result["summary"]))
        response.neurons.extend(pb2.ModelNeuronDiff(**n) for n in result.get("neurons", []))
        return response

    # ---- Failure analysis ("Why do these fail?") ------------------------------
    @staticmethod
    def _failure_state_pb(status: dict, success: bool = True, message: str = ""):
        fields = dict(status)
        question = fields.pop("question", None)
        messages = fields.pop("messages", [])
        state = pb2.FailureAnalysisState(
            success=success, message=message or fields.pop("message", ""),
            **{k: v for k, v in fields.items() if k in pb2.FailureAnalysisState.DESCRIPTOR.fields_by_name
               and k != "message"})
        state.messages.extend(pb2.FailureAnalysisMessage(**m) for m in messages)
        if question:
            state.question.CopyFrom(pb2.FailureAnalysisQuestion(
                id=question["id"], text=question["text"], multiple=bool(question["multiple"]),
                defaults=list(question.get("defaults") or []),
                confirm_label=question.get("confirm_label", ""),
                choices=[pb2.FailureAnalysisChoice(id=c["id"], label=c["label"], hint=c.get("hint", ""),
                                                   primary=bool(c.get("primary")))
                         for c in question["choices"]]))
        return state

    def StartFailureAnalysis(self, request, context):
        import json
        from weightslab.components.failure_analysis import FAILURES
        model, loaders = self._modeling_targets()
        if model is None:
            return pb2.FailureAnalysisState(success=False, state="error", message="No model registered")
        try:
            options = json.loads(request.options_json) if request.options_json else {}
            session = FAILURES.start(
                model, loaders, tag=request.tag, sample_ids=list(request.sample_ids),
                origins=list(request.origins), origin=request.origin,
                automatic=bool(request.automatic), options=options)
        except Exception as exc:
            logger.error(f"[failures] StartFailureAnalysis failed: {exc}", exc_info=True)
            return pb2.FailureAnalysisState(success=False, state="error", message=str(exc))
        return self._failure_state_pb(session.status())

    def GetFailureAnalysis(self, request, context):
        from weightslab.components.failure_analysis import FAILURES
        session = FAILURES.get(request.session_id)
        if session is None:
            return pb2.FailureAnalysisState(success=True, state="none")
        if request.set_automatic:
            session.set_automatic(bool(request.automatic))
        if request.cancel:
            session.cancel.set()
            session.answered.set()
        return self._failure_state_pb(session.status(int(request.since)))

    def AnswerFailureAnalysis(self, request, context):
        from weightslab.components.failure_analysis import FAILURES
        session = FAILURES.get(request.session_id)
        if session is None:
            return pb2.FailureAnalysisState(success=False, state="none", message="no analysis")
        ok, why = session.answer(request.question_id, request.choice, list(request.choices))
        return self._failure_state_pb(session.status(), success=ok, message=why)
