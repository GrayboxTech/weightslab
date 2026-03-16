import types
import logging
import traceback

import torch
import numpy as np

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.trainer_tools import process_sample, _get_input_tensor_for_sample
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType
from weightslab.components.global_monitoring import weightslab_rlock


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
        logger.debug(f"ExperimentServiceServicer.GetSamples({request})")

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
        logger.debug(f"ExperimentServiceServicer.GetWeights({request})")

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
        logger.debug(f"ExperimentServiceServicer.GetActivations({request})")

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
    def ManipulateWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ManipulateWeights({request})")

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

            with weightslab_rlock:
                model.apply_architecture_op(
                    op_type=op_type,
                    layer_id=layer_id,
                    neuron_indices=neuron_id,
                )

        else:
            for neuron_details in weight_operations.neuron_ids:
                layer_id = neuron_details.layer_id
                neuron_id = neuron_details.neuron_id

                with weightslab_rlock:
                    model.apply_architecture_op(
                        op_type=op_type,
                        layer_id=layer_id,
                        neuron_indices=neuron_id,
                    )

        answer = pb2.WeightsOperationResponse(
            success=True,
            message=f"{weight_operations.op_type} - {weight_operations.neuron_ids}",
        )

        return answer
