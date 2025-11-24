import io
import types
from threading import Thread
import time
import grpc
import time
import torch
import traceback
import numpy as np
import pandas as pd
import logging
import weightslab.proto.experiment_service_pb2 as pb2
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from concurrent import futures
from collections import defaultdict

from weightslab.trainer.trainer_tools import *
from weightslab.trainer.trainer_tools import _get_input_tensor_for_sample
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType

_LOGGER = logging.getLogger(__name__)


class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    def __init__(self, exp_name: str = None):
        # Accept an explicit experiment object or attempt to resolve one
        # from the global ledger to avoid depending on a single global
        # `experiment` instance.
        from weightslab.ledgers import GLOBAL_LEDGER

        self._exp_name = exp_name
        # Components resolved from GLOBAL_LEDGER (model, dataloaders, optimizer,
        # hyperparams, logger). We keep explicit experiment for backwards
        # compatibility if provided, otherwise we resolve parts lazily.
        self._components = {}
        
        # Data service components (initialized lazily on first use)
        self._all_datasets_df = None
        self._agent = None

    def _ensure_components(self):
        """Ensure ledger-backed components are resolved and available on
        `self` (model, train/test dataloaders, optimizer, hyperparams,
        logger). Raises RuntimeError when mandatory components are missing."""
        if getattr(self, '_components', None) and self._components:
            return
        from weightslab.ledgers import GLOBAL_LEDGER, get_hyperparams, list_hyperparams, get_model, list_models, get_dataloader, list_dataloaders, get_optimizer, list_optimizers, get_logger, list_loggers

        # resolve model
        model = None
        try:
            names = list_models()
            if self._exp_name and self._exp_name in names:
                model = get_model(self._exp_name)
            elif 'experiment' in names:
                model = get_model('experiment')
            elif len(names) == 1:
                model = get_model()
        except Exception:
            model = None

        # resolve dataloaders (prefer explicit names 'train' / 'eval' / 'test')
        train_loader = None
        test_loader = None
        try:
            dnames = list_dataloaders()
            if 'train' in dnames:
                train_loader = get_dataloader('train')
            elif len(dnames) == 1:
                train_loader = get_dataloader()

            if 'eval' in dnames:
                test_loader = get_dataloader('eval')
            elif 'test' in dnames:
                test_loader = get_dataloader('test')
            elif len(dnames) == 1 and train_loader is not None:
                test_loader = train_loader
        except Exception:
            train_loader = None
            test_loader = None

        # resolve optimizer
        optimizer = None
        try:
            onames = list_optimizers()
            if len(onames) == 1:
                optimizer = get_optimizer()
            elif '_optimizer' in onames:
                optimizer = get_optimizer('_optimizer')
        except Exception:
            optimizer = None

        # resolve hyperparams (by exp_name or single set)
        hyperparams = None
        try:
            hp_names = list_hyperparams()
            if self._exp_name and self._exp_name in hp_names:
                hyperparams = get_hyperparams(self._exp_name)
            elif len(hp_names) == 1:
                hyperparams = get_hyperparams()
        except Exception:
            hyperparams = None

        # resolve logger
        logger = None
        try:
            lnames = list_loggers()
            if len(lnames) == 1:
                logger = get_logger()
            elif 'main' in lnames:
                logger = get_logger('main')
        except Exception:
            logger = None

        self._components = {
            'model': model,
            'train_loader': train_loader,
            'test_loader': test_loader,
            'optimizer': optimizer,
            'hyperparams': hyperparams,
            'logger': logger,
        }

        # Build hyper-parameter descriptors used by the protocol. Use
        # ledger-backed hyperparams when available, with safe fallbacks.
        def _hp_getter(key, default=None):
            def _g():
                try:
                    hp = self._components.get('hyperparams')
                    if '.' in key:
                        parts = key.split('.') if key else []
                        cur = hp
                        for p in parts:
                            cur = cur[p]
                        return cur
                    if isinstance(hp, dict):
                        return hp.get(key, default)
                    elif hasattr(hp, 'get'):
                        return hp.get(key, default)
                except Exception:
                    pass
                return default
            return _g

        self.hyper_parameters = {
            ("Experiment Name", "experiment_name", "text", lambda: _hp_getter('experiment_name', 'Anonymous')()),
            ("Left Training Steps", "training_left", "number", _hp_getter('training_steps_to_do', 999)),
            ("Eval Frequency", "eval_frequency", "number", _hp_getter('eval_full_to_train_steps_ratio', 100)),
            ("Checkpoint Frequency", "checkpooint_frequency", "number", _hp_getter('experiment_dump_to_train_steps_ratio', 100)),

            ("Learning Rate", "learning_rate", "number", _hp_getter('optimizer.lr', 1e-4)),
            ("Batch Size", "batch_size", "number", _hp_getter('data.train_dataset.batch_size', 8))
        }

    def StreamStatus(self, request_iterator, context):
        # ensure ledger-backed components are available
        self._ensure_components()
        while True:
            # use ledger-backed logger queue when available
            logger = self._components.get('logger') if getattr(self, '_components', None) else None
            if logger is None or not hasattr(logger, 'queue'):
                raise RuntimeError('No logger with a queue registered in GLOBAL_LEDGER')
            log = logger.queue.get()
            if "metric_name" in log and "acc" in log["metric_name"]:
                # print(f"[LOG] {log['metric_name']} = {log['metric_value']:.2f}")
                pass

            if log is None:
                break
            metrics_status, annotat_status = None, None
            if "metric_name" in log:
                metrics_status = pb2.MetricsStatus(
                    name=log["metric_name"],
                    value=log["metric_value"],
                )
            elif "annotation" in log:
                annotat_status = pb2.AnnotatStatus(
                    name=log["annotation"])
                for key, value in log["metadata"].items():
                    annotat_status.metadata[key] = value

            training_status = pb2.TrainingStatusEx(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                experiment_name=log["experiment_name"],
                model_age=log["model_age"],
            )

            if metrics_status:
                training_status.metrics_status.CopyFrom(metrics_status)
            if annotat_status:
                training_status.annotat_status.CopyFrom(annotat_status)

            # mark task done on ledger logger queue
            logger = self._components.get('logger')
            if logger is not None and hasattr(logger, 'queue'):
                try:
                    logger.queue.task_done()
                except Exception:
                    pass

            yield training_status

    def ExperimentCommand(self, request, context):
        # print("ExperimentServiceServicer.ExperimentCommand", request)
        # ensure ledger-backed components are available
        self._ensure_components()
        if request.HasField('hyper_parameter_change'):
            # apply hyperparameter changes via the ledger
            hyper_parameters = request.hyper_parameter_change.hyper_parameters
            from weightslab.ledgers import set_hyperparam, list_hyperparams

            # resolve hp name: prefer explicit exp_name, else single set
            hp_name = None
            if self._exp_name:
                hp_name = self._exp_name
            else:
                hps = list_hyperparams()
                if len(hps) == 1:
                    hp_name = hps[0]

            if hp_name is None:
                return pb2.CommandResponse(success=False, message='Cannot resolve hyperparams name')

            try:
                if hyper_parameters.HasField('is_training'):
                    set_hyperparam(hp_name, 'is_training', hyper_parameters.is_training)

                if hyper_parameters.HasField('training_steps_to_do'):
                    set_hyperparam(hp_name, 'training_steps_to_do', hyper_parameters.training_steps_to_do)

                if hyper_parameters.HasField('learning_rate'):
                    set_hyperparam(hp_name, 'optimizer.lr', hyper_parameters.learning_rate)

                if hyper_parameters.HasField('batch_size'):
                    set_hyperparam(hp_name, 'data.train_dataset.batch_size', hyper_parameters.batch_size)

                # full_eval_frequency
                if hyper_parameters.HasField('full_eval_frequency'):
                    set_hyperparam(hp_name, 'eval_full_to_train_steps_ratio', hyper_parameters.full_eval_frequency)

                # checkpoint_frequency
                if hyper_parameters.HasField('checkpont_frequency'):
                    set_hyperparam(hp_name, 'experiment_dump_to_train_steps_ratio', hyper_parameters.checkpont_frequency)

            except Exception as e:
                return pb2.CommandResponse(success=False, message=f'Failed to set hyperparameters: {e}')

            return pb2.CommandResponse(success=True, message="Hyper parameter changed")
        if request.HasField('deny_samples_operation'):
            denied_cnt = len(request.deny_samples_operation.sample_ids)
            ds = self._components.get('train_loader')
            if ds is None:
                return pb2.CommandResponse(success=False, message='No train dataloader registered')
            dataset = getattr(ds, 'dataset', ds)
            dataset.denylist_samples(
                set(request.deny_samples_operation.sample_ids),
                accumulate = request.deny_samples_operation.accumulate
            )
            return pb2.CommandResponse(success=True, message=f"Denied {denied_cnt} train samples")
        if request.HasField('deny_eval_samples_operation'):
            denied_cnt = len(request.deny_eval_samples_operation.sample_ids)
            ds = self._components.get('test_loader')
            if ds is None:
                return pb2.CommandResponse(success=False, message='No eval dataloader registered')
            dataset = getattr(ds, 'dataset', ds)
            dataset.denylist_samples(
                set(request.deny_eval_samples_operation.sample_ids),
                accumulate = request.deny_eval_samples_operation.accumulate
            )
            return pb2.CommandResponse(success=True, message=f"Denied {denied_cnt} eval samples")

        if request.HasField('remove_from_denylist_operation'):
            allowed = set(request.remove_from_denylist_operation.sample_ids)
            ds = self._components.get('train_loader')
            if ds is None:
                return pb2.CommandResponse(success=False, message='No train dataloader registered')
            dataset = getattr(ds, 'dataset', ds)
            dataset.allowlist_samples(allowed)
            return pb2.CommandResponse(success=True, message=f"Un-denied {len(allowed)} train samples")

        if request.HasField('remove_eval_from_denylist_operation'):
            allowed = set(request.remove_eval_from_denylist_operation.sample_ids)
            ds = self._components.get('test_loader')
            if ds is None:
                return pb2.CommandResponse(success=False, message='No eval dataloader registered')
            dataset = getattr(ds, 'dataset', ds)
            dataset.allowlist_samples(allowed)
            return pb2.CommandResponse(success=True, message=f"Un-denied {len(allowed)} eval samples")

        if request.HasField('load_checkpoint_operation'):
            checkpoint_id = request.load_checkpoint_operation.checkpoint_id
            model = self._components.get('model')
            if model is None:
                return pb2.CommandResponse(success=False, message='No model registered to load checkpoint')
            if hasattr(model, 'load'):
                try:
                    model.load(checkpoint_id)
                except Exception as e:
                    return pb2.CommandResponse(success=False, message=str(e))

        response = pb2.CommandResponse(success=True, message="")
        if request.get_hyper_parameters:
            response.hyper_parameters_descs.extend(
                get_hyper_parameters_pb(self.hyper_parameters))
        if request.get_interactive_layers:
            model = self._components.get('model')
            if model is not None:
                if request.HasField('get_single_layer_info_id'):
                    response.layer_representations.extend([
                        get_layer_representation(
                            model.get_layer_by_id(request.get_single_layer_info_id))])
                else:
                    response.layer_representations.extend(get_layer_representations(model))
        if request.get_data_records:
            if request.get_data_records == "train":
                ds = self._components.get('train_loader')
                if ds is not None:
                    dataset = getattr(ds, 'dataset', ds)
                    response.sample_statistics.CopyFrom(
                        get_data_set_representation(
                            dataset,
                            types.SimpleNamespace(**{
                                'tasks': getattr(self._components.get('model'), 'tasks', None),
                                'task_type': getattr(self._components.get('model'), 'task_type', getattr(dataset, 'task_type', 'classification')),
                                'num_classes': getattr(self._components.get('model'), 'num_classes', None)
                            })
                        )
                    )
                    response.sample_statistics.origin = "train"
            elif request.get_data_records == "eval":
                ds = self._components.get('test_loader')
                if ds is not None:
                    dataset = getattr(ds, 'dataset', ds)
                    response.sample_statistics.CopyFrom(
                        get_data_set_representation(
                            dataset,
                            types.SimpleNamespace(**{
                                'tasks': getattr(self._components.get('model'), 'tasks', None),
                                'task_type': getattr(self._components.get('model'), 'task_type', getattr(dataset, 'task_type', 'classification')),
                                'num_classes': getattr(self._components.get('model'), 'num_classes', None)
                            })
                        )
                    )
                    response.sample_statistics.origin = "eval"

        return response

    def GetSample(self, request, context):
        # print(f"ExperimentServiceServicer.GetSample({request})")

        # ensure ledger-backed components are available
        self._ensure_components()

        if not request.HasField('sample_id') or not request.HasField('origin'):
            return pb2.SampleRequestResponse(
                error_message="Invalid request. Provide sample_id & origin.")

        if request.origin not in ["train", "eval"]:
            return pb2.SampleRequestResponse(
                error_message=f"Invalid origin {request.origin}")

        if request.sample_id < 0:
            return pb2.SampleRequestResponse(
                error_message=f"Invalid sample_id {request.sample_id}")

        dataset = None
        if request.origin == "train":
            ds = self._components.get('train_loader')
            dataset = getattr(ds, 'dataset', ds) if ds is not None else None
        elif request.origin == "eval":
            ds = self._components.get('test_loader')
            dataset = getattr(ds, 'dataset', ds) if ds is not None else None

        if dataset is None:
            return pb2.SampleRequestResponse(
                error_message=f"Dataset {request.origin} not found.")

        if dataset is None:
            return pb2.SampleRequestResponse(error_message=f"Dataset {request.origin} not found.")

        if request.sample_id >= len(dataset):
            return pb2.SampleRequestResponse(
                error_message=f"Sample {request.sample_id} not found.")

        transformed_tensor, idx, label = dataset._getitem_raw(request.sample_id)
        # #TODO: apply transform too
        
        transformed_image_bytes = tensor_to_bytes(transformed_tensor)

        try:
            pil_img = load_raw_image(dataset, request.sample_id)
            buf = io.BytesIO()
            # pil_img.save(buf, format='PNG')
            pil_img.save(buf, format='jpeg', quality=85)
            raw_image_bytes = buf.getvalue()
        except Exception as e:
            return pb2.SampleRequestResponse(error_message=str(e))

        response = pb2.SampleRequestResponse(
            sample_id=request.sample_id,
            origin=request.origin,
            label=label,
            raw_data=raw_image_bytes,         
            data=transformed_image_bytes, 
        )

        return response

    def GetSamples(self, request, context):
        print(f"ExperimentServiceServicer.GetSamples({request})")

        import concurrent.futures
        # ensure ledger-backed components are available
        self._ensure_components()

        ds = self._components.get('train_loader') if request.origin == "train" else self._components.get('test_loader')
        dataset = getattr(ds, 'dataset', ds)
        response = pb2.BatchSampleResponse()

        do_resize = request.HasField("resize_width") and request.HasField("resize_height")
        resize_dims = (request.resize_width, request.resize_height) if do_resize else None
        task_type = getattr(dataset, "task_type", getattr(self._components.get('model'), "task_type", "classification"))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut_map = {
                executor.submit(process_sample, sid, dataset, do_resize, resize_dims, types.SimpleNamespace(**{
                    'tasks': getattr(self._components.get('model'), 'tasks', None),
                    'task_type': getattr(self._components.get('model'), 'task_type', getattr(dataset, 'task_type', 'classification')),
                    'num_classes': getattr(self._components.get('model'), 'num_classes', None)
                })): sid
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
                    prediction=pred_bytes or b""  
                )
            elif pred_bytes and len(pred_bytes) > 0:
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=-1, 
                    data=transformed_bytes,  
                    raw_data=raw_bytes,      
                    mask=b"",  # Empty for reconstruction        
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

    def _apply_zerofy(self, layer, from_ids, to_ids):
        in_max = int(layer.in_neurons)
        out_max = int(layer.out_neurons)

        from_set = {i for i in set(from_ids) if 0 <= i < in_max}
        to_set   = {i for i in set(to_ids)   if 0 <= i < out_max}
        if from_set and to_set and hasattr(layer, "zerofy_connections_from"):
            layer.zerofy_connections_from(from_set, to_set)
        else:
            print("ZEROFY skipped (empty sets or no method)")

    def ManipulateWeights(self, request, context):
        # (f"ExperimentServiceServicer.ManipulateWeights({request})")
        # ensure ledger-backed components are available
        self._ensure_components()
        answer = pb2.WeightsOperationResponse(
            success=False, message="Unknown error")
        weight_operations = request.weight_operation

        if weight_operations.op_type == pb2.WeightOperationType.REMOVE_NEURONS:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id = neuron_id.layer_id
                layer_id_to_neuron_ids_list[layer_id].append(
                    neuron_id.neuron_id)

            model = self._components.get('model')
            if model is None or not hasattr(model, 'apply_architecture_op'):
                return pb2.WeightsOperationResponse(success=False, message='Model does not support architecture operations')
            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                model.apply_architecture_op(
                    op_type=ArchitectureNeuronsOpType.PRUNE,
                    layer_id=layer_id,
                    neuron_indices=set(neuron_ids))

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Pruned {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.ADD_NEURONS:
            model = self._components.get('model')
            if model is None or not hasattr(model, 'apply_architecture_op'):
                return pb2.WeightsOperationResponse(success=False, message='Model does not support architecture operations')
            model.apply_architecture_op(
                op_type=ArchitectureNeuronsOpType.ADD,
                layer_id=weight_operations.layer_id,
                neuron_indices=weight_operations.neurons_to_add)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=\
                    f"Added {weight_operations.neurons_to_add} "
                    f"neurons to layer {weight_operations.layer_id}")
        elif weight_operations.op_type == pb2.WeightOperationType.FREEZE:
            layer_id_to_neuron_ids_list = defaultdict(list)
            for neuron_id in weight_operations.neuron_ids:
                layer_id = neuron_id.layer_id
                layer_id_to_neuron_ids_list[layer_id].append(
                    neuron_id.neuron_id)
            model = self._components.get('model')
            if model is None or not hasattr(model, 'apply_architecture_op'):
                return pb2.WeightsOperationResponse(success=False, message='Model does not support architecture operations')
            for layer_id, neuron_ids in layer_id_to_neuron_ids_list.items():
                model.apply_architecture_op(
                    op_type=ArchitectureNeuronsOpType.FREEZE,
                    layer_id=layer_id,
                    neuron_indices=neuron_ids)
            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Frozen {str(dict(layer_id_to_neuron_ids_list))}")
        elif weight_operations.op_type == pb2.WeightOperationType.REINITIALIZE:
            model = self._components.get('model')
            if model is None or not hasattr(model, 'apply_architecture_op'):
                return pb2.WeightsOperationResponse(success=False, message='Model does not support architecture operations')
            for neuron_id in weight_operations.neuron_ids:
                model.apply_architecture_op(
                    op_type=ArchitectureNeuronsOpType.RESET,
                    layer_id=neuron_id.layer_id,
                    neuron_indices={neuron_id.neuron_id})

            answer = pb2.WeightsOperationResponse(
                success=True,
                message=f"Reinitialized {weight_operations.neuron_ids}")

        elif weight_operations.op_type == pb2.WeightOperationType.ZEROFY:
            layer_id = weight_operations.layer_id
            model = self._components.get('model')
            if model is None:
                return pb2.WeightsOperationResponse(success=False, message='No model registered')
            layer = model.get_layer_by_id(layer_id)

            from_ids = list(weight_operations.zerofy_from_incoming_ids)
            to_ids   = list(weight_operations.zerofy_to_neuron_ids)

            if len(weight_operations.zerofy_predicates) > 0:
                frozen = set()
                try:
                    for nid in range(getattr(layer, "out_neurons", 0)):
                        if layer.get_per_neuron_learning_rate(
                            nid,
                            is_incoming=False,
                            tensor_name='weight'
                        ) == 0.0:
                            frozen.add(nid)
                except Exception:
                    pass

                older = set()
                try:
                    td_tracker = getattr(layer, "train_dataset_tracker", None)
                    if td_tracker is not None:
                        for nid in range(getattr(layer, "out_neurons", 0)):
                            age = int(td_tracker.get_neuron_age(nid))
                            if age > 0:
                                older.add(nid)
                except Exception:
                    older = set()

                expanded = set(to_ids)
                for p in weight_operations.zerofy_predicates:
                    if p == pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_FROZEN:
                        expanded |= frozen
                    elif p == pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_OLDER:
                        expanded |= older
                to_ids = list(expanded)

            self._apply_zerofy(layer, from_ids, to_ids)

            return pb2.WeightsOperationResponse(
                success=True,
                message=f"Zerofied L{layer_id} from={sorted(set(from_ids))} to={sorted(set(to_ids))}"
            )

        return answer

    def GetWeights(self, request, context):
        print(f"ExperimentServiceServicer.GetWeights({request})")
        answer = pb2.WeightsResponse(success=True, error_message="")

        neuron_id = request.neuron_id
        layer = None
        # ensure ledger-backed components are available
        self._ensure_components()

        try:
            model = self._components.get('model')
            if model is None:
                answer.success = False
                answer.error_messages = 'No model registered'
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
            answer.error_messages = \
                f"Neuron {neuron_id.neuron_id} outside bounds."
            return answer

        if neuron_id.neuron_id < 0:
            # Return all weights
            weights = layer.weight.data.cpu().detach().numpy().flatten()
        else:
            weights = layer.weight[
                neuron_id.neuron_id].data.cpu().detach().numpy().flatten()
        answer.weights.extend(weights)

        return answer

    def GetActivations(self, request, context):
        print(f"ExperimentServiceServicer.GetActivations({request})")
        empty_resp = pb2.ActivationResponse(layer_type="", neurons_count=0)

        try:
            # ensure ledger-backed components are available
            self._ensure_components()

            model = self._components.get('model')
            if model is None:
                return empty_resp
            last_layer = model.layers[-1]
            last_layer_id = int(last_layer.get_module_id())
            if int(request.layer_id) == last_layer_id:
                return empty_resp

            ds = getattr(self._components.get('train_loader'), 'dataset', self._components.get('train_loader'))
            if request.origin == "eval":
                ds = getattr(self._components.get('test_loader'), 'dataset', self._components.get('test_loader'))

            sid = int(request.sample_id)
            if request.sample_id < 0 or request.sample_id >= len(ds):
                raise ValueError(f"No sample id {request.sample_id} for {request.origin}")

            x = _get_input_tensor_for_sample(ds, request.sample_id, getattr(model, 'device', 'cpu'))

            with torch.no_grad():
                intermediaries = {request.layer_id: None}
                model.forward(x, intermediary_outputs=intermediaries)

            if intermediaries[request.layer_id] is None:
                raise ValueError(f"No intermediary layer {request.layer_id}")

            layer = model.get_layer_by_id(request.layer_id)
            layer_type = layer.__class__.__name__
            amap = intermediaries[request.layer_id].squeeze(0).detach().cpu().numpy()
            resp = pb2.ActivationResponse(layer_type=layer_type)

            # At this point we will assume some things, otherwise we keep
            # rechecking same things over and over again.
            C, H, W = 1, 1, 1
            if amap.ndim == 3:  # Conv2d output (C, H, W)
                C, H, W = amap.shape
            elif amap.ndim == 1:  # Linear output (C, ), will use C as features
                C = amap.shape

            resp.neurons_count = C
            for c in range(C):
                vals = amap[c].astype(np.float32).reshape(-1).tolist()
                if not isinstance(vals, list):
                    vals = [vals, ]
                resp.activations.append(
                    pb2.ActivationMap(neuron_id=c, values=vals, H=H, W=W))
            return resp
        except (ValueError, Exception) as e:
            print(
                f"Error in GetActivations: {str(e)}",
                f"Traceback: {traceback.format_exc()}")

        return empty_resp
    
    # ========================================================================
    # Data Service Methods (for weights_studio UI integration)
    # ========================================================================
    
    def _initialize_data_service(self):
        """Initialize data service components using ledger-resolved dataloaders."""
        try:
            self._ensure_components()
            
            train_loader = self._components.get('train_loader')
            test_loader = self._components.get('test_loader')
            
            if train_loader is None or test_loader is None:
                _LOGGER.warning("Cannot initialize data service: dataloaders not in ledger")
                return
            
            # Generate combined dataframe
            train_dataset = getattr(train_loader, 'dataset', train_loader)
            test_dataset = getattr(test_loader, 'dataset', test_loader)
                        
            def _dataset_to_df(dataset, origin):
                """
                Convert a torch/torchvision dataset into a pandas DataFrame that the UI can consume.
                The DataFrame will contain:
                    - sample_id (int)
                    - label      (int)
                    - image      (np.ndarray)   <-- pixel data, shape (28,28) for MNIST
                    - origin     (str)
                """
                # Fast path for torchvision MNIST‑like datasets that expose .data and .targets
                if hasattr(dataset, "data") and hasattr(dataset, "targets"):
                    # `dataset.data` is a torch Tensor of shape (N, 28, 28)
                    # `dataset.targets` is a torch Tensor of shape (N,)
                    images = dataset.data.numpy()          # shape (N, 28, 28)
                    labels = dataset.targets.numpy()       # shape (N,)
                    records = [
                        {
                            "sample_id": i,
                            "label": int(labels[i]),
                            "image": images[i],   # keep as NumPy array – protobuf conversion will flatten it
                            "origin": origin,
                        }
                        for i in range(len(dataset))
                    ]
                else:
                    # Generic fallback – iterate over the dataset items
                    records = []
                    for i in range(len(dataset)):
                        try:
                            item = dataset[i]               # usually (image, label)
                            if isinstance(item, (tuple, list)):
                                img, lbl = item[0], item[-1]
                            else:
                                img, lbl = item, None
                            # Convert the image to a NumPy array (torch Tensor → np.ndarray)
                            if hasattr(img, "numpy"):
                                img_arr = img.numpy()
                            else:
                                img_arr = np.array(img)
                            records.append(
                                {
                                    "sample_id": i,
                                    "label": int(lbl) if lbl is not None else None,
                                    "image": img_arr,
                                    "origin": origin,
                                }
                            )
                        except Exception as e:
                            _LOGGER.warning(f"Failed to convert sample {i}: {e}")
                            continue

                df = pd.DataFrame(records)
                return df

                
            train_df = _dataset_to_df(train_dataset, 'train')
            eval_df = _dataset_to_df(test_dataset, 'eval')
            
            self._all_datasets_df = pd.concat([train_df, eval_df], ignore_index=True)
            _LOGGER.info(f"Created combined DataFrame with {len(self._all_datasets_df)} samples")
            _LOGGER.info(f"DataFrame columns: {list(self._all_datasets_df.columns)}")
            _LOGGER.info(f"DataFrame dtypes: {self._all_datasets_df.dtypes.to_dict()}")
            
            # Initialize agent (import from weights_studio if available)
            try:
                import sys
                sys.path.append('/Users/marcziegler/projects/work/graybox/v5/weights_studio')
                from agent import DataManipulationAgent
                self._agent = DataManipulationAgent(self._all_datasets_df)
                _LOGGER.info("Data service initialized successfully with agent")
            except ImportError as e:
                _LOGGER.warning(f"DataManipulationAgent not available: {e}")
                self._agent = None
                
        except Exception as e:
            _LOGGER.error(f"Data service initialization failed: {e}")
            self._agent = None
    
    def _get_stat_from_row(self, row, stat_name):
        """Extract stat from dataframe row and convert to DataStat message."""
        # Use try-except to safely get the value, avoiding 'in' operator issues with numpy arrays
        try:
            value = row[stat_name]
        except (KeyError, IndexError):
            return None
        
        # Check if value is None
        if value is None:
            return None
        
        # For scalar values, check if NaN
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return pb2.DataStat(
                name=stat_name, type='scalar', shape=[1], value=[float(value)]
            )
        elif isinstance(value, str):
            return pb2.DataStat(
                name=stat_name, type='string', shape=[1], value_string=value
            )
        elif isinstance(value, (list, np.ndarray)):
            flat_value = np.array(value).flatten()
            return pb2.DataStat(
                name=stat_name, type='array',
                shape=list(np.array(value).shape),
                value=flat_value.tolist()
            )
        return None
    
    def ApplyDataQuery(self, request, context):
        """Apply query to filter/sort/manipulate dataset."""
        if self._agent is None:
            self._initialize_data_service()
        
        # If query is empty, just return current dataframe info
        if request.query == "":
            if self._all_datasets_df is None:
                self._initialize_data_service()
            
            if self._all_datasets_df is None:
                return pb2.DataQueryResponse(
                    success=False,
                    message="Data service not available"
                )
            
            total_count = len(self._all_datasets_df)
            discarded_count = len(self._all_datasets_df[
                self._all_datasets_df.get('deny_listed', False) == True
            ]) if 'deny_listed' in self._all_datasets_df.columns else 0
            in_loop_count = total_count - discarded_count
            
            return pb2.DataQueryResponse(
                success=True,
                message=f"Current dataframe has {total_count} samples",
                number_of_all_samples=total_count,
                number_of_samples_in_the_loop=in_loop_count,
                number_of_discarded_samples=discarded_count
            )
        
        if not request.accumulate:
            self._initialize_data_service()
        
        if self._all_datasets_df is None:
            return pb2.DataQueryResponse(
                success=False,
                message="Data service not initialized"
            )
        
        try:
            if request.is_natural_language:
                if self._agent is None:
                    return pb2.DataQueryResponse(
                        success=False,
                        message="Natural language queries require Ollama agent (not available)"
                    )
                
                operation = self._agent.query(request.query)
                self._all_datasets_df = self._agent.apply_operation(self._all_datasets_df, operation)
                message = f"Applied operation: {operation['function']}"
            else:
                self._all_datasets_df = self._all_datasets_df.query(request.query)
                message = f"Query [{request.query}] applied"
            
            total_count = len(self._all_datasets_df)
            discarded_count = len(self._all_datasets_df[
                self._all_datasets_df.get('deny_listed', False) == True
            ]) if 'deny_listed' in self._all_datasets_df.columns else 0
            in_loop_count = total_count - discarded_count
            
            return pb2.DataQueryResponse(
                success=True,
                message=message,
                number_of_all_samples=total_count,
                number_of_samples_in_the_loop=in_loop_count,
                number_of_discarded_samples=discarded_count
            )
        except Exception as e:
            _LOGGER.error(f"Failed to apply query: {e}", exc_info=True)
            return pb2.DataQueryResponse(
                success=False,
                message=f"Failed to apply query: {str(e)}"
            )
    
    def GetDataSamples(self, request, context):
        """Retrieve samples with their data statistics."""
        print(f"DEBUG: GetDataSamples called with start_index={request.start_index}, count={request.records_cnt}")
        if self._all_datasets_df is None:
            self._initialize_data_service()
        
        if self._all_datasets_df is None:
            return pb2.DataSamplesResponse(
                success=False,
                message="Data service not available",
                data_records=[]
            )
        
        try:
            if request.start_index < 0 or request.records_cnt <= 0:
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Invalid start_index or records_cnt",
                    data_records=[]
                )
            
            end_index = request.start_index + request.records_cnt
            df_slice = self._all_datasets_df.iloc[request.start_index:end_index]
            
            if df_slice.empty:
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"No samples found at index {request.start_index}",
                    data_records=[]
                )
            
            self._ensure_components()
            train_loader = self._components.get('train_loader')
            test_loader = self._components.get('test_loader')
            
            data_records = []
            for _, row in df_slice.iterrows():
                origin = row.get('origin', 'unknown')
                sample_id = int(row.get('sample_id', 0))
                
                # Get dataset based on origin
                if origin == 'train':
                    dataset = getattr(train_loader, 'dataset', train_loader) if train_loader else None
                elif origin == 'eval':
                    dataset = getattr(test_loader, 'dataset', test_loader) if test_loader else None
                else:
                    continue
                
                if dataset is None:
                    continue
                
                # Build data stats from DataFrame columns
                data_stats = []
                
                # Get stats to retrieve - if empty, get ALL columns
                stats_to_retrieve = request.stats_to_retrieve
                if not stats_to_retrieve:
                    # Get all columns except sample_id
                    stats_to_retrieve = [col for col in df_slice.columns if col != 'sample_id']
                
                for stat_name in stats_to_retrieve:
                    stat = self._get_stat_from_row(row, stat_name)
                    if stat:
                        data_stats.append(stat)
                
                data_records.append(pb2.DataRecord(
                    sample_id=sample_id,
                    data_stats=data_stats
                ))
            
            return pb2.DataSamplesResponse(
                success=True,
                message=f"Retrieved {len(data_records)} data records",
                data_records=data_records
            )
        except Exception as e:
            _LOGGER.error(f"Failed to retrieve samples: {e}", exc_info=True)
            return pb2.DataSamplesResponse(
                success=False,
                message=f"Failed to retrieve samples: {str(e)}",
                data_records=[]
            )
    
    def EditDataSample(self, request, context):
        """Edit sample metadata (tags, deny_listed, etc.)."""
        self._ensure_components()
        
        if request.stat_name not in ["tags", "deny_listed"]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags' and 'deny_listed' stat editing is supported"
            )
        
        if request.type == pb2.SampleEditType.EDIT_ACCUMULATE:
            return pb2.DataEditsResponse(
                success=False,
                message="Accumulate tagging not supported"
            )
        
        train_loader = self._components.get('train_loader')
        test_loader = self._components.get('test_loader')
        
        for sid, origin in zip(request.samples_ids, request.sample_origins):
            dataset = None
            if origin == 'train':
                dataset = getattr(train_loader, 'dataset', train_loader) if train_loader else None
            elif origin == 'eval':
                dataset = getattr(test_loader, 'dataset', test_loader) if test_loader else None
            
            if dataset is None:
                continue
            
            try:
                if request.stat_name == "tags":
                    dataset.set(sid, "tags", request.string_value)
                elif request.stat_name == "deny_listed":
                    dataset.set(sid, "deny_listed", request.bool_value)
            except Exception as e:
                _LOGGER.warning(f"Could not edit sample {sid}: {e}")
        
        # Update dataframe if it exists
        if self._all_datasets_df is not None:
            for sid, origin in zip(request.samples_ids, request.sample_origins):
                mask = (self._all_datasets_df['sample_id'] == sid) & (self._all_datasets_df['origin'] == origin)
                value = request.string_value if request.stat_name == "tags" else request.bool_value
                self._all_datasets_df.loc[mask, request.stat_name] = value
        
        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples"
        )




def serve(threading=True):
    def serving_thread_callback():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=6))
        servicer = ExperimentServiceServicer()
        pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
        server.add_insecure_port('[::]:50051')
        try:
            server.start()
            print("Server started. Press Ctrl+C to stop.")
            server.wait_for_termination()
        except KeyboardInterrupt:
            # Brut force kill this service
            force_kill_all_python_processes()
    if threading:
        training_thread = Thread(target=serving_thread_callback)
        training_thread.start()
    else:
        serving_thread_callback()

if __name__ == '__main__':
    serve()
