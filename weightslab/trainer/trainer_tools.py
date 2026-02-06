import io
import sys
import io
import numpy as np
import torch
import logging
import subprocess
import numpy as np
import weightslab.proto.experiment_service_pb2 as pb2

from PIL import Image
from torchvision import transforms
from typing import List, Tuple, Iterable


# Get Global Logger
logger = logging.getLogger(__name__)


def get_hyper_parameters_pb(
        hype_parameters_desc_tuple: Tuple) -> List[pb2.HyperParameterDesc]:

    hyper_parameters_pb2 = []
    for (label, name, type_, getter) in hype_parameters_desc_tuple:
        hyper_parameter_pb2 = None
        value = getter()
        if type_ == "text":
            hyper_parameter_pb2 = pb2.HyperParameterDesc(
                label=label,
                name=name,
                type=type_,
                string_value=str(value) if value is not None else ""
            )
        else:
            # For numerical values, ensure we pass a float to gRPC to avoid "must be real number" errors
            try:
                if value is None:
                    num_val = 0.0
                elif isinstance(value, (int, float, bool)):
                    num_val = float(value)
                else:
                    # Try parsing string if necessary
                    num_val = float(str(value))
            except (ValueError, TypeError):
                num_val = 0.0

            hyper_parameter_pb2 = pb2.HyperParameterDesc(
                label=label,
                name=name,
                type=type_,
                numerical_value=num_val
            )
        hyper_parameters_pb2.append(hyper_parameter_pb2)

    return hyper_parameters_pb2


def get_neuron_representations(layer) -> Iterable[pb2.NeuronStatistics]:
    tensor_name = 'weight'
    layer_id = layer.get_module_id()
    neuron_representations = []
    for neuron_idx in range(layer.out_neurons):
        # SAFEGUARD: Ensure trackers are available
        if layer.train_dataset_tracker is None:
            continue
        if layer.eval_dataset_tracker is None:
            continue

        # Get neuron stats
        age = int(layer.train_dataset_tracker.get_neuron_age(neuron_idx))
        trate = layer.train_dataset_tracker.get_neuron_triggers(neuron_idx)
        erate = layer.eval_dataset_tracker.get_neuron_triggers(neuron_idx)
        evage = layer.eval_dataset_tracker.get_neuron_age(neuron_idx)

        trate = trate/age if age > 0 else 0
        erate = erate/evage if evage > 0 else 0

        neuron_lr = layer.get_per_neuron_learning_rate(
            neuron_idx,
            is_incoming=False,
            tensor_name=tensor_name
        )

        neuron_representation = pb2.NeuronStatistics(
            neuron_id=pb2.NeuronId(layer_id=layer_id, neuron_id=neuron_idx),
            neuron_age=age,
            train_trigger_rate=trate,
            eval_trigger_rate=erate,
            learning_rate=neuron_lr,
        )
        for incoming_id, incoming_lr in layer.incoming_neuron_2_lr[tensor_name].items():
            neuron_representation.incoming_lr[incoming_id] = incoming_lr

        neuron_representations.append(neuron_representation)

    return neuron_representations


def get_layer_representation(layer) -> pb2.LayerRepresentation:
    layer_representation = None
    layer_id = layer.get_module_id()
    layer_name = layer.__class__.__name__
    layer_type = layer.module_name
    incoming_neurons_count = layer.in_neurons
    neurons_count = layer.out_neurons
    kernel_size = (layer.kernel_size[0] if not isinstance(layer.kernel_size, (int, float)) else layer.kernel_size) if hasattr(layer, 'kernel_size') else None
    stride = (layer.stride[0] if not isinstance(layer.stride, (int, float)) else layer.stride) if hasattr(layer, 'stride') else None

    parameters = {
        'layer_id': layer_id,
        'layer_name': layer_name,
        'layer_type': layer_type,
        'incoming_neurons_count': incoming_neurons_count,
        'neurons_count': neurons_count,
        'kernel_size': kernel_size,
        'stride': stride
    }
    layer_representation = pb2.LayerRepresentation(**parameters)
    if layer_representation is None:
        return None

    layer_representation.neurons_statistics.extend(
        get_neuron_representations(layer))
    return layer_representation


def get_layer_representations(model):
    layer_representations = []
    for layer in model.layers:
        layer_representation = get_layer_representation(layer)
        if layer_representation is None:
            continue
        layer_representations.append(layer_representation)
    return layer_representations

def mask_to_png_bytes(mask, num_classes=21):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    mask = np.squeeze(mask)
    if mask.ndim == 1:
        sz = int(np.sqrt(mask.size))
        if sz * sz == mask.size:
            mask = mask.reshape((sz, sz))
        else:
            raise ValueError(f"Cannot reshape mask of size {mask.size} to square.")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask 2D, got shape {mask.shape}")

    # mask = (mask.astype(np.float32) * (255.0 / (num_classes - 1))).astype(np.uint8)
    mask = mask.astype(np.uint8)

    im = Image.fromarray(mask)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def _class_ids(x, num_classes=None, ignore_index=255):
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        return []
    u = np.unique(x.astype(np.int64))
    u = u[u != int(ignore_index)]
    if num_classes is not None:
        u = u[(u >= 0) & (u < int(num_classes))]
    return [int(v) for v in u.tolist()]

def _labels_from_mask_path_histogram(path, num_classes=None, ignore_index=255):
    with Image.open(path) as im:
        if im.mode not in ("P", "L"):
            im = im.convert("L")
        hist = im.histogram()  # length 256
    ub = 256 if num_classes is None else int(num_classes)
    ids = [i for i, cnt in enumerate(hist[:ub]) if cnt > 0]
    if ignore_index is not None:
        ig = int(ignore_index)
        ids = [i for i in ids if i != ig]
    return ids

def get_data_set_representation(dataset, experiment) -> pb2.SampleStatistics:
    sample_stats = pb2.SampleStatistics()
    # Robustly obtain a dataset length even when 'dataset' may be a ledger Proxy
    def _safe_dataset_length(ds):
        # Try len(ds) first (Proxy implements __len__ when underlying set)
        try:
            return len(ds)
        except Exception:
            pass

        # Try common wrapped attributes but guard against Proxy AttributeError
        for attr in ('wrapped_dataset', 'dataset', 'wrapped'):
            try:
                wrapped = getattr(ds, attr)
            except Exception:
                wrapped = None
            if wrapped is not None:
                try:
                    return len(wrapped)
                except Exception:
                    # try inspect records
                    try:
                        return len(list(getattr(wrapped, 'as_records')()))
                    except Exception:
                        try:
                            return len(list(getattr(ds, 'as_records')()))
                        except Exception:
                            pass

        # Last resort: try to iterate as_records on ds
        try:
            recs = ds.as_records()
            return len(list(recs))
        except Exception:
            return 0

    sample_stats.sample_count = _safe_dataset_length(dataset)

    raw_ds_task_type = getattr(dataset, "task_type", None)
    raw_exp_task_type = getattr(experiment, "task_type", None)

    task_type = raw_ds_task_type or raw_exp_task_type or "classification"
    # ensure it's a plain string for protobuf
    if isinstance(task_type, bytes):
        task_type = task_type.decode("utf-8", "ignore")
    else:
        task_type = str(task_type)

    sample_stats.task_type = task_type

    raw_ds_task_type = getattr(dataset, "task_type", None)
    raw_exp_task_type = getattr(experiment, "task_type", None)

    task_type = raw_ds_task_type or raw_exp_task_type or "classification"
    # ensure it's a plain string for protobuf
    if isinstance(task_type, bytes):
        task_type = task_type.decode("utf-8", "ignore")
    else:
        task_type = str(task_type)

    sample_stats.task_type = task_type

    ignore_index = getattr(dataset, "ignore_index", 255)
    num_classes  = getattr(dataset, "num_classes", getattr(experiment, "num_classes", None))

    # Safely iterate dataset records; if as_records isn't available or dataset is a placeholder
    # fall back to an empty iterator.
    try:
        records_iter = dataset.as_records()
    except Exception:
        records_iter = []

    for sample_id, row in enumerate(records_iter):
        loss = row.get('prediction_loss', -1)
        if not isinstance(loss, dict):
            loss = {'loss': loss}
        record = pb2.RecordMetadata(
            sample_id=row.get('sample_id', sample_id),
            sample_last_loss=float(row.get('prediction_loss', -1)),
            sample_discarded=bool(row.get('deny_listed', False)),
            task_type=sample_stats.task_type,
        )

        task_type = sample_stats.task_type
        if task_type == "segmentation":
            label = row.get("target")
            if isinstance(label, str):
                target_list = _labels_from_mask_path_histogram(label, num_classes, ignore_index)
            else:
                target_list = _class_ids(label, num_classes, ignore_index)
            pred_list = _class_ids(row.get("prediction_raw"), num_classes, ignore_index)
        else:
            target = row.get("label", row.get("target", -1))
            pred   = row.get("prediction_raw", -1)
            target_list = [int(target)] if not isinstance(target, (list, np.ndarray)) else [int(np.array(target).item())]
            pred_list   = [int(pred)]   if not isinstance(pred, (list, np.ndarray))   else [int(np.array(pred).item())]
        record.sample_label.extend(target_list)
        record.sample_prediction.extend(pred_list)

        sample_stats.records.append(record)
    return sample_stats


def load_raw_image(dataset, index):
    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    if hasattr(wrapped, "images") and isinstance(wrapped.images, list):
        img_path = wrapped.images[index]
        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, "files") and isinstance(wrapped.files, list):
        img_path = wrapped.files[index]
        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, "data"):
        np_img = wrapped.data[index]
        if hasattr(np_img, 'numpy'):
            np_img = np_img.numpy()
        if np_img.ndim == 2:
            return Image.fromarray(np_img.astype(np.uint8), mode="L")
        elif np_img.ndim == 3:
            return Image.fromarray(np_img.astype(np.uint8), mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape: {np_img.shape}")

    elif hasattr(wrapped, "samples") or hasattr(wrapped, "imgs"):
        if hasattr(wrapped, "samples"):
            img_path, _ = wrapped.samples[index]
        else:
            img_path, _ = wrapped.imgs[index]
        img = Image.open(img_path)
        return img.convert("L") if img.mode in ["1", "L", "I;16", "I"] else img.convert("RGB")

    else:
        raise ValueError("Dataset type not supported for raw image extraction.")

def _get_input_tensor_for_sample(dataset, sample_id, device):
    if hasattr(dataset, "_getitem_raw"):
        tensor, _, _ = dataset._getitem_raw(id=sample_id)
    else:
        tensor, _ = dataset[sample_id]

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)
    return tensor

def process_sample(sid, dataset, do_resize, resize_dims, experiment):
    try:
        if hasattr(dataset, "_getitem_raw"):
            tensor, idx, label = dataset._getitem_raw(id=sid)
        else:
            tensor, idx, label = dataset[sid]

        if isinstance(tensor, torch.Tensor):
            img = tensor.detach().cpu()
        else:
            img = torch.tensor(tensor)

        if img.ndim == 3:
            pil_img = transforms.ToPILImage()(img)
        elif img.ndim == 2:
            pil_img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
        else:
            raise ValueError("Unknown image shape.")

        if do_resize:
            pil_img = pil_img.resize(resize_dims, Image.BILINEAR)

        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        transformed_bytes = buf.getvalue()

        try:
            raw = load_raw_image(dataset, sid)
            if do_resize:
                raw = raw.resize(resize_dims, Image.BILINEAR)
            raw_buf = io.BytesIO()
            raw.save(raw_buf, format='PNG')
            raw_bytes = raw_buf.getvalue()
        except Exception:
            raw_bytes = transformed_bytes

        task_type = getattr(experiment, "task_type", getattr(dataset, "task_type", "classification"))

        cls_label = -1
        mask_bytes = b""
        pred_bytes = b""

        if task_type == "classification":
            if isinstance(label, (list, np.ndarray)):
                cls_label = int(np.array(label).item())
            elif hasattr(label, 'cpu'):
                cls_label = int(np.array(label.cpu()).item())
            else:
                cls_label = int(label)
        elif task_type == "segmentation":
            num_classes = getattr(dataset, "num_classes", 21)
            try:
                mask_bytes = mask_to_png_bytes(label, num_classes=num_classes)
            except Exception:
                mask_bytes = mask_to_png_bytes(label)

            try:
                if hasattr(dataset, "get_prediction_mask"):
                    pred_mask = dataset.get_prediction_mask(sid)
                    if pred_mask is not None:
                        pred_bytes = mask_to_png_bytes(pred_mask, num_classes=num_classes)
            except Exception:
                pred_bytes = b""

        return sid, transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"GetSamples({sid}) failed: {e}")
        return (sid, None, None, -1, b"", b"")

def force_kill_all_python_processes():
    """
    Tente de tuer TOUS les processus python en cours d'exécution sur la machine.
    *** ATTENTION : UTILISER AVEC EXTRÊME PRÉCAUTION ! ***
    """
    logger.warning("WARNING: Attempting to kill all Python processes. This could affect other applications.")

    if sys.platform.startswith('win'):
        # Windows : Utilise taskkill pour tuer tous les processus 'python.exe'
        try:
            # /F : Force la terminaison
            # /IM : Spécifie le nom de l'image (python.exe)
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], check=True)
            logger.info("All Python processes (Windows) have been terminated.")
        except subprocess.CalledProcessError as e:
            # Cela arrive si aucun processus python n'est trouvé
            logger.warning(f"No Python processes found or error during termination: {e}")

    elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        # Linux/macOS : Utilise pkill avec SIGKILL (-9) pour les processus 'python' ou 'python3'
        try:
            # pgrep trouve les PIDs des processus nommés 'python' et pkill envoie le signal 9 (SIGKILL)
            # -f : recherche le pattern dans la ligne de commande complète (y compris les arguments)
            subprocess.run(['pkill', '-9', '-f', 'python'], check=True)
            logger.info("All Python processes (Unix/Linux/macOS) have been terminated.")
        except subprocess.CalledProcessError as e:
            # Cela arrive si aucun processus python n'est trouvé
            logger.warning(f"No Python processes found or error during termination: {e}")

    else:
        logger.error(f"Operating system '{sys.platform}' not supported for forced shutdown.")
