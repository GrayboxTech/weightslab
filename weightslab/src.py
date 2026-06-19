"""Public WeightsLab SDK entry points.

This module provides the user-facing helpers that are re-exported at package
level (for example, ``weightslab.watch_or_edit`` and ``weightslab.signal``).
"""
import gc
import os
import sys
import time
import types
import ctypes
import logging
import inspect
import tempfile
import functools
import threading
import traceback
import numpy as np
import torch as th

from tqdm import tqdm
from typing import Callable, Optional, Any

from weightslab.backend.dataloader_interface import DataLoaderInterface
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.backend.optimizer_interface import OptimizerInterface
from weightslab.backend.ledgers import DEFAULT_NAME, get_checkpoint_manager, get_model, get_dataloader, get_dataframe, get_optimizer, register_hyperparams, watch_hyperparams_file, get_hyperparams, register_logger, get_logger, register_signal, get_signal, list_models
from weightslab.backend.model_interface import ModelInterface
from weightslab.trainer.trainer_services import grpc_serve
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.utils.logs import set_log_directory
from weightslab.utils.tools import detach_to_cpu
from weightslab.backend.logger import LoggerQueue
from weightslab.backend.cli import cli_serve
from weightslab.backend import ledgers
from weightslab.backend.ledgers import register_signal
from weightslab.components.global_monitoring import pause_controller as pause_ctrl, get_active_origin


def _rebind_caller_local(original_obj: Any, new_obj: Any) -> None:
    """CPython-specific: find every local variable in the *caller's* caller frame
    that points to *original_obj* and rebind it to *new_obj* in-place.

    This lets ``wl.watch_or_edit(parameters, ...)`` (without capturing the return
    value) transparently replace ``parameters`` with the returned Proxy in the
    calling scope.  Silently does nothing on non-CPython runtimes.
    """
    try:
        # frame 0 = _rebind_caller_local
        # frame 1 = watch_or_edit  (or whatever internal caller)
        # frame 2 = user code
        frame = sys._getframe(2)
        changed = False
        for name, val in list(frame.f_locals.items()):
            if val is original_obj:
                frame.f_locals[name] = new_obj
                changed = True
        if changed:
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(frame), ctypes.c_int(0)
            )
    except Exception:
        pass
from weightslab.backend.ledgers import list_models, get_model as _gm
from tqdm import tqdm


# Get global logger
logger = logging.getLogger(__name__)
# Get global dataframe proxy (auto-updated when ledger registers real manager)
DATAFRAME_M = None
# Global registry for custom signals
_REGISTERED_SIGNALS = {}

# Evaluation function registered via the @wl.eval_fn decorator.
_REGISTERED_EVAL_FN: Optional[Any] = None
_EVAL_WORKER_LOCK = threading.Lock()
_EVAL_WORKER_THREAD: Optional[threading.Thread] = None


class SignalContext:
    """
    Unified context object for WeightsLab signals.
    Carries all available metadata for a single sample during computation.
    """
    def __init__(self, sample_id, dataframe, data=None, subscribed_value=None, logger=None, origin=None):
        self.sample_id = sample_id
        self.dataframe = dataframe
        self.data = data
        self.subscribed_value = subscribed_value
        self.logger = logger
        self.origin = origin

    @property
    def image(self) -> Optional[np.ndarray]:
        """
        Standardized access to image data.
        Automatically converts 'ctx.data' (tensor, array, or path) to an HWC uint8 numpy image.

        Supports:
            - PyTorch Tensors (any device)
            - NumPy Arrays (CHW or HWC)
            - Scaling from [0, 1] to [0, 255]
        """
        if self.data is None:
            return None

        # 1. Handle Tensors & Base Conversion
        img = self.data
        if hasattr(img, "cpu") and hasattr(img, "numpy"):
            img = img.detach().cpu().numpy()

        img_np = np.asanyarray(img)

        # 2. Basic Shape Normalization
        # If it's a single-channel image (H, W) -> (H, W, 1)
        if img_np.ndim == 2:
            img_np = img_np[:, :, np.newaxis]

        # 3. Transpose check: (C, H, W) -> (H, W, C)
        # We assume if the first dim is 1 or 3 and it's much smaller than others, it's CHW
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3] and img_np.shape[0] < img_np.shape[1]:
            img_np = img_np.transpose(1, 2, 0)

        # 4. Data Type & Scaling
        if np.issubdtype(img_np.dtype, np.floating):
            # Safe scale [0, 1] -> [0, 255]
            if img_np.max() <= 1.05:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        return img_np

    @property
    def points(self) -> Optional[np.ndarray]:
        """
        Standardized access to point cloud data.
        Returns (N, 3) or (N, 4) numpy array if ctx.data represents a point cloud.
        """
        if self.data is None:
            return None

        data = self.data
        if hasattr(data, "cpu") and hasattr(data, "numpy"):
            data = data.detach().cpu().numpy()

        arr = np.asanyarray(data)

        # Heuristic for point cloud: 2D array where last dim is 3 (XYZ) or 4 (XYZI)
        if arr.ndim == 2 and arr.shape[1] in [3, 4]:
            return arr

        return None

    @property
    def is_static(self) -> bool:
        """True if running in pre-computation/static mode."""
        return self.data is not None

    @property
    def is_dynamic(self) -> bool:
        """True if running during training (triggered by a metric)."""
        return self.subscribed_value is not None


# #####################################################################################################################
# WEIGHTSLAB INTERNAL FUNCTIONS FOR LOGGING, SIGNAL EXTRACTION, WRAPPING, ETC. (not typically called directly by users)
# #####################################################################################################################

def _update_log_directory(new_log_dir: str):
    """
        Move the current log file to a new directory and update the file handler.

        This is useful for setting a user-specified log directory after initial
        setup with a temporary file. The function will:
        - Move the existing log file to the new directory (if it exists)
        - Update the logging FileHandler to point to the new location
    """

    # Update logging directory to use root_log_dir after parameters registration
    hp = get_hyperparams()
    try:
        set_log_directory(str(hp.get('root_log_dir', new_log_dir)))
    except Exception as e:
        logger.debug(f"Could not update log directory: {e}")


def _get_age(self):
    return self.current_step


def _get_step(step: int | None = None) -> int:
    """
        Attempt to get the current training step from the model in the ledger, if available. This is used for logging signals with the correct global step.
        The function will try multiple approaches to find a valid step value:
        1. Try to get the model from the ledger and access `get_age()` if
              available (preferred for models that track age/step internally).
        2. If `get_age()` is not available, try to access `current_step` attribute on the model (common pattern for external checkpoint managers).
        3. If neither is available, fall back to 0 or the provided step argument
                and add a `current_step` attribute to the model for future tracking.

        TODO (GP): Still not sure because if train and loss computed, then model age at loss value is currently model age minus 1. However, if we start first by evaluation, here model age is 0 as not trained yet. ? To check
    """
    # Fallback: if get_model() failed (e.g. ambiguity), try to find a valid model
    m = None
    full_list = list_models()
    if full_list:
        # Prefer "experiment" or "main" or the first one
        if 'experiment' in full_list:
            m = get_model('experiment')
        elif 'main' in full_list:
            m = get_model('main')
        else:
            m = get_model(full_list[0])

    if m is not None:
        # Safe attribute access (handle Proxy returning None for missing attr)
        if hasattr(m, 'get_age'):
            val = m.get_age() -1  # At this point, model already saw one batch, except if we started by evaluation
            if val is not None:
                step = max([int(val), 0])  # Use age-1 as step to reflect completed step; ensure non-negative

        elif hasattr(m, 'current_step'):
            val = m.current_step

            if val is not None:
                step = max([int(val), 0])  # Use current_step-1 as step to reflect completed step; ensure non-negative

            elif step is not None:
                # step = step # fallback to provided step
                m.current_step = step  # add current_step attribute to model for future tracking

            m.get_age = types.MethodType(_get_age, m)  # To make a proper bound method so `self` is passed correctly, we use types.MethodType

        elif step is not None:
            # If model doesn't have current_step, force it to 0 or try to infer from checkpoint manager
            m.current_step = step  # add current_step attribute to model for future tracking
            m.get_age = types.MethodType(_get_age, m)  # To make a proper bound method so `self` is passed correctly, we use types.MethodType

    return step


def _extract_scalar_from_tensor(batch_scalar: th.Tensor | np.ndarray, out: th.Tensor | np.ndarray = None, ids: th.Tensor = None) -> tuple[float | None, th.Tensor | np.ndarray | None]:
    # extract scalar
    scalar = None
    try:
        # 1. Use manual batch scalar if provided (preferred for complex loss outputs)
        if batch_scalar is not None:
            if isinstance(batch_scalar, th.Tensor):
                batch_scalar = batch_scalar.detach().cpu()
                if batch_scalar.ndim == 0:
                    scalar = float(batch_scalar.item())
                else:
                    scalar = float(batch_scalar.mean().item())
            elif isinstance(batch_scalar, (dict, list)):
                # If it's already a dict/list, we assume it's pre-processed or a multi-signal object
                # Attempt to extract a mean if it's a list of numbers, otherwise skip scalar extraction
                try:
                    import numpy as _np
                    _tmp = _np.array(batch_scalar)
                    if _tmp.dtype.kind in 'fiu':
                        scalar = float(_tmp.mean())
                        if not isinstance(batch_scalar, dict):
                            batch_scalar = _tmp
                except Exception:
                    pass
            else:
                try:
                    import numpy as _np
                    _tmp = _np.array(batch_scalar)
                    scalar = float(_tmp.mean())
                    batch_scalar = _tmp
                except Exception:
                    pass
            # Merged batch scalar with ids
            if isinstance(batch_scalar, (th.Tensor, np.ndarray)) and ids is not None and len(batch_scalar) == len(ids):
                batch_scalar = {ids[i].item() if isinstance(ids, th.Tensor) else ids[i]: batch_scalar[i].item() for i in range(len(batch_scalar))}
        # 2. Otherwise fall back to extracting from 'out'
        elif out is not None:
            if isinstance(out, th.Tensor):
                batch_scalar = out.detach().cpu()
                if batch_scalar.ndim == 0:
                    scalar = float(batch_scalar.item())
                else:
                    scalar = float(batch_scalar.mean().item())
            else:
                try:
                    import numpy as _np
                    batch_scalar = _np.array(out)
                    scalar = float(batch_scalar.mean())
                except Exception:
                    pass
            # Merged batch scalar with ids
            batch_scalar = {ids[i].item() if isinstance(ids, th.Tensor) else ids[i]: batch_scalar[i].item() for i in range(len(batch_scalar))}
    except Exception:
        pass

    return scalar, batch_scalar


def _log_signal(scalar: float, signal_per_sample: dict, reg_name: str, step: int = 0, **kwargs) -> None:
    """
        Log the given scalar signal to the registered logger in the ledger, if available and logging is enabled.

        Args:
            scalar (float): The scalar value to log.
            signal_per_sample (dict): A dictionary containing per-sample signals.
            reg_name (str): The registration name of the signal, used as the key in logging.
            step (int, optional): The current training step to log as global_step. Defaults to 0.
            kwargs (dict, optional): Additional keyword arguments that may contain logging preferences. Defaults to {}.
    """
    # Check if logging is enabled
    if not kwargs.get('log', True):
        return

    # log if requested and logger present
    if scalar is not None:
        try:
            # try to get a ledger-registered logger
            logger = None
            try:
                logger = get_logger()
            except Exception:
                try:
                    logger = get_logger('main')
                except Exception:
                    logger = None

            if logger is not None and hasattr(logger, 'add_scalars'):
                # Add to logger
                logger.add_scalars(
                    reg_name,
                    {reg_name: scalar},
                    global_step=step,
                    signal_per_sample=signal_per_sample,
                    aggregate_by_step=kwargs.get('per_sample', True)  # Aggregate per-sample signals by step for logging if per_sample is True,
                )
        except Exception:
            traceback.print_exc()
            pass


def _update_log_directory(new_log_dir: str):
    """
        Move the current log file to a new directory and update the file handler.

        This is useful for setting a user-specified log directory after initial
        setup with a temporary file. The function will:
        - Move the existing log file to the new directory (if it exists)
        - Update the logging FileHandler to point to the new location
    """

    # Update logging directory to use root_log_dir after parameters registration
    hp = get_hyperparams()
    try:
        set_log_directory(str(hp.get('root_log_dir', new_log_dir)))
    except Exception as e:
        logger.debug(f"Could not update log directory: {e}")



def _move_to_cpu(value: Any) -> Any:
    """Recursively move tensor containers to CPU."""
    if isinstance(value, th.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _move_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(v) for v in value)
    return value


def _release_gpu_resources() -> None:
    """Best-effort migration of tracked Torch objects to CPU and CUDA cache cleanup."""
    try:
        model_names = list_models() or []
    except Exception:
        model_names = []

    for model_name in model_names:
        try:
            model_obj = get_model(model_name)

            # Try direct module first
            if hasattr(model_obj, 'to') and callable(getattr(model_obj, 'to')):
                model_obj.to('cpu')

            # Fallback for wrappers exposing an inner model/module
            inner_model = getattr(model_obj, 'model', None)
            if inner_model is not None and hasattr(inner_model, 'to') and callable(getattr(inner_model, 'to')):
                inner_model.to('cpu')

            module = getattr(model_obj, 'module', None)
            if module is not None and hasattr(module, 'to') and callable(getattr(module, 'to')):
                module.to('cpu')
        except Exception as e:
            logger.debug(f"Could not move model '{model_name}' to CPU: {e}")

    try:
        optimizer = get_optimizer()
        if optimizer is not None and hasattr(optimizer, 'state'):
            for _, state in optimizer.state.items():
                if isinstance(state, dict):
                    for key, value in state.items():
                        state[key] = _move_to_cpu(value)
    except Exception as e:
        logger.debug(f"Could not move optimizer state to CPU: {e}")

    # Clean cached data and free memory
    try:
        gc.collect()
    except Exception:
        pass

    cuda_module = getattr(th, 'cuda', None)
    if cuda_module is not None:
        try:
            cuda_initialized = bool(
                hasattr(cuda_module, 'is_initialized') and cuda_module.is_initialized()
            )
        except Exception:
            cuda_initialized = False

        # Important: avoid creating a fresh CUDA context just for cleanup,
        # which can reserve baseline VRAM in idle keep-serving mode.
        if cuda_initialized:
            try:
                cuda_module.empty_cache()
            except Exception as e:
                logger.debug(f"Could not empty CUDA cache: {e}")
            try:
                cuda_module.ipc_collect()
            except Exception:
                pass


@functools.lru_cache(maxsize=128)
def _fwd_params(fn):
    """Return the frozenset of parameter names accepted by *fn*. Cached per function."""
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (ValueError, TypeError):
        return frozenset()

_WL_KWARGS = ('flag', 'batch_ids', 'group_id', 'preds', 'signals', 'targets')


def wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw):
    """
        Wrapper for forward methods to log and save statistics.
        Args:
            original_forward (Callable): The original forward method.
            kwargs (dict): The keyword arguments passed to the forward method.
            reg_name (str): The registration name of the signal.
        Returns:
            The output of the original forward method.
    """

    # Pop WL-specific kwargs only when original_forward doesn't accept them,
    # so that forwards that declare e.g. batch_ids/preds receive them as usual.
    fwd_params = _fwd_params(original_forward)
    wl_kw = {k: kw.pop(k) for k in _WL_KWARGS if k in kw and k not in fwd_params}

    # Extract torch function parameters
    _ = wl_kw.get('flag')
    preds_raw = a[0] if len(a) > 0 else None

    # User parameters
    batch_ids = wl_kw.get('batch_ids')
    group_ids = wl_kw.get('group_id')
    batch_scalar = wl_kw.get('signals')
    preds = wl_kw.get('preds')
    targets = wl_kw.get('targets') if 'targets' in wl_kw else None

    # Make sure we always have targets if they are passed positionally (for backward compatibility), but allow them to be overridden by keyword if both are present
    targets = a[1] if len(a) > 1 and targets is None else targets

    # Original forward of the signal
    out = original_forward(*a, **kw)

    # discarded samples/tainted groups from the loss tensor.
    origin = kw.get('origin') or kwargs.get('origin') or get_active_origin()

    if origin and batch_ids is not None and hasattr(out, 'device') and out.ndim > 0:
        try:
            # Multi-sample Group Masking
            if group_ids is not None:
                mask = get_active_group_mask(group_ids, origin).to(out.device)
                if len(mask) == len(out):
                    out = out * mask

            # Per-sample Individual Masking
            else:
                mask = get_active_sample_mask(batch_ids, origin).to(out.device)
                if len(mask) == len(out):
                    out = out * mask
        except Exception as e:
            logger.debug(f"Automatic backend discard masking failed: {e}")

    # Per-instance handling: extract instance values + batch_idx mapping
    # and save per-annotation to dataframe. `out` may be a dict
    # {'instance', 'sample', 'batch'} or a flat tensor of instance values.
    per_instance = kwargs.get('per_instance', False)
    instance_values = None
    instance_batch_idx = None
    if per_instance:
        # Resolve instance tensor (dict from PerInstanceDetectionLoss, or flat tensor from PerInstanceIoU)
        if isinstance(out, dict) and 'instance' in out:
            instance_values = out['instance']
        else:
            instance_values = out

        if instance_batch_idx is None and 'batch_idx' in kw:
            instance_batch_idx = kw['batch_idx']
        elif instance_batch_idx is None and targets is not None and isinstance(targets, list):
            instance_batch_idx = [i for i, tars in enumerate(targets) for _ in tars]  # Auto determine batch_idx from targets if not explicitly provided (assumes targets is list of lists of annotations)
        else:
            instance_batch_idx = ledgers.get_dataframe()._df.loc[batch_ids].index.get_level_values(1).tolist()  # Query directly instance_ids related and ordered to the samples_ids in the batch
            batch_ids = ledgers.get_dataframe()._df.loc[batch_ids].index.get_level_values(0).tolist()

    # If output is a dict (from PerInstanceDetectionLoss), pick 'sample'
    # for downstream per-sample logging while keeping the original dict for
    # the caller (so they can still use out['batch'] for backward).
    out_original = out
    if isinstance(out, dict):
        if 'sample' in out:
            out = out['sample']
        elif 'instance' in out:
            out = out['instance']

    if kwargs.get('per_sample', False) and not isinstance(out, dict):
        if hasattr(out, 'ndim') and out.ndim > 1:
            out = out.mean(dim=tuple(range(1, out.ndim)))  # Reduce to [B,]0

    # Extract scalar from tensor
    scalar, batch_scalar = _extract_scalar_from_tensor(batch_scalar, out, batch_ids)

    # Log if requested
    step = _get_step()

    # Save per-instance values to dataframe with annotation_id
    if per_instance and instance_values is not None:
        try:
            save_instance_signals(
                signals={reg_name: instance_values},
                batch_ids=batch_ids,
                batch_idx=instance_batch_idx,
                targets=targets,
                step=step,
                log=False,  # already logged sample-level above
            )
        except Exception as e:
            traceback.print_exc() if os.environ.get('WEIGHTSLAB_LOG_LEVEL') == 'DEBUG' else None
            logger.debug(f"Per-instance signal save failed for {reg_name}: {e}")
    else:
        _log_signal(scalar, batch_scalar, reg_name, step=step, **kwargs)

    # CHECK FOR SUBSCRIBERS (Dynamic Signals)
    # Allows @wl.signal(subscribe_to="metric_name")
    dynamic_updates = {}
    ids_np = None
    if batch_ids is not None:
        if hasattr(batch_ids, 'detach'):
            ids_np = batch_ids.detach().cpu().numpy()
        else:
            try:
                # Handle list of tensors (common in Siamese/multi-output)
                ids_np = np.array([i.detach().cpu().numpy() if hasattr(i, 'detach') else i for i in batch_ids])
            except Exception:
                ids_np = np.asarray(batch_ids)

    if ids_np is not None:
         # Identify Subscribers for this specific signal (reg_name)
         subscribers = []
         for name, func in _REGISTERED_SIGNALS.items():
             meta = getattr(func, '_wl_signal_meta', {})
             if meta.get('subscribe_to') == reg_name:
                 subscribers.append((name, func))

         if subscribers:
             # Resolve generic value vector
             val_tensor = out
             if hasattr(val_tensor, 'detach'):
                  val_tensor = val_tensor.detach().cpu().numpy()
             else:
                  val_tensor = np.asarray(val_tensor)

             # Need (B,) vector for per-sample signals
             val_vec = None
             if val_tensor.ndim > 1:
                  val_vec = val_tensor.mean(axis=tuple(range(1, val_tensor.ndim)))
             elif val_tensor.ndim == 1:
                  val_vec = val_tensor

             if val_vec is not None and len(val_vec) == len(ids_np):
                 # Attempt to get current step for frequency control
                 current_step = 0
                 try:
                     # Quick check for common model names
                     m = None
                     if 'experiment' in list_models(): m = _gm('experiment')
                     elif 'main' in list_models(): m = _gm('main')

                     if m is not None:
                         val = getattr(m, 'current_step', None)
                         if val is not None: current_step = int(val)
                 except Exception:
                     pass

                 # Get Dataframe Proxy for injection
                 try:
                     df_proxy = get_dataframe()
                 except:
                     df_proxy = None

                 # Iterate subscribers
                 for name, func in subscribers:
                     meta = getattr(func, '_wl_signal_meta', {})
                     compute_every = meta.get('compute_every_n_steps', 1)

                     # Frequency Check
                     if current_step % compute_every != 0:
                         continue

                     try:
                         batch_res = {}
                         for i, uid in enumerate(ids_np):
                             # Generic 'value' argument
                             val = float(val_vec[i])

                             # Unified Context Pattern
                             ctx = SignalContext(
                                 sample_id=int(uid),
                                 subscribed_value=val,
                                 logger=ledgers.get_logger(),
                                 dataframe=df_proxy,
                                 origin=kwargs.get('origin', 'train')
                             )
                             try:
                                 res = func(ctx)  # Compute per sample result with unified context
                             except TypeError:
                                 # Fallback for legacy subscriber functions
                                 res = func(sample_id=int(uid), value=val, dataframe=df_proxy)

                             batch_res[uid] = res
                         signal_value = list(batch_res.values())
                         dynamic_updates[name] = signal_value
                         if dynamic_updates and meta.get('log', True):
                             logger.debug(f"Dynamic updates computed for signal '{reg_name}': {list(dynamic_updates.keys())}")
                             _log_signal(sum(signal_value)/len(signal_value), signal_value, name, step=step, **kwargs)  # Log custom subscribed signals
                     except Exception as e:
                         logger.debug(f"Dynamic signal {name} failed: {e}")
                         pass  # User function error, skip

    # Save statistics if requested and applicable.
    # Skip the per-sample save path when per_instance=True — instance values
    # don't map 1:1 to batch_ids, so they're saved separately above via
    # save_instance_signals (keyed by (sample_id, annotation_id)).
    if not per_instance and ((batch_scalar is not None and batch_ids is not None) or dynamic_updates):
        signals = {
            reg_name: list(batch_scalar.values()) if isinstance(batch_scalar, dict) else batch_scalar.detach().cpu().tolist()
        } if batch_scalar is not None else {}
        if dynamic_updates:
            signals.update(dynamic_updates) # Merge dynamic signals

        preds = detach_to_cpu(preds)
        preds_raw = detach_to_cpu(preds_raw)

        # Enqueue signals and data
        save_signals(
            signals=signals,
            batch_ids=batch_ids,
            preds_raw=preds_raw,
            preds=preds,
            targets=targets,
            log=False  # Already logged above, no need to log again in save_signals; set to False to avoid duplicate logging if save_signals is called separately without logging
        )

    # Return the original output (dict for per-instance losses so caller can
    # use out['batch'] for backward, tensor for standard per-sample losses).
    return out_original if isinstance(out_original, dict) else out


# ##############################################################################################################
# USER FUNCTION EXPOSED TO SERVE SIGNALS, TAG SAMPLES, ETC. (can be called from training script to manually set)
# ##############################################################################################################

def watch_or_edit(obj: Callable, obj_name: str = None, flag: str = None, **kwargs) -> None:
    """
    Register or wrap an object so WeightsLab can observe and interact with it.

    Args:
        obj: Object to register. Common values are model, dataloader,
            optimizer, logger, metric/loss object, or hyperparameter config.
        obj_name: Optional explicit object name.
        flag: Registration mode. Supported values include ``model``,
            ``dataloader``/``dataset``/``data``, ``optimizer``, ``logger``,
            ``loss``/``metric``/``signal``, and
            ``hp``/``hyperparams``/``parameters``.
        **kwargs: Additional options forwarded to the corresponding wrapper.

    Custom wrapping parameters:
        forced_model_wrapping: Optional, force to load current model and not from a checkpoint.


    Returns:
        A ledger proxy when available, otherwise the wrapped/registered object.

    Raises:
        ValueError: If ``flag`` is missing or unsupported.
    """

    # Sanity check on obj name attribute
    if not hasattr(obj, '__name__'):
        if obj_name is None and 'name' not in kwargs:
            try:
                obj.__name__ = type(obj).__name__
            except Exception:
                if isinstance(obj, Callable):
                    obj.__name__ = str(time.time())
                    logger.warning(
                        "Warning: Watching or editing anonymous object '" +
                        f"{obj.__name__}'."
                    )
        else:
            if hasattr(obj, '__name__'):
                obj.__name__ = obj_name

    # Register logger in backend for model training
    if ledgers.get_logger() == None:
        LoggerQueue()
        logger.info('LoggerQueue for experiment history has been initialized and registered.')

    # Model
    if 'model' in flag.lower() or (hasattr(obj, '__name__') and 'model' in obj.__name__.lower()):
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_model` will create a Proxy if
        # the name is not yet present.
        _model = get_model()

        # Architecture operations require dependencies to be available.
        # Keep backward-compatible behavior by enabling dependency computation
        # unless the caller explicitly disables it.
        forced_model_wrapping = kwargs.pop('forced_model_wrapping', False)

        # Now construct the wrapper and let it register into the ledger.
        wrapper = ModelInterface(obj, **kwargs)  if forced_model_wrapping or _model == None else _model

        # No rebind here since the model wrapper is designed to be a drop-in replacement for the original model

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return _model if _model != None else wrapper

    # DataLoader
    elif 'data' in flag.lower() or flag.lower() == 'dataset' or flag.lower() == 'dataloader' or (hasattr(obj, '__name__') and 'data' in obj.__name__.lower()):
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_dataloader` will create a Proxy if
        # the name is not yet present.\[]
        try:
            _dataloader = get_dataloader(kwargs.get('loader_name', DEFAULT_NAME))
        except Exception:
            _dataloader = None

        # Auto-inject root_log_dir from hyperparameters if not provided
        if kwargs is None or 'root_log_dir' not in kwargs:
            try:
                from weightslab.backend.ledgers import resolve_hp_name
                hp_name = resolve_hp_name()
                if hp_name:
                    hp_dict = get_hyperparams(hp_name)

                    # Use root_log_dir from hyperparameters if available
                    if isinstance(hp_dict, dict) and 'root_log_dir' in hp_dict:
                        kwargs['root_log_dir'] = hp_dict['root_log_dir']

                    # Update kwargs with relevant hyperparameters
                    kwargs.update(
                        {
                            u:v for u,v in hp_dict.get('data', {}).get(kwargs.get('loader_name', kwargs.get('name')), {}).items() if u not in kwargs
                        }
                    )

                    # Legacy support: set loader_name from name if not provided
                    if 'loader_name' not in kwargs and 'name' in kwargs:
                        kwargs['loader_name'] = kwargs['name']
            except Exception:
                pass  # If we can't get hyperparameters, continue without root_log_dir

        # Now construct the wrapper and let it register into the ledger.
        wrapper = DataLoaderInterface(obj, **kwargs)
        _dataloader.__pl_saved_kwargs = kwargs  # Force pytorch lightning compatibility

        # There is not rebind here because obj can be a dataloader or a dataset

        # Prefer returning the _dataloader (if one exists) so external callers hold
        # a stable reference that will see updates. If no _dataloader was
        # obtainable, return the wrapper itself.
        return _dataloader if _dataloader != None else wrapper

    # Optimizer
    elif 'optimizer' in flag.lower() or (hasattr(obj, '__name__') and 'opt' in obj.__name__.lower()):
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_optimizer` will create a Proxy if
        # the name is not yet present.
        try:
            _opt = get_optimizer()
        except Exception:
            _opt = None

        # Now construct the wrapper and let it register into the ledger.
        wrapper = OptimizerInterface(obj, **kwargs)

        # rebind caller,, i.e., in-place update
        _rebind_caller_local(obj, _opt)

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return _opt

    # Logger
    elif 'logger' in flag.lower() or (hasattr(obj, '__name__') and 'log' in obj.__name__.lower()):
        # Ensure there's a proxy placeholder if callers already requested the logger
        try:
            _logger = get_logger()
        except Exception:
            _logger = None

        # Register the logger into the ledger. This will update any proxy in-place.
        register_logger(obj)

        # rebind caller,, i.e., in-place update
        _rebind_caller_local(obj, _logger)

        # Return a stable handle (proxy) when available, otherwise the registered logger
        return _logger

    # Signals
    # # Loss
    elif 'loss' in flag.lower() or flag.lower() in ('criterion', 'signal', 'signals', 'watch'):
        # derive registration name from second part of flag if provided
        reg_name = kwargs.get('signal_name', kwargs.get('name'))
        if 'log' not in kwargs:
            kwargs['log'] = True

        # decide how to wrap: loss-like (forward) or metric-like (compute)
        # wrap forward
        try:
            if hasattr(obj, 'forward') and callable(getattr(obj, 'forward')):
                original_forward = obj.forward

                # New forward with logging and save stats
                @functools.wraps(original_forward)
                def new_forward(*a, **kw):
                    return wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw)
                obj.forward = new_forward

            # register wrapped signal in ledger
            try:
                register_signal(obj, name=reg_name)
            except Exception:
                pass

            # rebind caller, i.e., in-place update
            _loss = get_signal(reg_name)
            _rebind_caller_local(obj, _loss)

            return _loss

        except Exception:
            # fall back to hyperparams branch if something unexpected
            pass

    # # Metric
    elif 'metric' in flag.lower() or flag.lower() in ('signal', 'signals', 'watch'):
        # derive registration name from second part of flag if provided
        reg_name = kwargs.get('signal_name', kwargs.get('name'))

        # decide how to wrap: loss-like (forward) or metric-like (compute)
        # wrap forward
        try:
            if hasattr(obj, 'compute') and callable(getattr(obj, 'compute')):
                original_forward = obj.compute

                # New forward with logging and save stats
                @functools.wraps(original_forward)
                def new_forward(*a, **kw):
                    return wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw)
                obj.compute = new_forward

            elif hasattr(obj, 'forward') and callable(getattr(obj, 'forward')):
                original_forward = obj.forward

                # New forward with logging and save stats
                @functools.wraps(original_forward)
                def new_forward(*a, **kw):
                    return wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw)
                obj.forward = new_forward

            # register wrapped signal in ledger
            try:
                register_signal(obj, name=reg_name)
            except Exception:
                pass

            # rebind caller, i.e., in-place update
            _metric = get_signal(reg_name)
            _rebind_caller_local(obj, _metric)

            return _metric

        except Exception:
            # fall back to hyperparams branch if something unexpected
            pass

    # Hyper parameters
    else:
        # Support hyperparameters/watchable parameter dicts or YAML paths.
        if flag is None:
            raise ValueError("Obj name should contains at least 'model', 'data', 'optimizer' or 'hp'.")

        fl = flag.lower()
        if fl in ('hp', 'hyperparams', 'params', 'hyperparameters', 'parameters', 'config'):
            # If obj is a string, treat as a file path and start watcher
            try:
                # Initialize CheckpointManager if we have a root dir (fallback to default root)
                root_log_dir = obj.get('root_log_dir') or tempfile.mkdtemp()
                try:
                    # Check if a checkpoint manager is already registered in ledger
                    try:
                        existing_manager = ledgers.get_checkpoint_manager()
                        if existing_manager is not None and not isinstance(existing_manager, ledgers.Proxy):
                            _checkpoint_manager = existing_manager
                            logger.info("Using checkpoint manager from ledger")
                        else:
                            raise KeyError("No manager in ledger")
                    except (KeyError, AttributeError):
                        # Create new manager and register it
                        _checkpoint_manager = CheckpointManager(root_log_dir=root_log_dir)
                        try:
                            ledgers.register_checkpoint_manager(_checkpoint_manager)
                            logger.info("Registered new checkpoint manager in ledger")
                        except Exception:
                            pass
                except Exception:
                    _checkpoint_manager = None

                # Check if hyperparameters are available in checkpoint manager
                checkpoint_hp_loaded = False
                try:
                    chkpt_manager = get_checkpoint_manager()
                    if chkpt_manager != None and isinstance(chkpt_manager, ledgers.Proxy):
                        # Try to get latest hash and load hyperparameters from checkpoint
                        latest_hash = None
                        if hasattr(chkpt_manager, 'current_exp_hash') and chkpt_manager.current_exp_hash:
                            latest_hash = chkpt_manager.current_exp_hash
                        elif hasattr(chkpt_manager, 'manifest') and chkpt_manager.manifest:
                            manifest = chkpt_manager.manifest
                            latest_hash = getattr(manifest, 'latest_hash', None)

                        if latest_hash:
                            checkpoint_data = chkpt_manager.load_checkpoint(
                                exp_hash=latest_hash,
                                load_model=False,
                                load_weights=False,
                                load_config=obj.get('checkpoint_manager', {}).get('load_config', True),
                                force=True,
                                load_data=False
                            )
                            if checkpoint_data.get('config'):
                                config = checkpoint_data['config']
                                register_hyperparams(config)
                                logger.info(f"Loaded hyperparameters from checkpoint {latest_hash[:16]}")
                                checkpoint_hp_loaded = True
                except Exception:
                    pass  # If checkpoint loading fails, proceed with normal registration

                defaults = kwargs.get('defaults', None)
                if not checkpoint_hp_loaded:
                    # Normal registration if no checkpoint hyperparameters were loaded
                    if isinstance(obj, str):
                        path = obj
                        # ensure proxy placeholder exists before registering
                        _hp = get_hyperparams()
                        # register empty/defaults if provided in kwargs
                        if defaults:
                            register_hyperparams(defaults)
                        # start ledger-managed watcher
                        watch_hyperparams_file(path, poll_interval=kwargs.get('poll_interval', 1.0))

                        # Update log directory if root_log_dir provided in hyperparameters or defaults
                        new_log_dir = None
                        if defaults and 'root_log_dir' in defaults:
                            new_log_dir = defaults['root_log_dir']
                        elif 'root_log_dir' in kwargs:
                            new_log_dir = kwargs['root_log_dir']
                        if new_log_dir:
                            _update_log_directory(new_log_dir)

                        # return the ledger handle (proxy or dict)
                        _hp = get_hyperparams()
                        _rebind_caller_local(obj, _hp)
                        return _hp

                    elif isinstance(obj, dict):
                        # ensure proxy placeholder exists before registering
                        get_hyperparams()
                        register_hyperparams(obj)

                        # Update log directory if root_log_dir provided in hyperparameters or defaults
                        new_log_dir = None
                        if defaults and 'root_log_dir' in defaults:
                            new_log_dir = defaults['root_log_dir']
                        elif 'root_log_dir' in kwargs:
                            new_log_dir = kwargs['root_log_dir']
                        if new_log_dir:
                            _update_log_directory(new_log_dir)

                        _hp = get_hyperparams()
                        _rebind_caller_local(obj, _hp)
                        return _hp
                    else:
                        # unsupported type for hp; attempt best-effort registration
                        try:
                            register_hyperparams(dict(obj))

                            # Update log directory if root_log_dir provided in hyperparameters or defaults
                            new_log_dir = None
                            if defaults and 'root_log_dir' in defaults:
                                new_log_dir = defaults['root_log_dir']
                            elif 'root_log_dir' in kwargs:
                                new_log_dir = kwargs['root_log_dir']
                            if new_log_dir:
                                _update_log_directory(new_log_dir)

                            _hp = get_hyperparams()
                            _rebind_caller_local(obj, _hp)
                            return _hp
                        except Exception:
                            raise ValueError('Unsupported hyperparams object; provide dict or YAML path')

                _hp = get_hyperparams()
                _rebind_caller_local(obj, _hp)
                return _hp
            except Exception:
                # bubble up original error
                raise

        raise ValueError(f"Obj name {obj} should contains at least 'model', 'data' or 'optimizer'.")


# ##############################################################################################################
# USER FUNCTION EXPOSED TO SERVE SIGNALS, TAG SAMPLES, ETC. (can be called from training script to manually set)
# ##############################################################################################################

def start_training(timeout: int = None) -> None:
    """Start WeightsLab training mode with optional background services.

    Args:
        timeout: Maximum number of seconds to keep running. If ``None``, runs
            until interrupted.
    """
    if timeout is not None and isinstance(timeout, int) and timeout > 0:
        logger.info(f"Starting WeightsLab training mode with a timeout of {timeout} seconds.")
        time.sleep(timeout)
    pause_ctrl.resume()  # Ensure we're not paused if start_training is called after serve

def serve(serving_cli: bool = False, serving_grpc: bool = False, **kwargs) -> None:
    """Start WeightsLab services.

    Args:
        serving_cli: Start the interactive CLI server.
        serving_grpc: Start the gRPC server.
        **kwargs: Extra server options passed to underlying backends.
    """

    if serving_grpc:
        grpc_serve(**kwargs)

    if serving_cli:
        cli_serve(**kwargs)


def keep_serving(timeout: int = None, release_gpu: bool = True) -> None:
    """Keep process alive while background WeightsLab services are running.

    Args:
        timeout: Maximum number of seconds to keep running. If ``None``, runs
            until interrupted.
        release_gpu: If ``True``, move tracked torch objects to CPU and release
            CUDA cached memory before entering the wait loop.
    """
    if release_gpu:
        _release_gpu_resources()
        logger.info("WeightsLab switched to CPU idle mode for serving.")

    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.info("Timeout reached, stopping WeightsLab services.")
                break
    except KeyboardInterrupt:
        logger.info("Shutting down WeightsLab services.")


def signal(name: str, subscribe_to: str = None, compute_every_n_steps: int = 1, **kwargs):
    """
    Decorator that registers a custom signal function.

    Static signals are typically computed from samples via
    :func:`compute_signals`. Dynamic signals can subscribe to a training metric
    and run every ``compute_every_n_steps``.

    Args:
        name: Public signal name. Defaults to decorated function name.
        subscribe_to: Optional signal/metric name this signal reacts to.
        compute_every_n_steps: Frequency for dynamic computation.
        **kwargs: Additional signal metadata stored in the ledger.

    Returns:
        Callable: A decorator that registers the function and returns it.

    Usage:
        @wl.signal(name="blue_pixels")
        def compute_blue(sample, **kwargs): ...

        @wl.signal(name="weighted_loss", subscribe_to="train_loss", compute_every_n_steps=10)
        def compute_weighted(value, sample_id, **kwargs): ...
    """
    def decorator(func):
        reg_name = name or func.__name__
        if reg_name in _REGISTERED_SIGNALS:
            logger.warning(f"Overwriting already registered signal: {reg_name}")
        _REGISTERED_SIGNALS[reg_name] = func

        # Attach metadata
        func._wl_signal_meta = kwargs
        func._wl_signal_meta['subscribe_to'] = subscribe_to
        func._wl_signal_meta['compute_every_n_steps'] = compute_every_n_steps
        func._wl_signal_name = reg_name

        return func
    return decorator


def compute_signals(dataset_or_loader, origin: str = None, signals: list[str] = None):
    """
    Execute static registered signals over a dataset and persist results.

    Args:
        dataset_or_loader: Dataset, dataloader, or WeightsLab tracked loader.
        origin: Dataset split (for example ``train`` or ``val``). If omitted,
            this is inferred when possible.
        signals: Optional subset of signal names to run. If omitted, all
            registered static signals are used.

    Returns:
        None. Results are upserted into the global dataframe ledger.
    """
    global DATAFRAME_M
    if DATAFRAME_M is None:
        DATAFRAME_M = get_dataframe()

    # Unwrap loader if needed
    dataset = dataset_or_loader
    if hasattr(dataset_or_loader, "tracked_dataset"): # WeightsLab Loader
        dataset = dataset_or_loader.tracked_dataset
    elif hasattr(dataset_or_loader, "dataset"): # Loader
        dataset = dataset_or_loader.dataset

    # GP: Do not unwrap wrapped_dataset (DataSampleTrackingWrapper) blindly,
    # as it holds the 'unique_ids' map when compute_hash=True.
    # Only unwrap if it's a wrapper that doesn't provide IDs but obscures the underlying data?
    # For now, we trust the wrapper if it behaves like a dataset.

    # ... but we might need the raw dataset for specialized methods if the wrapper doesn't delegate.
    # DataSampleTrackingWrapper delegates __getitem__ and __len__, so it's fine.

    # Attempt to resolve origin
    if origin is None:
        if hasattr(dataset, "_dataset_split"):
            origin = dataset._dataset_split
        elif hasattr(dataset, "split"):
            origin = dataset.split
        else:
            origin = "unknown"

    # Resolve signals to run
    signal_fns = {}
    target_names = signals or _REGISTERED_SIGNALS.keys()

    for name in target_names:
        if name in _REGISTERED_SIGNALS:
            func = _REGISTERED_SIGNALS[name]
            # Only run static signals (compute_every is None or 0)
            compute_every = getattr(func, '_wl_signal_meta', {}).get('compute_every', 0)
            if not compute_every:
                signal_fns[name] = func

    if not signal_fns:
        logger.info("No signals to compute.")
        return

    logger.info(f"Computing {len(signal_fns)} signals for {len(dataset)} samples in '{origin}'...")

    batch_updates = []

    # helper for unpacking
    def _get_image(item):
        # Handle tuple (img, label) or dict/object
        if isinstance(item, (tuple, list)):
            return item[0]
        return item

    # Iterate
    # Note: Accessing dataset directly by index is assumed safe
    for i in tqdm(range(len(dataset)), desc=f"Signals [{origin}]"):
        try:
            sample_id = i
            # 1. Hashed ID (compute_hash=True)
            if hasattr(dataset, "unique_ids") and dataset.unique_ids is not None:
                try:
                    sample_id = dataset.unique_ids[i]
                except (IndexError, TypeError):
                    pass
            # 2. Dataset-specific mapping (legacy)
            elif hasattr(dataset, "get_index_from_sample_id"):
                try:
                    val = dataset.get_index_from_sample_id(i)
                    if val is not None:
                        sample_id = val
                except Exception:
                    pass

            # Retrieve raw item
            # We try to get the rawest form possible to avoid transforms if not needed,
            # but for consistency we often just use __getitem__
            if i == 0:
                logger.info(f"DEBUG: compute_signals first sample_id: {sample_id} (type: {type(sample_id)}) for origin: {origin}")

            raw_item = dataset[i]
            input_data = _get_image(raw_item)

            row = {
                "sample_id": str(sample_id),
                "origin": origin,
            }

            for sig_name, sig_func in signal_fns.items():
                try:
                    # Unified Context Pattern
                    ctx = SignalContext(
                        sample_id=sample_id,
                        data=input_data,
                        dataframe=DATAFRAME_M,
                        origin=origin
                    )
                    try:
                        val = sig_func(ctx)
                    except TypeError:
                        # Fallback for legacy static signals
                        val = sig_func(input_data)

                    # Prefix 'signals_' if not already present to group in UI
                    key = sig_name if sig_name.startswith("signals") else f"signals_{sig_name}"
                    row[key] = val
                except Exception as e:
                    logger.debug(f"Static signal {sig_name} failed: {e}")

            batch_updates.append(row)

        except Exception as e:
            logger.error(f"Failed to compute signals for index {i}: {e}")

    # Upsert to Ledger
    if batch_updates:
        import pandas as pd
        df_new = pd.DataFrame(batch_updates)
        logger.info(f"Registering {len(df_new)} signal records to global ledger.")

        # Use upsert with force_flush
        enqueue = getattr(DATAFRAME_M, 'upsert_df', None)
        if callable(enqueue):
            enqueue(df_new, force_flush=True)


def tag_samples(
    sample_ids: list[int],
    tag: str,
    mode: str = 'add'
) -> bool:
    """
    Add or remove tags from samples.

    Args:
        sample_ids: List of sample IDs to tag
        tag: Tag name to add/remove (e.g., 'difficult', 'outlier')
        mode: Operation mode - 'add', 'remove', or 'set' (override all tags)

    Returns:
        bool: True if successful

    Examples:
        >>> # Tag difficult samples
        >>> tag_samples([0, 5, 12], 'difficult', mode='add')

        >>> # Remove outlier tag
        >>> tag_samples([5, 12], 'outlier', mode='remove')

        >>> # Set exclusive tag (removes other tags)
        >>> tag_samples([0], 'validated', mode='set')
    """
    import pandas as pd
    from weightslab.backend.ledgers import get_dataframe

    df_manager = get_dataframe()
    if df_manager is None:
        logger.error("Dataframe manager not initialized. Call this after registering your dataloader.")
        return False

    # Build tag column name
    tag_col = f"{SampleStatsEx.TAG.value}:{tag.strip()}"

    try:
        if mode == 'add':
            # Set tag column to True for specified samples
            rows = [{"sample_id": sid, tag_col: True}
                    for sid in sample_ids]
        elif mode == 'remove':
            # Set tag column to False for specified samples
            rows = [{"sample_id": sid, tag_col: False}
                    for sid in sample_ids]
        elif mode == 'set':
            # Override: first clear all existing tags, then set new one
            # This would require more complex logic - for now just add
            logger.warning("Mode 'set' not fully implemented yet, using 'add' instead")
            rows = [{"sample_id": sid, tag_col: True}
                    for sid in sample_ids]
        else:
            logger.error(f"Invalid mode '{mode}'. Use 'add', 'remove', or 'set'")
            return False

        # Build directly on the (sample_id, annotation_id=0) multi-index: tags and
        # the discarded flag are SAMPLE-LEVEL, so they live on the canonical sample
        # row (annotation_id 0). Never hand upsert a bare single-level frame.
        df_update = pd.DataFrame(rows)
        df_update["annotation_id"] = 0
        df_update = df_update.set_index(["sample_id", "annotation_id"])
        df_manager.upsert_df(df_update, force_flush=True)

        logger.info(f"Tagged {len(sample_ids)} samples with '{tag}' (mode={mode})")
        return True

    except Exception as e:
        logger.error(f"Failed to tag samples: {e}", exc_info=True)
        return False


def register_categorical_tag(name: str, categories: list[str], replace: bool = False) -> list:
    """Declare a categorical (multi-value) tag and its allowed category values.

    Categorical tags hold one string value per sample chosen from a predefined
    set (e.g. ``weather`` -> ``rainy`` / ``sunny`` / ``cloudy``), as opposed to
    boolean tags which are simply present/absent. Declaring the tag up-front lets
    the UI render the full set of choices even before any sample uses a value,
    and ensures the complete category set survives the dataframe/H5 round-trip.

    Args:
        name: Tag name (with or without the ``tag:`` prefix), e.g. ``"weather"``.
        categories: Allowed category values, e.g. ``["rainy", "sunny", "cloudy"]``.
        replace: If True, replace any existing categories; otherwise merge.

    Returns:
        The resulting ordered list of allowed categories (empty on failure).

    Examples:
        >>> wl.register_categorical_tag("weather", ["rainy", "sunny", "cloudy"])
        >>> wl.set_categorical_tag([0, 3], "weather", "rainy")
    """
    from weightslab.backend.ledgers import get_dataframe

    df_manager = get_dataframe()
    if df_manager == None:
        logger.error("Dataframe manager not initialized. Call this after registering your dataloader.")
        return []
    try:
        return df_manager.register_categorical_tag(name, categories, replace=replace)
    except Exception as e:
        logger.error(f"Failed to register categorical tag '{name}': {e}", exc_info=True)
        return []


def set_categorical_tag(sample_ids: list[int], name: str, value: str) -> bool:
    """Set a categorical tag value on samples (e.g. weather='rainy').

    The tag is auto-registered and the value added to its allowed categories if
    not already present. Pass ``value=None`` or ``""`` to clear (unset) the tag.

    Args:
        sample_ids: Sample IDs to update.
        name: Categorical tag name (with or without ``tag:`` prefix).
        value: The category value to set, or None/"" to clear.

    Returns:
        bool: True if successful.
    """
    import pandas as pd
    from weightslab.backend.ledgers import get_dataframe

    df_manager = get_dataframe()
    if df_manager is None:
        logger.error("Dataframe manager not initialized. Call this after registering your dataloader.")
        return False

    pref = f"{SampleStatsEx.TAG.value}:"
    tag_name = name[len(pref):] if str(name).startswith(pref) else str(name).strip()
    tag_col = f"{pref}{tag_name}"
    cleaned = None if value is None or str(value).strip() == "" else str(value).strip()

    try:
        if cleaned is not None:
            df_manager.register_categorical_tag(tag_name, [cleaned])
        rows = [{"sample_id": sid, tag_col: cleaned} for sid in sample_ids]
        # Build directly on the (sample_id, annotation_id=0) multi-index: tags and
        # the discarded flag are SAMPLE-LEVEL, so they live on the canonical sample
        # row (annotation_id 0). Never hand upsert a bare single-level frame.
        df_update = pd.DataFrame(rows)
        df_update["annotation_id"] = 0
        df_update = df_update.set_index(["sample_id", "annotation_id"])
        df_manager.upsert_df(df_update, force_flush=True)
        logger.info(f"Set categorical tag '{tag_name}'={cleaned!r} on {len(sample_ids)} samples")
        return True
    except Exception as e:
        logger.error(f"Failed to set categorical tag '{name}': {e}", exc_info=True)
        return False


def discard_samples(
    sample_ids: list[int],
    discarded: bool = True,
) -> bool:
    """
    Mark samples as discarded (excluded from training) or restore them.

    Args:
        sample_ids: List of sample IDs to discard/restore
        discarded: True to discard, False to restore

    Returns:
        bool: True if successful

    Examples:
        >>> # Discard corrupted samples
        >>> discard_samples([3, 7, 19], discarded=True)

        >>> # Restore previously discarded samples
        >>> discard_samples([3, 7], discarded=False)
    """
    import pandas as pd
    from weightslab.backend.ledgers import get_dataframe

    df_manager = get_dataframe()
    if df_manager is None:
        logger.error("Dataframe manager not initialized. Call this after registering your dataloader.")
        return False

    try:
        rows = [{"sample_id": sid, SampleStatsEx.DISCARDED.value: bool(discarded)}
                for sid in sample_ids]

        # Build directly on the (sample_id, annotation_id=0) multi-index: tags and
        # the discarded flag are SAMPLE-LEVEL, so they live on the canonical sample
        # row (annotation_id 0). Never hand upsert a bare single-level frame.
        df_update = pd.DataFrame(rows)
        df_update["annotation_id"] = 0
        df_update = df_update.set_index(["sample_id", "annotation_id"])
        df_manager.upsert_df(df_update, force_flush=True)

        action = "Discarded" if discarded else "Restored"
        logger.info(f"{action} {len(sample_ids)} samples ")
        return True

    except Exception as e:
        logger.error(f"Failed to update discard status: {e}", exc_info=True)
        return False


def get_samples_by_tag(
    tag: str,
    origin: str = 'train',
    limit: int = None
) -> list[int]:
    """
    Get sample IDs that have a specific tag.

    Args:
        tag: Tag name to search for
        origin: Dataset split ('train', 'eval', etc.). Default: 'train'
        limit: Maximum number of sample IDs to return (None = all)

    Returns:
        List of sample IDs with the specified tag

    Examples:
        >>> # Get all difficult samples
        >>> difficult_ids = get_samples_by_tag('difficult', origin='train')
        >>> print(f"Found {len(difficult_ids)} difficult samples")
    """
    from weightslab.backend.ledgers import get_dataframe

    df_manager = get_dataframe()
    if df_manager is None:
        logger.error("Dataframe manager not initialized.")
        return []

    try:
        tag_col = f"{SampleStatsEx.TAG.value}:{tag.strip()}"
        df = df_manager.get_df_view(origin, limit=limit)

        if tag_col not in df.columns:
            logger.warning(f"Tag '{tag}' not found in {origin} dataset")
            return []

        # Filter samples where tag column is True
        tagged_df = df[df[tag_col] == True]
        sample_ids = tagged_df.index.tolist()

        return sample_ids

    except Exception as e:
        logger.error(f"Failed to get samples by tag: {e}", exc_info=True)
        return []


def get_discarded_samples(
    origin: str = 'train',
    limit: int = None
) -> list[int]:
    """
    Get sample IDs that are marked as discarded.

    Args:
        origin: Dataset split ('train', 'eval', etc.). Default: 'train'
        limit: Maximum number of sample IDs to return (None = all)
    Returns:
        List of discarded sample IDs
    Examples:
        >>> # Get all discarded samples
        >>> discarded_ids = get_discarded_samples(origin='train')
        >>> print(f"Found {len(discarded_ids)} discarded samples")
    """
    from weightslab.backend.ledgers import get_dataframe

    df_manager = get_dataframe()
    if df_manager is None:
        logger.error("Dataframe manager not initialized.")
        return []

    try:
        discard_col = SampleStatsEx.DISCARDED.value
        df = df_manager.get_df_view(origin, limit=limit)

        if discard_col not in df.columns:
            logger.warning(f"Discard column not found in {origin} dataset")
            return []

        # Filter samples where discard column is True
        discarded_df = df[df[discard_col] == True]
        sample_ids = discarded_df.index.tolist()

        return sample_ids

    except Exception as e:
        logger.error(f"Failed to get discarded samples: {e}", exc_info=True)
        return []


def save_signals(
    signals: dict,
    batch_ids: th.Tensor | np.ndarray | list ,
    preds_raw: th.Tensor | np.ndarray | dict = None,
    targets: th.Tensor | np.ndarray | dict = None,
    preds: th.Tensor | np.ndarray | dict = None,
    step: int | None = None,
    log: bool = False
):
    """Save **per-sample** statistics to the tracked dataset.

    This is the *one-value-per-sample* path: every array you pass is indexed by
    ``batch_ids`` and written onto that sample's canonical row — the
    ``(sample_id, annotation_id=0)`` row of the multi-index. Use this for losses,
    metrics, predictions and targets that describe the whole sample (e.g. the
    classification loss of an image, the per-image mAP, the predicted mask).

    For tasks where a single sample has *several* instances (detection boxes,
    segmentation masks, ...) and you want one value **per instance**, use
    :func:`save_instance_signals` instead — it writes to ``annotation_id >= 1``.

    Shapes / formats:
        - ``signals`` values: shape ``(B,)`` (one scalar per sample) or
          ``(B, ...)`` — extra dims are mean-reduced to ``(B,)`` before storage.
        - ``batch_ids``: length ``B``; the i-th entry names the sample the i-th
          row of every other array belongs to. Coerced to ``str`` internally.
        - ``preds`` / ``preds_raw`` / ``targets``: array of length ``B`` (or a
          ``dict`` of such arrays, or a ``list`` of length ``B`` for
          inhomogeneous per-sample shapes). 1-D arrays get a trailing axis.

    Args:
        signals (dict | th.Tensor): ``{name: values}`` where ``values`` is a
            length-``B`` tensor/array (or a bare tensor → stored as
            ``signals//default``). Each becomes a ``signals//<name>`` column.
        batch_ids (th.Tensor | np.ndarray | list): Sample IDs, length ``B``.
        preds_raw (optional): Raw model outputs (e.g. logits), length ``B``.
            Skip it if a watched loss wrapper already saved it.
        targets (optional): Ground-truth targets, length ``B``.
        preds (optional): Post-processed predictions, length ``B``.
        step (int, optional): Training step. Inferred from the training context
            when omitted.
        log (bool, optional): Also push the (mean) scalar to the dashboard
            logger, not only to the per-sample row. Defaults to False.

    Examples:
        Classification — one loss scalar per image::

            for inputs, targets, ids in train_loader:   # ids: sample IDs, len B
                logits = model(inputs)                   # (B, num_classes)
                loss = loss_fn(logits, targets)          # (B,) per-sample loss
                wl.save_signals(
                    signals={"train_loss": loss},        # (B,) -> signals//train_loss
                    batch_ids=ids,
                    preds_raw=logits,                    # (B, num_classes)
                    targets=targets,                     # (B,)
                    step=current_step,
                    log=True,
                )

        Several named per-sample metrics at once::

            wl.save_signals(
                signals={"iou": iou_per_image, "dice": dice_per_image},  # each (B,)
                batch_ids=ids,
            )
    """

    global DATAFRAME_M
    if DATAFRAME_M is None:
        DATAFRAME_M = get_dataframe()

    # Get current model step
    step = _get_step(step=step)

    # Log if requested for each signals
    if log:
        for reg_name, batch_scalar in signals.items():
            # Extract scalar from tensor
            scalar, batch_scalar = _extract_scalar_from_tensor(batch_scalar, ids=batch_ids)

            # Log if requested
            _log_signal(scalar, batch_scalar, reg_name, step=step)

        if batch_ids is None or len(batch_ids) == 0:
            logger.warning("No batch IDs provided for signal saving; skipping logging and saving.")
            return

    # Convert tensors to numpy for lightweight buffering
    if isinstance(batch_ids, th.Tensor):
        batch_ids_np = [str(i) for i in batch_ids.detach().cpu().tolist()]
    else:
        batch_ids_np = [str(i) for i in batch_ids] if batch_ids is not None else None

    # Normalize to np arrays
    def to_numpy(t):
        arr = t.detach().cpu().numpy() if isinstance(t, th.Tensor) else np.asarray(t)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32)
        return arr.astype(np.uint16)

    def normalize(x):
        if x is None:
            return None
        if isinstance(x, list) and isinstance(x[0], list):
            return [np.max(np.array([to_numpy(t) for t in row]), axis=0) for row in x]
        elif isinstance(x, list):
            return [to_numpy(t) for t in x]
        if isinstance(x, th.Tensor):
            return to_numpy(x)
        return None

    def expand_dim(x):
        """Add axis if 1D — skip for lists (inhomogeneous shapes)."""
        if x is None or isinstance(x, list):
            return x
        if x.ndim == 1:
            return x[:, np.newaxis]
        return x

    preds_np     = normalize(preds)
    preds_raw_np = normalize(preds_raw)
    target_np    = normalize(targets)

    # Processing signals
    if isinstance(signals, dict):
        losses_data = {
            'signals//' + k: (lambda arr: arr.mean(axis=tuple(range(1, arr.ndim))) if isinstance(arr, np.ndarray) and arr.ndim > 1 else arr)(
                v.detach().cpu().numpy() if hasattr(v, 'detach') else normalize(v)
            )
            for k, v in signals.items()
        }
    elif signals is not None and isinstance(signals, (th.Tensor, np.ndarray, list)):
        losses_data = {
            "signals//default": (lambda arr: arr.mean(axis=tuple(range(1, arr.ndim))) if isinstance(arr, np.ndarray) and arr.ndim > 1 else arr)(
                signals.detach().cpu().numpy() if hasattr(signals, 'detach') else normalize(signals)
            )
        }
    else:
        losses_data = None

    # Expand dims for 1D arrays (skipped for lists)
    target_np    = expand_dim(target_np)
    preds_np     = expand_dim(preds_np)
    preds_raw_np = expand_dim(preds_raw_np)

    # Enqueue to dataframe manager buffer for efficiency
    DATAFRAME_M.enqueue_batch(
        sample_ids=batch_ids_np,
        preds_raw=preds_raw_np if preds_raw_np is not None else preds_raw,
        preds=preds_np if preds_np is not None else preds,
        targets=target_np if target_np is not None else targets,
        losses=losses_data,
        step=step
    )


def save_instance_signals(
    signals: dict,
    batch_ids: th.Tensor | np.ndarray | list,
    batch_idx: th.Tensor | np.ndarray | list,
    step: int | None = None,
    origin: str | None = None,
    targets: th.Tensor | np.ndarray | dict = None,
    log: bool = False,
):
    """Save **per-instance** (per-annotation) signals to the dataframe.

    This is the *one-value-per-instance* counterpart of :func:`save_signals`.
    A sample can own several instances (detection boxes, segmentation masks,
    keypoints, ...); their values are stored on the instance rows
    ``(sample_id, annotation_id)`` with ``annotation_id >= 1`` (``annotation_id 0``
    stays reserved for the per-sample row written by :func:`save_signals`).

    **Flat, sample-major (Ultralytics) format**

    Instances are passed *flattened across the whole batch*, exactly the layout
    Ultralytics/YOLO uses for a detection batch: all targets of all images are
    concatenated into one ``(num_instances_total, ...)`` tensor, and a companion
    ``batch_idx`` tensor says which image each row belongs to. So you pass the
    Ultralytics ``batch["batch_idx"]`` straight through here:

        - ``signals[name]``: flat tensor of length ``num_instances_total``
          (one scalar per instance, in sample-major order).
        - ``batch_idx``: flat tensor of length ``num_instances_total``; entry
          ``i`` is the *position in* ``batch_ids`` (i.e. the image index within
          the batch, in ``[0, B)``) that instance ``i`` belongs to.
        - ``batch_ids``: the ``B`` sample IDs for the batch (one per image).

    The ``annotation_id`` is derived automatically: instances are numbered in the
    order they appear per sample → the first instance of a sample becomes
    ``annotation_id 1``, the second ``2``, and so on (1-based, since ``0`` is the
    sample row).

    Worked example — ``batch_ids = ["img7", "img3"]`` (B = 2), 5 boxes total::

        # box:        0      1      2      3      4
        batch_idx = [ 0,     0,     1,     1,     1 ]   # boxes 0-1 -> img7, 2-4 -> img3
        ious      = [0.91,  0.62,  0.50,  0.74,  0.30]  # one IoU per box

        wl.save_instance_signals(
            signals={"iou_instance": ious},   # -> signals//iou_instance
            batch_ids=["img7", "img3"],
            batch_idx=batch_idx,
            origin="train",
        )
        # writes:
        #   ("img7", 1)=0.91  ("img7", 2)=0.62
        #   ("img3", 1)=0.50  ("img3", 2)=0.74  ("img3", 3)=0.30

    Typical detection loop using the Ultralytics batch dict directly::

        image, batch_ids, batch = inputs[0], inputs[1], inputs[3]["batch"]
        raw_preds = model(image)
        iou_per_box = compute_iou(raw_preds, batch)            # flat [total_instances]
        wl.save_instance_signals(
            signals={"iou_instance": iou_per_box},
            batch_ids=batch_ids,
            batch_idx=batch["batch_idx"],                      # Ultralytics flat index
            step=current_step,
        )

    Persisting per-instance ground truth — pass ``targets`` as a **nested**
    per-sample list (``targets[s]`` = sample ``s``'s list of instance targets),
    in the same per-sample order ``batch_idx`` implies. It is flattened
    sample-major to align with the instances::

        targets = [                       # batch_ids = ["img7", "img3"]
            [box7_0, box7_1],             # img7's two boxes  -> annotation_id 1, 2
            [box3_0, box3_1, box3_2],     # img3's three boxes -> annotation_id 1, 2, 3
        ]
        wl.save_instance_signals(signals={"iou_instance": ious},
                                 batch_ids=["img7", "img3"],
                                 batch_idx=[0, 0, 1, 1, 1],
                                 targets=targets)

    Args:
        signals (dict): ``{name: values}`` with ``values`` a flat tensor/array of
            shape ``(num_instances_total,)`` (extra dims are mean-reduced). Each
            becomes a ``signals//<name>`` column on the instance rows.
        batch_ids (th.Tensor | np.ndarray | list): The ``B`` sample IDs of the
            batch (one per image). Coerced to ``str`` internally.
        batch_idx (th.Tensor | np.ndarray | list): Flat instance→image map of
            length ``num_instances_total``, values in ``[0, B)`` — the Ultralytics
            ``batch_idx``. Out-of-range entries are skipped.
        step (int, optional): Training step. Inferred from context when omitted.
        origin (str, optional): Dataset split (``"train"``, ``"val"``, ...). When
            omitted, the active training origin is used (fallback ``"train"``).
        targets (optional): Per-instance ground truth to persist alongside the
            signals — a **nested** ``list[B]`` where ``targets[s]`` is sample
            ``s``'s list of instance targets (or a ``dict`` of such). Flattened
            sample-major to match ``batch_idx``.
        log (bool, optional): Also push the per-sample aggregated mean of each
            signal to the dashboard logger. Defaults to False.
    """
    global DATAFRAME_M
    if DATAFRAME_M is None:
        DATAFRAME_M = get_dataframe()

    step = _get_step(step=step)

    # Normalize batch_idx to numpy ints (per-instance → batch-position map)
    if isinstance(batch_idx, th.Tensor):
        batch_idx_np = batch_idx.detach().cpu().numpy().astype(int).flatten()
    else:
        batch_idx_np = np.asarray(batch_idx).astype(int).flatten()

    if len(batch_idx_np) == 0:
        return

    # Build per-instance (sample_id, annotation_id) lists from the flat batch_idx map.
    # batch_idx[i] is the image position (in batch_ids) that instance i belongs to.
    # Both lists have length num_instances_total and are ALIGNED with the flat
    # signal/target order. enqueue_instance_batch zips sample_ids with annotation_ids
    # and requires len(sample_ids) == len(annotation_ids); passing the raw batch_ids
    # (length B) instead would trip its length guard and silently drop everything for
    # any batch with more instances than images.
    #
    # annotation_id is 1-based per sample (instance_id 0 is reserved for the sample
    # row): the k-th instance of a given image becomes annotation_id k (1, 2, ...).
    # Out-of-range batch_idx entries are kept as placeholders (annotation_id 0, which
    # enqueue_instance_batch skips) so the flat signal index stays aligned.
    def _coerce_sid(x):
        # Normalize a sample id (tensor/np scalar/float) to a clean string,
        # e.g. tensor(1) -> "1", 1.0 -> "1", "0" -> "0".
        if hasattr(x, "item"):
            try:
                x = x.item()
            except Exception:
                pass
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        return str(x)

    num_images = len(batch_ids)
    instance_sample_ids: list = []
    instance_annotation_ids: list[int] = []
    _per_sample_count: dict = {}
    for img_pos in batch_idx_np.tolist():
        if 0 <= img_pos < num_images:
            sid = _coerce_sid(batch_ids[img_pos])
            _per_sample_count[sid] = _per_sample_count.get(sid, 0) + 1
            instance_sample_ids.append(sid)
            instance_annotation_ids.append(_per_sample_count[sid])
        else:
            instance_sample_ids.append(None)
            instance_annotation_ids.append(0)

    if not instance_sample_ids:
        return

    # Optionally log per-sample aggregated mean to dashboard
    if log:
        for name, values in signals.items():
            try:
                arr = values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)
                if arr.size == 0:
                    continue
                scalar = float(arr.mean())
                _log_signal(scalar, None, name, step=step)
            except Exception:
                pass

    # During evaluation mode, don't mutate dataframe state
    try:
        from weightslab.components.evaluation_controller import eval_controller
        if eval_controller.is_running():
            return
    except Exception:
        pass

    # Build losses dict with signals// prefix to match save_signals convention
    losses_data = {}
    for name, values in signals.items():
        key = name if name.startswith("signals//") else f"signals//{name}"
        try:
            arr = values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)
            if arr.ndim > 1:
                arr = arr.reshape(arr.shape[0], -1).mean(axis=1)
            losses_data[key] = arr.astype(np.float32)
        except Exception:
            continue

    if not losses_data:
        return

    # origin is intentionally NOT forwarded: instance rows (annotation_id >= 1) don't
    # carry an origin; the flush derives it from the sample row (annotation_id 0).
    DATAFRAME_M.enqueue_instance_batch(
        sample_ids=instance_sample_ids,
        annotation_ids=instance_annotation_ids,
        losses=losses_data,
        step=step,
        targets=targets,
    )

    # Mirror scalar instance signals into the per-instance logger history so they
    # are queryable via query_instance_history / query_per_instance.
    try:
        _inst_logger = get_logger()
        if _inst_logger is not None and hasattr(_inst_logger, "add_instance_scalars"):
            _log_step = step if step is not None else 0
            for _key, _arr in losses_data.items():
                _sig_name = _key.replace("signals//", "")
                try:
                    _inst_logger.add_instance_scalars(
                        graph_name=_sig_name,
                        sample_ids=instance_sample_ids,
                        annotation_ids=instance_annotation_ids,
                        values=_arr,
                        global_step=_log_step,
                    )
                except Exception:
                    pass
    except Exception:
        pass


def get_active_group_mask(
    group_ids: list[str],
    origin: str,
) -> th.Tensor:
    """Return a boolean mask (one entry per group_id) indicating active (non-tainted) groups.

    A group is "tainted" when at least one of its members has been marked as
    discarded in the UI. Tainted groups should be excluded from group-level loss
    computations so the model is not updated based on broken pairs/triplets.

    This should be called **before** `.mean()` in your training step, so the
    gradient for tainted groups is properly zeroed, not just suppressed in the UI.

    Args:
        group_ids: List of group ID strings for the current batch (one per pair/group).
        origin: Dataset split name matching the ledger (e.g. 'train_loader').

    Returns:
        A float tensor of shape (len(group_ids),) with 1.0 for active groups
        and 0.0 for tainted groups. Safe to multiply against your loss vector.

    Example::

        # Cosine embedding loss — one value per pair in the batch
        loss_embed = loss_cosine(e1, e2, y)            # shape: (B/2,)
        group_mask = wl.get_active_group_mask(group_ids, origin="train_loader")
        # Zero out tainted pairs so they don't update weights
        n_active = group_mask.sum().clamp(min=1)
        loss_embed = (loss_embed * group_mask).sum() / n_active
    """
    global DATAFRAME_M
    if DATAFRAME_M is None:
        DATAFRAME_M = get_dataframe()

    group_ids = [str(g) for g in group_ids]
    mask = th.ones(len(group_ids), dtype=th.float32)

    if DATAFRAME_M is None or not hasattr(DATAFRAME_M, 'get_tainted_group_ids'):
        return mask

    try:
        tainted = DATAFRAME_M.get_tainted_group_ids(group_ids, origin)
        if tainted:
            for i, gid in enumerate(group_ids):
                if gid in tainted:
                    mask[i] = 0.0
    except Exception:
        pass  # Fail-safe: if check fails, treat all groups as active

    return mask


def get_active_sample_mask(
    sample_ids: list[str],
    origin: str,
) -> th.Tensor:
    """Return a boolean mask (one entry per sample_id) indicating active (non-discarded) samples.

    This ensures that any sample marked as discarded in the UI is immediately
    excluded from per-sample loss calculations, even if it is already in the
    current training batch (before the sampler's next epoch refresh).

    Args:
        sample_ids: List of sample ID strings/ints for the current batch.
        origin: Dataset split name (e.g. 'train_loader').

    Returns:
        A float tensor of shape (len(sample_ids),) with 1.0 for active samples
        and 0.0 for discarded samples.
    """
    global DATAFRAME_M
    if DATAFRAME_M is None:
        DATAFRAME_M = get_dataframe()

    sample_ids = [str(sid) for sid in sample_ids]
    mask = th.ones(len(sample_ids), dtype=th.float32)

    if DATAFRAME_M is None or not hasattr(DATAFRAME_M, 'get_discarded_sample_ids'):
        return mask

    try:
        discarded_ids = DATAFRAME_M.get_discarded_sample_ids(sample_ids, origin)
        if discarded_ids:
            for i, sid in enumerate(sample_ids):
                if sid in discarded_ids:
                    mask[i] = 0.0
    except Exception:
        pass

    return mask


def save_group_signals(
    signals: dict,
    group_ids: list[str] | th.Tensor,
    origin: str = 'train',
    step: int | None = None,
    log: bool = True
):
    """
    Save and broadcast group-level statistics (e.g., contrastive loss for an image pair).

    Args:
        signals: Dictionary of {name: value} metrics for the group.
        group_ids: List of group IDs the signals belong to.
        origin: Split name ('train', 'val').
        step: Optional training step.
        log: Whether to log to the global metrics dashboard.
    """
    global DATAFRAME_M
    if DATAFRAME_M is None:
        DATAFRAME_M = get_dataframe()

    # Get current model step
    step = _get_step(step=step)

    # Convert to standard format
    # Convert to standard format, forcing to strings for consistent ledger indexing
    if isinstance(group_ids, th.Tensor):
        group_ids = group_ids.detach().cpu().numpy().astype(str).tolist()
    else:
        group_ids = [str(gid) for gid in group_ids]

    # Process signals and handle batches
    batch_signals = {}
    scalar_signals = {}

    for k, v in signals.items():
        # Prefix for UI grouping
        key = k if k.startswith("signals//") else f"signals//{k}"

        # Detect if this is a batch vector matching group_ids
        val_to_log = v

        if hasattr(v, '__len__') and not isinstance(v, (str, dict)) and len(v) == len(group_ids):
            if hasattr(v, 'detach'):
                v = v.detach().cpu().numpy()
            batch_signals[key] = v
            val_to_log = np.mean(v)
        else:
            if hasattr(v, 'item'):
                v = v.item()
            scalar_signals[key] = v
            val_to_log = v

        # Log mean/scalar to dashboard
        if log:
            _log_signal(float(val_to_log), None, k, step=step)

    # During evaluation mode we must not mutate dataframe state.
    try:
        from weightslab.components.evaluation_controller import eval_controller
        if eval_controller.is_running():
            return
    except Exception:
        pass

    # --- Group-discard filtering ---
    # If any member of a group is discarded, skip the group loss for that group.
    # Per-sample losses (e.g. classification) are still computed since samples stay in the batch.
    tainted_group_ids: set = set()
    if DATAFRAME_M is not None and hasattr(DATAFRAME_M, 'get_tainted_group_ids'):
        try:
            tainted_group_ids = DATAFRAME_M.get_tainted_group_ids(group_ids, origin)
        except Exception:
            pass  # Never block training on best-effort discard check

    # Broadcast to all members in ledger (skip tainted groups)
    all_updates = []
    active_group_ids = []
    for i, gid in enumerate(group_ids):
        if gid in tainted_group_ids:
            continue  # Skip: at least one member was discarded; group loss is undefined

        # We also record the last seen step for all members
        updates = scalar_signals.copy()
        for k, v_batch in batch_signals.items():
            updates[k] = v_batch[i]

        if step is not None:
            updates[SampleStatsEx.LAST_SEEN.value] = step

        all_updates.append(updates)
        active_group_ids.append(gid)

    if not active_group_ids:
        return  # All groups were tainted; nothing to write

    # Bulk update for performance (avoids repeated dataframe scans)
    DATAFRAME_M.update_by_groups_bulk(origin=origin, group_ids=active_group_ids, updates_list=all_updates)


def clear_all():
    """Clear all WeightsLab registries (models, dataloaders, etc.)."""
    ledgers.clear_all()


def _unpack_batch(batch, device=None):
    """Heuristically unpack (inputs, targets, ids) from a batch.

    Supports tuple/list of 1-3 elements and dict-style batches.
    Returns ``(inputs, targets, ids, metadata)`` — targets, ids, and metadata may be ``None``.
    """
    inputs = targets = ids = metadata = None

    if isinstance(batch, (list, tuple)):
        n = len(batch)
        if n >= 1:
            inputs = batch[0]
        if n >= 2:
            ids = batch[1]
        if n >= 3:
            targets = batch[2]
        if n >= 4:
            metadata = batch[3]
    elif isinstance(batch, dict):
        for key in ("image", "input", "x", "data"):
            if key in batch:
                inputs = batch[key]
                break
        if inputs is None and batch:
            inputs = next(iter(batch.values()))
        for key in ("label", "target", "y", "mask"):
            if key in batch:
                targets = batch[key]
                break
        for key in ("id", "sample_id", "idx", "index"):
            if key in batch:
                ids = batch[key]
                break
        for key in ("metadata", "meta", "info"):
            if key in batch:
                metadata = batch[key]
                break
    else:
        inputs = batch

    if device is not None:
        for obj in (inputs, targets):
            if obj is not None and hasattr(obj, "to"):
                try:
                    obj = obj.to(device)
                except Exception:
                    pass
        # re-assign after potential device move
        if isinstance(batch, (list, tuple)):
            inputs = inputs
            targets = targets

    return inputs, ids, targets, metadata


# ##############################################################################################################
# EVALUATION MODE PUBLIC API
# ##############################################################################################################

def _make_default_eval_fn(model):
    """Return a default evaluation callable that uses all registered ledger signals.

    This is used when no ``@wl.eval_fn`` decorator was applied.  For every
    batch it:

    1. Unpacks ``(inputs, targets, ids)`` using a heuristic (tuple/list/dict).
    2. Runs ``model(inputs)`` → ``preds`` under ``torch.no_grad()``.
    3. Calls every signal registered in the ledger as
       ``signal(preds, targets, batch_ids=ids)`` so the wrapped
       ``forward`` / ``compute`` methods fire and log averages to the
       evaluation-mode buffer.

    Loss-style signals (wrapped ``forward``) and metric-style signals
    (wrapped ``compute``) are both handled.  Per-signal errors are silently
    skipped so a missing target or shape mismatch does not abort the whole
    evaluation.
    """
    def _default_eval(loader):
        from weightslab.backend.ledgers import list_signals, get_signal

        try:
            model.eval()
        except Exception:
            pass

        try:
            import torch as _th
            no_grad_ctx = _th.no_grad()
            no_grad_ctx.__enter__()
        except Exception:
            no_grad_ctx = None

        try:
            device = None
            try:
                device = next(model.parameters()).device
            except (StopIteration, Exception):
                pass

            # Resolve registered signals once before the loop.
            signal_names = []
            try:
                signal_names = list_signals() or []
            except Exception:
                pass

            for batch in loader:
                try:
                    inputs, ids, targets, _ = _unpack_batch(batch, device)

                    if inputs is None:
                        continue

                    if device is not None and hasattr(inputs, "to"):
                        try:
                            inputs = inputs.to(device)
                        except Exception:
                            pass
                    if targets is not None and device is not None and hasattr(targets, "to"):
                        try:
                            targets = targets.to(device)
                        except Exception:
                            pass

                    preds = model(inputs)  # infer predictions

                    # Call each registered signal so its wrapped forward/compute
                    # fires and feeds into the evaluation-mode logger buffer.
                    for sig_name in signal_names:
                        try:
                            sig = get_signal(sig_name)
                            if sig is None:
                                continue
                            # Signal wrappers are not uniform: some accept
                            # batch_ids, some only (preds, targets), and a few
                            # only predictions. Try the richest signature first
                            # then fall back gracefully.
                            attempted = False
                            if targets is not None and ids is not None:
                                try:
                                    sig(preds, targets, batch_ids=ids)
                                    attempted = True
                                except TypeError:
                                    pass

                            if not attempted and targets is not None:
                                try:
                                    sig(preds, targets)
                                    attempted = True
                                except TypeError:
                                    pass

                            if not attempted and ids is not None:
                                try:
                                    sig(preds, batch_ids=ids)
                                    attempted = True
                                except TypeError:
                                    pass

                            if not attempted:
                                sig(preds)
                        except Exception as _se:
                            logger.debug(
                                "[wl.default_eval] signal '%s' failed: %s\nAre you sure signal {%s} is compatible with weightslab?", sig_name, _se, sig_name
                            )

                except Exception as _be:
                    logger.debug("[wl.default_eval] batch forward failed: %s", _be)
        finally:
            if no_grad_ctx is not None:
                try:
                    no_grad_ctx.__exit__(None, None, None)
                except Exception:
                    pass

        try:
            model.train()
        except Exception:
            pass

    return _default_eval


def eval_fn(func):
    """Decorator that registers a function as the evaluation runner.

    The decorated function receives a single *loader* argument — a
    ``_EvalManagedLoader`` wrapping the requested split's
    ``DataLoaderInterface``.  It should iterate that loader and compute
    the watched criteria / metrics exactly as in a normal test pass.  All
    ``add_scalars`` calls are intercepted by the logger's evaluation-mode
    buffer.

    Usage::

        @wl.eval_fn
        def eval_pass(loader):
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    inputs, targets = batch[:2]
                    preds = model(inputs)
                    criterion(preds, targets)
    """
    global _REGISTERED_EVAL_FN
    _REGISTERED_EVAL_FN = func
    return func


def pointcloud_thumbnail(func):
    """Register a custom 2D thumbnail renderer for point-cloud samples.

    For ``task_type = "detection_pointcloud"`` the studio previews each sample
    as a server-rendered 2D image (default: BEV; ``range`` for a LiDAR
    scan-style spherical projection). This decorator overrides that with your
    own function, e.g. a range/spherical projection:

        @wl.pointcloud_thumbnail
        def to_range_image(points):           # points: [M, 2..F] float
            return my_range_projection(points)  # -> (H, W, 3) uint8 or PIL.Image

    (Note: ``@wl.3d_pc_thumb`` isn't valid Python — identifiers can't start
    with a digit — so the verb is spelled out.) A ``render_thumbnail_2d``
    method on the dataset takes precedence over this global registration.
    See also [[detection-3d-pipeline]].
    """
    from weightslab.data.point_cloud_utils import register_thumbnail_fn
    return register_thumbnail_fn(func)


def pointcloud_boxes(func):
    """Register a custom box projector for point-cloud thumbnails/overlays.

    Maps metric boxes ([N, 7..9] 3D or [N, 4..6] 2D rows) to normalized
    ``[x1, y1, x2, y2, cls, conf]`` boxes in the thumbnail image frame, so the
    overlay lines up with a custom ``@wl.pointcloud_thumbnail`` projection:

        @wl.pointcloud_boxes
        def boxes_to_range(boxes):
            return my_boxes_in_range_frame(boxes)  # -> [N, 6] normalized

    A ``project_boxes_2d`` method on the dataset takes precedence.
    """
    from weightslab.data.point_cloud_utils import register_boxes_fn
    return register_boxes_fn(func)


def trigger_pending_evaluation_async() -> bool:
    """Start a background worker to execute the pending evaluation.

    Model, loaders, and eval function are resolved automatically from the
    ledger (registered via ``wl.watch_or_edit`` and ``@wl.eval_fn``).

    Returns ``True`` when a worker is active or started, ``False`` when
    there is no pending/running evaluation to service.
    """
    from weightslab.components.evaluation_controller import eval_controller

    if not (eval_controller.is_pending() or eval_controller.is_running()):
        return False

    def _worker() -> None:
        global _EVAL_WORKER_THREAD
        try:
            while True:
                _ = run_pending_evaluation()

                if not (eval_controller.is_pending() or eval_controller.is_running()):
                    break

                time.sleep(0.02)
        except Exception as exc:
            logger.exception("[wl.trigger_pending_evaluation_async] worker failed: %s", exc)
        finally:
            with _EVAL_WORKER_LOCK:
                _EVAL_WORKER_THREAD = None
            # Belt-and-suspenders: if the controller is somehow still in a live
            # state after the worker exits (e.g. unhandled exception bypassed the
            # mark_error call inside run_pending_evaluation), forcibly mark it as
            # errored so the UI and watchdog both see a terminal state.
            try:
                from weightslab.components.evaluation_controller import eval_controller as _ec
                if _ec.is_running() or _ec.is_pending():
                    _ec.mark_error("Evaluation worker terminated unexpectedly")
            except Exception:
                pass

    global _EVAL_WORKER_THREAD
    with _EVAL_WORKER_LOCK:
        if _EVAL_WORKER_THREAD is not None and _EVAL_WORKER_THREAD.is_alive():
            return True

        _EVAL_WORKER_THREAD = threading.Thread(
            target=_worker,
            name="WL-EvalWorker",
            daemon=True,
        )
        _EVAL_WORKER_THREAD.start()

    return True


def run_pending_evaluation(
    loaders: dict = None,
    model=None,
    eval_fn=None,
    device=None,
) -> bool:
    """Check for a pending evaluation request and execute it if one is present.

    All parameters are optional — when omitted, each is resolved from the
    ledger automatically (model registered via ``wl.watch_or_edit``,
    loaders via ``wl.watch_or_edit(..., flag='data')``, eval function via
    the ``@wl.eval_fn`` decorator).

    Can still be called from the training loop with explicit arguments for
    backwards-compatibility::

        if wl.run_pending_evaluation():  # ledger mode — no args needed
            continue

    Args:
        loaders:  Optional mapping of *loader_name* → ``DataLoaderInterface``.
                  When ``None``, the loader is looked up by split name from
                  the ledger.
        model:    Optional tracked model instance (used to read ``get_age()``).
                  When ``None``, resolved from the ledger.
        eval_fn:  Optional callable with signature ``eval_fn(loader) -> None``.
                  When ``None``, the function registered via ``@wl.eval_fn``
                  is used.
        device:   Unused; kept for API symmetry.

    Returns:
        ``True`` if an evaluation was executed (caller should ``continue``
        the training loop), ``False`` otherwise.
    """
    from weightslab.components.evaluation_controller import eval_controller
    from weightslab.components.global_monitoring import pause_controller
    from weightslab.backend.ledgers import get_logger, get_checkpoint_manager

    req = eval_controller.consume_request()
    if req is None:
        return False

    split_name: str = req.get("split_name", "")
    tags: list = req.get("tags", [])
    use_full_set: bool = req.get("use_full_set", True)
    was_paused: bool = req.get("was_paused", False)
    max_steps = req.get("max_steps", None)

    logger_obj = logging.getLogger(__name__)
    logger_obj.info(
        "[wl.run_pending_evaluation] split=%s tags=%s full_set=%s was_paused=%s max_steps=%s",
        split_name, tags, use_full_set, was_paused, max_steps,
    )

    # Resolve model from ledger when not provided explicitly.
    _model = model
    if _model is None:
        try:
            _model = get_model()
        except Exception:
            _model = None

    # Resolve eval_fn from decorator registry → fall back to built-in default.
    _eval_fn = eval_fn if eval_fn is not None else _REGISTERED_EVAL_FN
    if _eval_fn is None:
        if _model is not None:
            _eval_fn = _make_default_eval_fn(_model)
        else:
            eval_controller.mark_error(
                "No evaluation function and no model available. Register a model with "
                "wl.watch_or_edit(model, flag='model') or decorate your eval function with @wl.eval_fn."
            )
            if was_paused:
                pause_controller.pause()
            return True

    # Get the target DataLoaderInterface from explicit loaders dict or ledger.
    if loaders is not None:
        loader_if = loaders.get(split_name)
    else:
        try:
            loader_if = get_dataloader(split_name)
        except Exception:
            loader_if = None

    if loader_if is None:
        available = list(loaders.keys()) if loaders is not None else "(from ledger)"
        eval_controller.mark_error(
            f"Loader '{split_name}' not found. Available: {available}"
        )
        if was_paused:
            pause_controller.pause()
        return True

    # ------------------------------------------------------------------
    # 1. Save sampler state and force shuffle=False during evaluation
    # ------------------------------------------------------------------
    sampler = _resolve_eval_sampler(loader_if)
    prev_shuffle = bool(getattr(sampler, "shuffle", False)) if sampler is not None else False
    if sampler is not None and hasattr(sampler, "shuffle"):
        sampler.shuffle = False

    # ------------------------------------------------------------------
    # 2. Apply tag filter via eval allow-list (similar to discard logic)
    # ------------------------------------------------------------------
    prev_eval_allow_list = None
    filtered_count = None
    if not use_full_set and tags and (sampler is None or not hasattr(sampler, "_eval_allow_list")):
        eval_controller.mark_error(
            f"Loader '{split_name}' does not support tag-filter evaluation"
        )
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list)
        if was_paused:
            pause_controller.pause()
        return True

    if sampler is not None and hasattr(sampler, "_eval_allow_list"):
        prev_eval_allow_list = getattr(sampler, "_eval_allow_list", None)
        if not use_full_set and tags:
            allow_list = _build_eval_allow_list(loader_if, tags, split_name)
            sampler._eval_allow_list = allow_list
        else:
            sampler._eval_allow_list = None
        # After setting allow-list, check if any samples remain
        try:
            filtered_count = len(loader_if)
        except Exception:
            filtered_count = None
        if filtered_count == 0:
            eval_controller.mark_error(f"Evaluation set is empty after filtering (tags={tags}, split={split_name})")
            logger_obj.error(f"[wl.run_pending_evaluation] Evaluation set is empty after filtering (tags={tags}, split={split_name})")
            # Restore sampler state
            _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list)
            if was_paused:
                pause_controller.pause()
            return True

    # Reset dataloader iterator so we start from index 0
    if hasattr(loader_if, "reset_iterator"):
        try:
            loader_if.reset_iterator()
        except Exception:
            if hasattr(loader_if, "_iterator"):
                loader_if._iterator = None
    elif hasattr(loader_if, "_iterator"):
        loader_if._iterator = None

    # ------------------------------------------------------------------
    # 3. Compute eval hash and start logger evaluation mode
    # ------------------------------------------------------------------
    chkpt_mgr = get_checkpoint_manager()
    base_hash = chkpt_mgr.get_current_experiment_hash() if chkpt_mgr else "unknown"
    signal_logger = get_logger()

    eval_count = 1
    eval_hash = f"{base_hash}_{eval_count}"
    if signal_logger is not None:
        eval_count = signal_logger.get_next_evaluation_count(base_hash)
        eval_hash = f"{base_hash}_{eval_count}"
        signal_logger.start_evaluation_mode(split_name, eval_hash, evaluation_tags=tags)

    # ------------------------------------------------------------------
    # 4. Compute total samples for progress reporting
    # ------------------------------------------------------------------
    try:
        total_batches = len(loader_if)
    except Exception:
        total_batches = 0

    try:
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps > 0:
                total_batches = min(total_batches, max_steps) if total_batches > 0 else max_steps
        else:
            max_steps = None
    except Exception:
        max_steps = None

    eval_controller.report_progress(0, total_batches, f"Evaluating '{split_name}'…")

    # ------------------------------------------------------------------
    # 5. Run evaluation with bounded timeout + cancellation checks
    # ------------------------------------------------------------------
    # Freeze model age during evaluation by switching to EVAL tracking mode.
    _prev_tracking_mode = None
    if _model is not None and hasattr(_model, "set_tracking_mode"):
        try:
            from weightslab.components.tracking import TrackingMode as _TrackingMode
            _prev_tracking_mode = getattr(_model, "tracking_mode", None)
            _model.set_tracking_mode(_TrackingMode.EVAL)
            _model.eval() if hasattr(_model, 'eval') else None
        except Exception:
            pass

    controlled_loader = _EvalManagedLoader(loader_if, split_name, total_batches, max_batches=max_steps)
    eval_error = None

    # Set evaluation context (exempt from watchdog timeouts) and guarding
    from weightslab.components.global_monitoring import set_in_evaluation, reset_in_evaluation
    eval_context_token = set_in_evaluation(True)

    try:
        with th.no_grad():
            _eval_fn(controlled_loader)
    except _EvalCanceled as exc:
        logger_obj.warning("[wl.run_pending_evaluation] canceled: %s", exc)
        eval_controller.mark_canceled(str(exc))
        if signal_logger is not None and hasattr(signal_logger, "abort_evaluation_mode"):
            signal_logger.abort_evaluation_mode()
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)
        if was_paused:
            pause_controller.pause()
        return True
    except _EvalTimeout as exc:
        logger_obj.error("[wl.run_pending_evaluation] timeout: %s", exc)
        eval_error = f"Evaluation timeout: {exc}"
        eval_controller.mark_error(eval_error)
        if signal_logger is not None and hasattr(signal_logger, "abort_evaluation_mode"):
            signal_logger.abort_evaluation_mode()
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)
        if was_paused:
            pause_controller.pause()
        return True
    except Exception as exc:
        import traceback
        tb_str = traceback.format_exc()
        eval_error = f"{type(exc).__name__}: {exc}"
        logger_obj.error("[wl.run_pending_evaluation] eval_fn raised: %s\n%s", exc, tb_str)
        eval_controller.mark_error(eval_error)
        if signal_logger is not None and hasattr(signal_logger, "abort_evaluation_mode"):
            signal_logger.abort_evaluation_mode()
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)
        if was_paused:
            pause_controller.pause()
        return True
    finally:
        # Reset evaluation context
        reset_in_evaluation(eval_context_token)

    # A cancel request can arrive just as eval_fn returns. In that race window,
    # honor cancellation before finalizing marker persistence.
    if eval_controller.is_cancel_requested():
        cancel_reason = f"Evaluation on '{split_name}' canceled by user"
        logger_obj.warning("[wl.run_pending_evaluation] canceled after eval_fn return: %s", cancel_reason)
        eval_controller.mark_canceled(cancel_reason)
        if signal_logger is not None and hasattr(signal_logger, "abort_evaluation_mode"):
            signal_logger.abort_evaluation_mode()
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)
        if was_paused:
            pause_controller.pause()
        return True

    # ------------------------------------------------------------------
    # 6. Finalise: stop eval mode → creates average markers in logger
    # ------------------------------------------------------------------
    if eval_controller.is_cancel_requested():
        cancel_reason = f"Evaluation on '{split_name}' canceled by user"
        logger_obj.warning("[wl.run_pending_evaluation] canceled before marker finalization: %s", cancel_reason)
        eval_controller.mark_canceled(cancel_reason)
        if signal_logger is not None and hasattr(signal_logger, "abort_evaluation_mode"):
            signal_logger.abort_evaluation_mode()
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)
        if was_paused:
            pause_controller.pause()
        return True

    model_age = 0
    try:
        model_age = _model.get_age() - 1 if _model is not None and hasattr(_model, "get_age") else 0  # Model anticipates a step after eval, so subtract 1 to report the age corresponding to the just-evaluated checkpoint.
    except Exception:
        pass

    # One more cancel check immediately before marker finalization - this catches
    # cancel requests that arrived during the very end of eval_fn execution.
    if eval_controller.is_cancel_requested():
        cancel_reason = f"Evaluation on '{split_name}' canceled by user (final pre-marker check)"
        logger_obj.warning("[wl.run_pending_evaluation] canceled before marker finalization (final): %s", cancel_reason)
        eval_controller.mark_canceled(cancel_reason)
        if signal_logger is not None and hasattr(signal_logger, "abort_evaluation_mode"):
            signal_logger.abort_evaluation_mode()
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)
        if was_paused:
            pause_controller.pause()
        return True

    result: dict = {}
    if signal_logger is not None:
        result = signal_logger.stop_evaluation_mode(model_age)

    # Another narrow race window exists between stop_evaluation_mode() and
    # mark_done(); if cancel arrives here, purge that eval hash and report canceled.
    if eval_controller.is_cancel_requested():
        cancel_reason = f"Evaluation on '{split_name}' canceled by user"
        logger_obj.warning("[wl.run_pending_evaluation] canceled after marker finalization: %s", cancel_reason)
        if signal_logger is not None:
            # Remove evaluation hash from history and pending queue
            if hasattr(signal_logger, "remove_evaluation_hash"):
                signal_logger.remove_evaluation_hash(eval_hash)
            # Also abort any remaining evaluation mode state for extra safety
            if hasattr(signal_logger, "abort_evaluation_mode"):
                signal_logger.abort_evaluation_mode()
        eval_controller.mark_canceled(cancel_reason)
        _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)
        if was_paused:
            pause_controller.pause()
        return True

    eval_controller.report_progress(total_batches, total_batches, "Done")

    # ------------------------------------------------------------------
    # 7. Restore shuffle + allow-list + tracking mode
    # ------------------------------------------------------------------
    _restore_eval_state(sampler, prev_shuffle, prev_eval_allow_list, model=_model, prev_tracking_mode=_prev_tracking_mode)

    # ------------------------------------------------------------------
    # 8. Pause training
    # ------------------------------------------------------------------
    pause_controller.pause()

    # Atomic completion: if cancel was requested in the final race window,
    # convert to canceled and purge markers for this eval hash.
    if hasattr(eval_controller, "mark_done_unless_canceled"):
        marked_done = eval_controller.mark_done_unless_canceled(result)
        if not marked_done:
            if signal_logger is not None:
                if hasattr(signal_logger, "remove_evaluation_hash"):
                    signal_logger.remove_evaluation_hash(eval_hash)
                if hasattr(signal_logger, "abort_evaluation_mode"):
                    signal_logger.abort_evaluation_mode()
            return True
    else:
        eval_controller.mark_done(result)

    # Console output — visible even without Weights Studio connected.
    logger.info(f"\n{'='*70}")
    logger.info(f"[WeightsLab] Evaluation Results")
    logger.info(f"{'='*70}")
    logger.info(f"  Split:        {split_name}")
    logger.info(f"  Model Step:   {model_age}")
    logger.info(f"  Tags:         {tags}")
    logger.info(f"  Total Samples: {filtered_count if filtered_count is not None else 'unknown'}")
    logger.info(f"  Total Batches: {total_batches}")
    logger.info(f"  Eval Hash:    {eval_hash}")

    if result:
        logger.info(f"  Metrics:\n")
        for k, v in result.items():
            if isinstance(v, float):
                logger.info(f"    {k:30s} = {v:.6f}")
            else:
                logger.info(f"    {k:30s} = {v}")
    else:
        logger.info(f"  Status:       No metrics recorded")
        error_msg = (
            f"Evaluation did not produce any metrics.\n"
            f"  Possible causes:\n"
            f"    • Evaluation function is not compatible with the experiment setup\n"
            f"    • No signals were computed during evaluation\n"
            f"    • Model or data loader not registered in the ledger\n\n"
            f"  Solution: Create a custom evaluation function decorated with @wl.eval_fn.\n"
            f"  This function should:\n"
            f"    1. Accept only one parameter: loader\n"
            f"    2. Be fully based on the WeightsLab ledger\n"
            f"    3. Retrieve model, device, and metrics from wl.ledger.*\n"
            f"    4. Register loss/metric functions with wl.watch_or_edit(..., flag='loss/metric')\n\n"
            f"  Example from detection use case:\n"
            f"    @wl.eval_fn\n"
            f"    def validate(loader):\n"
            f"        model = wl.ledger.get_model()\n"
            f"        device = wl.ledger.get_device()\n"
            f"        for batch in loader:\n"
            f"            ...\n\n"
            f"  See documentation: https://grayboxtech.github.io/weightslab/latest/index.html"
        )
        logger.warning(error_msg)

    logger.info(f"{'='*70}\n")

    logger_obj.info(
        "[wl.run_pending_evaluation] Evaluation complete on '%s' @ step %d: %s",
        split_name, model_age, result,
    )
    return True


def _build_eval_allow_list(loader_if, tags: list, split_name: str) -> set:
    """Build a set of sample UIDs that have ALL the requested tags.

    Uses the DataFrameManager associated with the loader's tracked dataset
    to filter by tag columns (same logic as Break-By-Slice).
    """
    from weightslab.data.sample_stats import SampleStatsEx
    from weightslab.backend.ledgers import get_dataframe

    allow_set: set = set()
    try:
        df_manager = get_dataframe()
        if df_manager is None:
            # No dataframe → return all UIDs (effectively no filter)
            tracked = getattr(loader_if, "tracked_dataset", None)
            if tracked is not None:
                uids = getattr(tracked, "unique_ids", None) or getattr(tracked, "physical_uids", None)
                if uids is not None:
                    return {str(u) for u in uids}
            return allow_set

        df = df_manager.get_df_view()

        # Build compound mask: sample must have ALL tags set to True
        mask = None
        for tag in tags:
            col = f"{SampleStatsEx.TAG.value}:{tag}"
            if col in df.columns:
                col_mask = df[col] == True  # noqa: E712
                mask = col_mask if mask is None else (mask & col_mask)

        if mask is None:
            # Tags not found → return all (no filter)
            allow_set = {str(idx) for idx in df.index}
            return allow_set

        # Filter by origin if the split name appears in index
        filtered = df[mask]

        # Support both MultiIndex (origin, sample_id) and flat index
        if isinstance(filtered.index, type(None)):
            return allow_set

        import pandas as pd
        if isinstance(filtered.index, pd.MultiIndex):
            # Try to filter by origin matching split_name (either exact or prefix match)
            origin_level = filtered.index.get_level_values(SampleStatsEx.ORIGIN.value) \
                if SampleStatsEx.ORIGIN.value in filtered.index.names else \
                filtered[filtered[SampleStatsEx.ORIGIN.value] == split_name].index.get_level_values(0)
            sub = filtered.loc[origin_level]
            # Extract sample_id level
            sid_level_name = SampleStatsEx.SAMPLE_ID.value \
                if SampleStatsEx.SAMPLE_ID.value in sub.index.names else None
            if sid_level_name:
                allow_set = {str(v) for v in sub.index.get_level_values(sid_level_name)}
            else:
                allow_set = {str(v) for v in sub.index.get_level_values(1)}
        else:
            allow_set = {str(idx) for idx in filtered.index}
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "[_build_eval_allow_list] Could not build allow list: %s", exc
        )
        # Fallback: no filter (evaluate all)
        tracked = getattr(loader_if, "tracked_dataset", None)
        if tracked is not None:
            uids = getattr(tracked, "unique_ids", None) or getattr(tracked, "physical_uids", None)
            if uids is not None:
                return {str(u) for u in uids}
    return allow_set


def _restore_eval_state(sampler, prev_shuffle: bool, prev_eval_allow_list, model=None, prev_tracking_mode=None) -> None:
    """Restore sampler shuffle, allow-list, and model tracking mode to pre-evaluation values."""
    if model is not None and prev_tracking_mode is not None and hasattr(model, "set_tracking_mode"):
        try:
            model.set_tracking_mode(prev_tracking_mode)
            if prev_tracking_mode == 'train':
                model.train() if hasattr(model, 'train') else None
        except Exception:
            pass
    if sampler is None:
        return
    if hasattr(sampler, "shuffle"):
        sampler.shuffle = prev_shuffle
    if hasattr(sampler, "_eval_allow_list"):
        sampler._eval_allow_list = prev_eval_allow_list


def _resolve_eval_sampler(loader_if):
    """Best-effort resolution of the sampler used by the evaluation loader."""
    sampler = getattr(loader_if, "_mutable_batch_sampler", None)
    if sampler is not None:
        return sampler

    dataloader = getattr(loader_if, "dataloader", None)
    if dataloader is None:
        dataloader = loader_if

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None:
        return batch_sampler

    return getattr(dataloader, "sampler", None)


def _get_eval_timeout_config() -> tuple[float, float, float]:
    """Return (multiplier, min_seconds, absolute_seconds_override)."""

    try:
        multiplier = max(1.0, float(os.getenv("WEIGHTSLAB_EVAL_TIMEOUT_MULTIPLIER", "13")))
    except Exception:
        multiplier = 1.3

    try:
        min_seconds = max(0.0, float(os.getenv("WEIGHTSLAB_EVAL_TIMEOUT_MIN_SECONDS", "5")))
    except Exception:
        min_seconds = 5.0

    try:
        absolute_timeout = max(0.0, float(os.getenv("WEIGHTSLAB_EVAL_TIMEOUT_SECONDS", "0")))
    except Exception:
        absolute_timeout = 0.0

    return multiplier, min_seconds, absolute_timeout


class _EvalCanceled(RuntimeError):
    pass


class _EvalTimeout(RuntimeError):
    pass


class _EvalManagedLoader:
    """Iterator wrapper adding progress, cancel checks, and timeout enforcement."""

    def __init__(self, loader, split_name: str, total_batches: int, max_batches: Optional[int] = None) -> None:
        from weightslab.components.evaluation_controller import eval_controller

        self._loader = loader
        self._split_name = split_name
        self._total_batches = max(0, int(total_batches))
        self._max_batches = int(max_batches) if isinstance(max_batches, int) and max_batches > 0 else None
        self._controller = eval_controller

        self._start_time = time.monotonic()
        self._processed_batches = 0
        self._avg_batch_seconds = 0.0
        self._multiplier, self._min_seconds, self._absolute_timeout = _get_eval_timeout_config()

        # UL deps
        self.dataset = loader.dataset if hasattr(loader, 'dataset') else None

    def _check_cancel_or_timeout(self) -> None:
        if self._controller.is_cancel_requested():
            raise _EvalCanceled(f"Evaluation on '{self._split_name}' canceled by user")

        elapsed = time.monotonic() - self._start_time
        if self._absolute_timeout > 0 and elapsed > self._absolute_timeout:
            raise _EvalTimeout(
                f"Evaluation timeout on '{self._split_name}' after {elapsed:.1f}s (configured {self._absolute_timeout:.1f}s)"
            )

        if self._total_batches <= 0 or self._processed_batches <= 0 or self._avg_batch_seconds <= 0:
            return

        projected = self._avg_batch_seconds * self._total_batches
        timeout_seconds = max(self._min_seconds, projected * self._multiplier)
        if elapsed > timeout_seconds:
            raise _EvalTimeout(
                f"Evaluation timeout on '{self._split_name}' after {elapsed:.1f}s "
                f"(projected={projected:.1f}s, limit={timeout_seconds:.1f}s, multiplier={self._multiplier:.2f})"
            )
    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        it = iter(self._loader)
        while True:
            if self._max_batches is not None and self._processed_batches >= self._max_batches:
                return

            self._check_cancel_or_timeout()
            try:
                batch = next(it)
            except StopIteration:
                return

            batch_started = time.monotonic()
            yield batch

            batch_seconds = max(1e-6, time.monotonic() - batch_started)
            self._processed_batches += 1
            self._avg_batch_seconds = (
                (self._avg_batch_seconds * (self._processed_batches - 1)) + batch_seconds
            ) / self._processed_batches

            progress_total = self._total_batches
            progress_current = min(self._processed_batches, progress_total) if progress_total > 0 else self._processed_batches
            self._controller.report_progress(
                progress_current,
                progress_total,
                f"Evaluating '{self._split_name}'…",
            )


# ##############################################################################################################
# SIGNAL HISTORY QUERY HELPERS
# ##############################################################################################################

def get_current_experiment_hash() -> str | None:
    """Return the hash of the currently active experiment run.

    Reads the hash from the registered checkpoint manager.  Returns ``None``
    when no experiment is active or no checkpoint manager has been registered.

    Example::

        h = wl.get_current_experiment_hash()
        wl.write_history("/tmp/run", experiment_hash=h)
    """
    try:
        cm = get_checkpoint_manager()
        if cm is None:
            return None
        h = cm.get_current_experiment_hash()
        return h if isinstance(h, str) else None
    except Exception:
        return None


def query_signal_history(
    signal_name: str,
    exp_hash: str | None = None,
) -> list:
    """Return per-sample history for *signal_name* across all samples.

    Returns a list of ``(sample_id, step, value, experiment_hash)`` tuples.
    Pass *exp_hash* to restrict to a single experiment run.

    Example::

        history = wl.query_signal_history("train/loss")
        for sample_id, step, loss, h in history:
            print(sample_id, step, loss)
    """
    _lg = get_logger()
    if _lg is None:
        return []
    return _lg.query_per_sample(signal_name, sample_ids=None, exp_hash=exp_hash)


def query_sample_history(
    sample_id: str,
    signal_name: str | None = None,
    exp_hash: str | None = None,
) -> list:
    """Return the full logged history for a given *sample_id*.

    Returns a list of ``(signal_name, step, value, experiment_hash)`` tuples.
    Pass *signal_name* to restrict to a single metric; pass *exp_hash* to
    restrict to a single experiment run.

    Example::

        for sig, step, val, h in wl.query_sample_history("img_0042"):
            print(sig, step, val)
    """
    _lg = get_logger()
    if _lg is None:
        return []
    names = (
        [signal_name]
        if signal_name
        else _lg.list_sample_signal_names()
    )
    results = []
    for name in names:
        for sid, step, val, h in _lg.query_per_sample(
            name, sample_ids=[sample_id], exp_hash=exp_hash
        ):
            results.append((name, step, val, h))
    return results


def query_instance_history(
    sample_id: str,
    annotation_id: int,
    signal_name: str | None = None,
    exp_hash: str | None = None,
) -> list:
    """Return the full logged history for a ``(sample_id, annotation_id)`` instance.

    *annotation_id* is 1-based (0 is the per-sample row; instances start at 1).
    Returns a list of ``(signal_name, step, value, experiment_hash)`` tuples.

    Example::

        for sig, step, val, h in wl.query_instance_history("img_0042", annotation_id=1):
            print(sig, step, val)
    """
    _lg = get_logger()
    if _lg is None:
        return []
    names = (
        [signal_name]
        if signal_name
        else _lg.list_instance_signal_names()
    )
    results = []
    for name in names:
        for sid, aid, step, val, h in _lg.query_per_instance(
            name, sample_id=sample_id, annotation_id=annotation_id, exp_hash=exp_hash
        ):
            results.append((name, step, val, h))
    return results


def write_history(
    path: str | None = None,
    format: str = "json",
    type_of_history=None,
    graph_name=None,
    experiment_hash: str | None = None,
    sample_id=None,
    instance_id=None,
) -> str:
    """Dump signal history to *path* as JSON or CSV.

    Parameters
    ----------
    path : str, optional
        Output file path **or** directory.  When omitted (``None``), the
        ``root_log_dir`` from the active checkpoint manager is used as the
        output directory.

        - If *path* points to a file (has an extension) the file is written
          directly.
        - If *path* has no extension or is an existing directory, a filename
          is auto-generated as ``<hash>_history.<format>`` inside that
          directory, where ``<hash>`` is an 8-character hex MD5 of the
          normalized call parameters (*type_of_history*, *graph_name*,
          *experiment_hash*, *sample_id*, *instance_id*).  The same filter
          combination always produces the same filename; different filters
          produce different filenames.
        - The directory is created automatically if it does not exist.
    format : {"json", "csv"}
        Output format (default ``"json"``).
    type_of_history : {None, "all", "global", "sample", "instance", "instances"}
        Which history to include.  ``None`` or ``"all"`` writes every type.
        ``"global"`` writes the aggregated training-curve history.
        ``"sample"`` writes per-sample history.
        ``"instance"`` / ``"instances"`` writes per-instance history.
    graph_name : str or list of str, optional
        Restrict to one or more signal / metric names.
    experiment_hash : str, optional
        ``None`` (default) — use the current experiment hash from the
        checkpoint manager.  ``"all"`` — include every hash.
        Any other string — restrict to that specific experiment run.
    sample_id : str or list of str, optional
        Restrict per-sample and per-instance rows to one or more sample IDs.
        Has no effect on global history.
    instance_id : int or list of int, optional
        Restrict per-instance rows to one or more annotation IDs.
        Has no effect on global or per-sample history.

    Returns
    -------
    str
        Absolute path of the file that was written.

    Examples
    --------
    Write all history — directory inferred from ``root_log_dir``::

        wl.write_history()

    Write all history into a specific directory::

        wl.write_history(r"C:\\tmp\\myrun")

    Write only per-sample data for one experiment to CSV::

        wl.write_history(
            r"C:\\tmp\\myrun",
            format="csv",
            type_of_history="sample",
            experiment_hash="abc123",
        )
    """
    import csv as _csv
    import json as _json
    import os as _os
    import hashlib as _hashlib

    logger.debug(
        "write_history called: path=%r, format=%r, type_of_history=%r, "
        "graph_name=%r, experiment_hash=%r, sample_id=%r, instance_id=%r",
        path, format, type_of_history, graph_name, experiment_hash,
        sample_id, instance_id,
    )

    _lg = get_logger()
    if _lg is None:
        logger.warning(
            "write_history: no active logger (get_logger() returned None); "
            "nothing to write. Returning path=%r.", path or "."
        )
        return path or "."

    # Resolve path: fall back to root_log_dir from the checkpoint manager
    if path is None:
        try:
            _cm = _lg.chkpt_manager
            if _cm is not None:
                _rld = _cm.root_log_dir
                path = str(_rld) if _rld is not None else "."
            else:
                path = "."
        except Exception as _e:
            logger.debug("write_history: failed to resolve root_log_dir (%s); "
                         "falling back to current directory.", _e)
            path = "."
        logger.info("write_history: no path given, using output directory %r.", path)

    fmt = format.lower().strip()

    # --- Normalize all parameters first (needed for the auto-filename hash) ---

    # Resolve experiment_hash:
    #   None      → use the current hash from the checkpoint manager (default)
    #   "all"     → no filter, include every hash
    #   any str   → filter to that specific hash
    if experiment_hash is None or experiment_hash == 'last':
        try:
            _current = (
                _lg.chkpt_manager.get_current_experiment_hash()
                if _lg.chkpt_manager is not None
                else None
            )
            experiment_hash = _current if isinstance(_current, str) else None
        except Exception:
            experiment_hash = None
    elif experiment_hash == "all":
        experiment_hash = None  # sentinel: skip hash filtering below

    # Normalize graph_name → set or None
    _gn_filter = None
    if graph_name is not None:
        _gn_filter = {graph_name} if isinstance(graph_name, str) else set(graph_name)

    # Normalize sample_id → list or None
    _sid_filter = None
    if sample_id is not None:
        _sid_filter = [sample_id] if isinstance(sample_id, str) else list(sample_id)

    # Normalize instance_id → list or None
    _aid_filter = None
    if instance_id is not None:
        _aid_filter = [instance_id] if isinstance(instance_id, int) else list(instance_id)

    _type = (type_of_history or "all").lower().strip()
    if _type == "instances":
        _type = "instance"
    write_global = _type in ("all", "global")
    write_sample = _type in ("all", "sample")
    write_instance = _type in ("all", "instance")

    logger.info(
        "write_history: resolved filters → type=%r, experiment_hash=%s, "
        "graph_name=%s, sample_id=%s, instance_id=%s",
        _type,
        experiment_hash if experiment_hash is not None else "<all>",
        sorted(_gn_filter) if _gn_filter is not None else "<all>",
        _sid_filter if _sid_filter is not None else "<all>",
        _aid_filter if _aid_filter is not None else "<all>",
    )

    # --- Resolve output path ---
    # When path has no file extension (or is an existing directory), generate a
    # filename from a short hash of the normalized call parameters so that the
    # same filter combination always produces the same filename.
    _base = _os.path.basename(path)
    if not _os.path.splitext(_base)[1] or _os.path.isdir(path):
        _params_key = (
            _type,
            tuple(sorted(_gn_filter)) if _gn_filter is not None else None,
            experiment_hash,
            tuple(sorted(_sid_filter)) if _sid_filter is not None else None,
            tuple(sorted(int(x) for x in _aid_filter)) if _aid_filter is not None else None,
        )
        _phash = _hashlib.md5(str(_params_key).encode()).hexdigest()[:8]
        path = _os.path.join(path, f"{_phash}_history.{fmt}")
        logger.info("write_history: auto-generated filename %r (params hash "
                    "%s).", _os.path.basename(path), _phash)
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    logger.info("write_history: output file → %s", _os.path.abspath(path))

    global_rows: list = []
    sample_rows: list = []
    instance_rows: list = []

    if write_global:
        for gn, hashes in _lg.get_signal_history().items():
            if _gn_filter is not None and gn not in _gn_filter:
                continue
            for h, steps in hashes.items():
                if experiment_hash is not None and h != experiment_hash:
                    continue
                for step, entries in steps.items():
                    for entry in entries:
                        val = (
                            entry.get("metric_value")
                            if isinstance(entry, dict)
                            else float(entry)
                        )
                        global_rows.append({
                            "graph_name": gn,
                            "experiment_hash": h if h is not None else "",
                            "step": step,
                            "metric_value": val,
                        })
        logger.debug("write_history: collected %d global row(s).",
                     len(global_rows))

    if write_sample:
        graphs_s = (
            list(_gn_filter)
            if _gn_filter is not None
            else _lg.list_sample_signal_names()
        )
        for gn in graphs_s:
            for sid, step, val, h in _lg.query_per_sample(
                gn,
                sample_ids=_sid_filter,
                exp_hash=experiment_hash,
            ):
                sample_rows.append({
                    "graph_name": gn,
                    "experiment_hash": h if h is not None else "",
                    "sample_id": sid,
                    "step": step,
                    "metric_value": val,
                })
        logger.debug("write_history: collected %d sample row(s) across %d "
                     "graph(s).", len(sample_rows), len(graphs_s))

    if write_instance:
        graphs_i = (
            list(_gn_filter)
            if _gn_filter is not None
            else _lg.list_instance_signal_names()
        )
        # query_per_instance filters by a single (sample_id, annotation_id); iterate when multiple given
        _sid_iter = _sid_filter if _sid_filter is not None else [None]
        _aid_iter = _aid_filter if _aid_filter is not None else [None]
        for gn in graphs_i:
            for _sid in _sid_iter:
                for _aid in _aid_iter:
                    for sid, aid, step, val, h in _lg.query_per_instance(
                        gn,
                        sample_id=_sid,
                        annotation_id=_aid,
                        exp_hash=experiment_hash,
                    ):
                        instance_rows.append({
                            "graph_name": gn,
                            "experiment_hash": h if h is not None else "",
                            "sample_id": sid,
                            "annotation_id": aid,
                            "step": step,
                            "metric_value": val,
                        })
        logger.debug("write_history: collected %d instance row(s) across %d "
                     "graph(s).", len(instance_rows), len(graphs_i))

    if fmt == "json":
        payload = {}
        if write_global:
            payload["global"] = global_rows
        if write_sample:
            payload["sample"] = sample_rows
        if write_instance:
            payload["instance"] = instance_rows
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2)

    elif fmt == "csv":
        _CSV_FIELDS = [
            "type", "graph_name", "experiment_hash", "step", "metric_value",
            "sample_id", "annotation_id",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in global_rows:
                writer.writerow({"type": "global", **row})
            for row in sample_rows:
                writer.writerow({"type": "sample", **row})
            for row in instance_rows:
                writer.writerow({"type": "instance", **row})

    else:
        logger.error("write_history: unsupported format %r (expected 'json' "
                     "or 'csv').", format)
        raise ValueError(
            f"write_history: unsupported format {format!r}. Use 'json' or 'csv'."
        )

    _total = len(global_rows) + len(sample_rows) + len(instance_rows)
    logger.info(
        "write_history: wrote %d row(s) (global=%d, sample=%d, instance=%d) "
        "as %s to %s",
        _total, len(global_rows), len(sample_rows), len(instance_rows),
        fmt, _os.path.abspath(path),
    )
    if _total == 0:
        logger.warning(
            "write_history: output is empty — no rows matched the given "
            "filters (type=%r, experiment_hash=%r). Check that the signals "
            "have been logged for the requested experiment hash.",
            _type, experiment_hash,
        )

    return path


def write_dataframe(
    path: str | None = None,
    format: str = "json",
    columns=None,
    sample_id=None,
    instance_id=None,
) -> str:
    """Dump the WeightsLab sample dataframe to *path* as JSON or CSV.

    Parameters
    ----------
    path : str, optional
        Output file path **or** directory.  When omitted (``None``), the
        ``root_log_dir`` from the active checkpoint manager is used.

        - If *path* has a file extension the file is written directly.
        - If *path* has no extension or is an existing directory, a filename is
          auto-generated as ``<hash>_dataframe.<format>`` inside that directory.
          ``<hash>`` is an 8-character MD5 hex digest of the normalized call
          parameters (*columns*, *sample_id*, *instance_id*).  Same filters →
          same filename (idempotent overwrite); different filters → different
          file.
        - The directory is created automatically if it does not exist.
    format : {"json", "csv"}
        Output format.  Default ``"json"``.
    columns : str or list of str, optional
        Which columns to include (index levels ``sample_id`` / ``annotation_id``
        are always written).

        - ``None`` / ``"all"`` — every column.
        - ``"tags"`` — only columns prefixed with ``tag:`` (categorical and
          boolean tags, e.g. ``tag:loss_shape``, ``tag:weather``).
        - ``"signals"`` — only columns prefixed with ``signals`` (per-sample
          signals logged by ``wl.watch_or_edit`` or ``wl.save_signals``).
        - ``"discarded"`` — only the boolean ``discarded`` column.
        - A list of any mix of the above group names and/or exact column names.
    sample_id : str or list of str, optional
        Restrict to one or more sample IDs (index level 0).  ``None`` keeps all.
    instance_id : int or list of int, optional
        Restrict to one or more annotation IDs (index level 1, 0 = sample row,
        ≥ 1 = per-instance rows).  ``None`` keeps all.

    Returns
    -------
    str
        Absolute path of the file that was written.

    Notes
    -----
    The function calls ``flush()`` on the dataframe manager before reading so
    that any in-flight writes are included in the output.  Pass
    ``instance_id=0`` to keep only sample-level rows; pass ``instance_id=[1,2]``
    to keep specific annotation rows.

    Examples
    --------
    Dump everything (path inferred from ``root_log_dir``)::

        wl.write_dataframe()

    Dump only tags to CSV::

        wl.write_dataframe("tags.csv", format="csv", columns="tags")

    Dump signals + discarded for specific samples::

        wl.write_dataframe(
            "subset.json",
            columns=["signals", "discarded"],
            sample_id=["img_001", "img_042"],
        )
    """
    import json as _json
    import os as _os
    import hashlib as _hashlib

    logger.debug(
        "write_dataframe called: path=%r, format=%r, columns=%r, "
        "sample_id=%r, instance_id=%r",
        path, format, columns, sample_id, instance_id,
    )

    _dm = get_dataframe()
    if _dm is None:
        logger.warning(
            "write_dataframe: no active dataframe manager; nothing to write. "
            "Returning path=%r.", path or "."
        )
        return path or "."

    # Resolve path: fall back to root_log_dir from the checkpoint manager
    if path is None:
        _lg = get_logger()
        try:
            _cm = _lg.chkpt_manager if _lg is not None else None
            _rld = _cm.root_log_dir if _cm is not None else None
            path = str(_rld) if _rld is not None else "."
        except Exception as _e:
            logger.debug("write_dataframe: failed to resolve root_log_dir (%s); "
                         "falling back to current directory.", _e)
            path = "."
        logger.info("write_dataframe: no path given, using output directory %r.", path)

    fmt = format.lower().strip()

    # Normalize sample_id → list[str] or None
    _sid_filter = None
    if sample_id is not None:
        _sid_filter = [str(sample_id)] if isinstance(sample_id, str) else [str(s) for s in sample_id]

    # Normalize instance_id → list[int] or None
    _iid_filter = None
    if instance_id is not None:
        _iid_filter = [int(instance_id)] if isinstance(instance_id, int) else [int(x) for x in instance_id]

    # Normalize columns filter
    _col_filter = None
    if columns is not None and not (isinstance(columns, str) and columns.lower() == "all"):
        _col_filter = [columns] if isinstance(columns, str) else list(columns)

    logger.info(
        "write_dataframe: resolved filters → columns=%s, sample_id=%s, instance_id=%s",
        _col_filter if _col_filter is not None else "<all>",
        _sid_filter if _sid_filter is not None else "<all>",
        _iid_filter if _iid_filter is not None else "<all>",
    )

    # Resolve output path (same convention as write_history)
    _base = _os.path.basename(path)
    if not _os.path.splitext(_base)[1] or _os.path.isdir(path):
        _params_key = (
            "dataframe",
            tuple(sorted(_col_filter)) if _col_filter is not None else None,
            tuple(sorted(_sid_filter)) if _sid_filter is not None else None,
            tuple(sorted(_iid_filter)) if _iid_filter is not None else None,
        )
        _phash = _hashlib.md5(str(_params_key).encode()).hexdigest()[:8]
        path = _os.path.join(path, f"{_phash}_dataframe.{fmt}")
        logger.info("write_dataframe: auto-generated filename %r (params hash %s).",
                    _os.path.basename(path), _phash)
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    logger.info("write_dataframe: output file → %s", _os.path.abspath(path))

    # Flush pending buffer to H5 before reading
    try:
        _dm.flush()
        logger.debug("write_dataframe: buffer flushed to H5.")
    except Exception as _e:
        logger.warning("write_dataframe: flush failed (%s); proceeding with "
                       "in-memory data only.", _e)

    # Retrieve the full dataframe
    try:
        _df = _dm.get_combined_df()
    except Exception as _e:
        logger.error("write_dataframe: failed to retrieve dataframe (%s).", _e)
        raise

    import pandas as _pd
    if _df is None or _df.empty:
        logger.warning("write_dataframe: dataframe is empty; writing empty output.")
        _df = _pd.DataFrame()

    df_out = _df

    # Filter by sample_id (MultiIndex level 0)
    if _sid_filter is not None and not df_out.empty:
        _sid_set = set(_sid_filter)
        mask = df_out.index.get_level_values(0).astype(str).isin(_sid_set)
        df_out = df_out.loc[mask]
        logger.debug("write_dataframe: after sample_id filter → %d row(s).", len(df_out))

    # Filter by instance_id / annotation_id (MultiIndex level 1)
    if _iid_filter is not None and not df_out.empty:
        _iid_set = set(_iid_filter)
        try:
            mask = df_out.index.get_level_values(1).astype(int).isin(_iid_set)
            df_out = df_out.loc[mask]
        except Exception:
            pass  # non-integer annotation_ids — skip this filter
        logger.debug("write_dataframe: after instance_id filter → %d row(s).", len(df_out))

    # Filter columns by group or exact name
    if _col_filter is not None and not df_out.empty:
        _selected: list = []
        for _item in _col_filter:
            _lc = str(_item).lower()
            if _lc == "tags":
                _selected += [
                    c for c in df_out.columns
                    if str(c).startswith("tag:") or str(c).startswith("TAG:")
                ]
            elif _lc == "signals":
                _selected += [
                    c for c in df_out.columns
                    if str(c).lower().startswith("signals")
                ]
            elif _lc == "discarded":
                if "discarded" in df_out.columns:
                    _selected.append("discarded")
            else:
                if _item in df_out.columns:
                    _selected.append(_item)
        _selected = list(dict.fromkeys(_selected))  # deduplicate, preserve order
        df_out = df_out[_selected] if _selected else df_out[[]]
        logger.debug("write_dataframe: column filter → %d column(s): %s",
                     len(_selected), _selected)

    # Reset index so sample_id / annotation_id appear as regular columns in output
    df_out = df_out.reset_index()

    if fmt == "json":
        _json_str = df_out.to_json(orient="records", default_handler=str)
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(_json.loads(_json_str), fh, indent=2)

    elif fmt == "csv":
        df_out.to_csv(path, index=False, encoding="utf-8")

    else:
        logger.error("write_dataframe: unsupported format %r (expected 'json' or 'csv').", format)
        raise ValueError(
            f"write_dataframe: unsupported format {format!r}. Use 'json' or 'csv'."
        )

    logger.info(
        "write_dataframe: wrote %d row(s) × %d column(s) as %s to %s",
        len(df_out), len(df_out.columns), fmt, _os.path.abspath(path),
    )
    if df_out.empty:
        logger.warning(
            "write_dataframe: output is empty — no rows matched the given filters "
            "(sample_id=%r, instance_id=%r).",
            _sid_filter, _iid_filter,
        )

    return path


# ##############################################################################################################
# MAIN EXAMPLES USAGE (can be called from training script to manually set)
# ##############################################################################################################

if __name__ == "__main__":
    from weightslab import tag_samples, discard_samples, get_samples_by_tag

    # Init the global dataframe manager
    # ...

    # In your training script after registering dataloader:
    # Tag difficult samples
    tag_samples([0, 5, 12, 19], 'difficult', origin='train', mode='add')

    # Tag outliers
    tag_samples([5, 12], 'outlier', origin='train', mode='add')

    # Discard corrupted samples
    discard_samples([99, 100], discarded=True, origin='train')

    # Query and process tagged samples
    difficult_ids = get_samples_by_tag('difficult', origin='train')
    logger.info(f"Found {len(difficult_ids)} difficult samples")

    # Remove tag after review
    tag_samples([5], 'outlier', mode='remove')

    # Keep script alive
    keep_serving(timeout=60)
