""" The Experiment class is the main class of the graybox package.
It is used to train and evaluate models. """
import os
import sys
import time
import functools
import logging
import numpy as np
import torch as th

from typing import Callable

from weightslab.backend.model_interface import ModelInterface
from weightslab.backend.dataloader_interface import DataLoaderInterface
from weightslab.backend.optimizer_interface import OptimizerInterface
from weightslab.backend.ledgers import get_checkpoint_manager, list_hyperparams, register_checkpoint_manager, get_model, get_dataloader, get_dataframe, get_optimizer, register_hyperparams, watch_hyperparams_file, get_hyperparams, register_logger, get_logger, register_signal, get_signal
from weightslab.backend.cli import cli_serve
from weightslab.trainer.trainer_services import grpc_serve
from weightslab.ui.weightslab_ui import ui_serve
from weightslab.utils.logger import LoggerQueue
from weightslab.backend import ledgers
from weightslab.components.checkpoint_manager_v2 import CheckpointManagerV2


# Get global logger
logger = logging.getLogger(__name__)
# Get global dataframe proxy (auto-updated when ledger registers real manager)
DATAFRAME_M = None


def save_signals(
    batch_ids: th.Tensor,
    signals: dict,
    preds_raw: th.Tensor,
    targets: th.Tensor,
    preds: th.Tensor = None
):
    """
        Save data statistics to the tracked dataset.
        Args:
            batch_ids (th.Tensor): The batch ids.
            signals (th.Tensor): The batch losses.
            preds (th.Tensor, optional): The batch predictions. Defaults to None.
    """
    global DATAFRAME_M
    if DATAFRAME_M is None:
        DATAFRAME_M = get_dataframe()

    # Convert tensors to numpy for lightweight buffering
    batch_ids_np = batch_ids.detach().cpu().numpy().astype(int)
    pred_np = preds.detach().cpu().numpy().astype(np.uint16) if preds is not None else None
    pred_raw_np = preds_raw.detach().cpu().numpy().astype(float) if preds_raw is not None else None
    target_np = targets.detach().cpu().numpy().astype(np.uint16) if targets is not None else None

    # Processing
    # # Process signals
    if isinstance(signals, dict):
        losses_data = {\
            # Convert losses map of shape (B, ...) to (B,) by averaging spatial dimensions if needed
            'signals//' + k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)).mean((1, 2)) if v.ndim > 2 else (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v))
            for k, v in signals.items()
        }
    elif signals is not None:
        losses_data = {"signals//default": signals.detach().cpu().numpy() if hasattr(signals, 'detach') else np.asarray(signals)}
    else:
        losses_data = None
    # # Process targets
    if target_np.ndim == 1:
        target_np = target_np[:, np.newaxis]
    if pred_np is not None and pred_np.ndim == 1:
        pred_np = pred_np[:, np.newaxis]
    if pred_raw_np is not None and pred_raw_np.ndim == 1:
        pred_raw_np = pred_raw_np[:, np.newaxis]

    # Enqueue to dataframe manager buffer for efficientcy
    enqueue = getattr(DATAFRAME_M, 'enqueue_batch', None)
    if callable(enqueue):
        enqueue(
            sample_ids=batch_ids_np,
            preds_raw=pred_raw_np,
            preds=pred_np,
            targets=target_np,
            losses=losses_data,
        )

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

    # Remove parameters
    _ = kw.pop('flag', None)
    ids = kw.pop('batch_ids', None)
    preds = kw.pop('preds', None)
    manual_signals_batch = kw.pop('signals', None)
    preds_raw = a[0]
    targets = a[1]

    # Original forward of the signal
    out = original_forward(*a, **kw)

    if kwargs.get('per_sample', False):
        if out.ndim > 1:
            out = out.mean(dim=tuple(range(1, out.ndim)))  # Reduce to [B,]

    # extract scalar
    batch_scalar = manual_signals_batch
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
            else:
                try:
                    import numpy as _np
                    batch_scalar = _np.array(batch_scalar)
                    scalar = float(batch_scalar.mean())
                except Exception:
                    pass
        # 2. Otherwise fall back to extracting from 'out'
        elif isinstance(out, th.Tensor):
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
    except Exception:
        pass

    # log if requested and logger present
    if kwargs.get('log', False) and scalar is not None:
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
                # attempt to get a sensible global_step
                step = 0
                try:
                    m = None
                    try:
                        m = get_model()
                    except Exception:
                        # Fallback: if get_model() failed (e.g. ambiguity), try to find a valid model
                        from weightslab.backend.ledgers import list_models, get_model as _gm
                        full_list = list_models()
                        if full_list:
                             # Prefer "experiment" or "main" or the first one
                            if 'experiment' in full_list:
                                m = _gm('experiment')
                            elif 'main' in full_list:
                                m = _gm('main')
                            else:
                                m = _gm(full_list[0])

                    if m is not None:
                        # Safe attribute access (handle Proxy returning None for missing attr)
                        val = getattr(m, 'current_step', None)
                        if val is not None:
                            step = int(val)
                        else:
                            val_age = getattr(m, 'get_age', None)
                            if callable(val_age):
                                step = int(val_age())
                            else:
                                step = 0
                except Exception:
                    step = 0
                logger.add_scalars(
                    reg_name,
                    {reg_name: scalar},
                    global_step=step
                )
        except Exception:
            pass

    # Save statistics if requested and applicable
    if batch_scalar is not None and ids is not None:
        signals = {reg_name: batch_scalar}
        save_signals(
            batch_ids=ids,
            preds=preds,
            preds_raw=preds_raw,
            signals=signals,
            targets=targets
        )
    return out


def watch_or_edit(obj: Callable, obj_name: str = None, flag: str = None, **kwargs) -> None:
    """
    Watch or edit the given object.

    Args:
        obj (Callable): The object to watch or edit.
        flag (str): The flag specifying the type of object to watch or
        edit.
        kwargs (Any): Additional keyword arguments to pass.
    """

    # Sanity check
    if not hasattr(obj, '__name__'):
        if obj_name is None and 'name' not in kwargs:
            try:
                obj.__name__ = type(obj).__name__
            except Exception:
                obj.__name__ = str(time.time())
            logger.warning(
                "Warning: Watching or editing anonymous object '" +
                f"{obj.__name__}'."
            )
            logger.warning(
                "Please add a 'name' attribute to the object."
            )
        else:
            if hasattr(obj, '__name__'):
                obj.__name__ = obj_name

    # Model
    if 'model' in flag.lower() or (hasattr(obj, '__name__') and 'model' in obj.__name__.lower()):
        # Derive a sane registration name: prefer explicit `name` kwarg,
        # then a meaningful __name__ if it is not the generic 'model',
        # then the class name. This avoids accidental registration under
        # the literal 'model' which can lead to duplicates.
        if kwargs.get('name'):
            reg_name = kwargs.get('name')
        else:
            candidate = getattr(obj, '__name__', None)
            if candidate and candidate.lower() != 'model':
                reg_name = candidate
            else:
                clsname = getattr(obj.__class__, '__name__', None)
                reg_name = clsname if clsname and clsname.lower() != 'model' else (kwargs.get('name') or 'model')

        # First ensure that the model has module input_shape
        if not hasattr(obj, 'input_shape'):
            raise ValueError("Model object must have 'input_shape' attribute for proper registration with WeightsLab.")

        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_model` will create a Proxy if
        # the name is not yet present.
        try:
            proxy = get_model(reg_name)
        except Exception:
            proxy = None

        # Now construct the wrapper and let it register into the ledger.
        wrapper = ModelInterface(obj, **kwargs)

        # Register related logger for model training
        # # Init the logger
        LoggerQueue(name=reg_name)

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy is not None else wrapper

    # DataLoader
    elif 'data' in flag.lower() or flag.lower() == 'dataset' or flag.lower() == 'dataloader' or (hasattr(obj, '__name__') and 'data' in obj.__name__.lower()):
        reg_name = kwargs.get('name') or getattr(getattr(obj, 'dataset', obj), '__name__', None) or getattr(getattr(obj, 'dataset', obj), '__class__', type(getattr(obj, 'dataset', obj))).__name__
        kwargs['name'] = reg_name

        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_dataloader` will create a Proxy if
        # the name is not yet present.
        try:
            proxy = get_dataloader(reg_name)
        except Exception:
            proxy = None

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
                            u:v for u,v in hp_dict.get('data', {}).get(reg_name, {}).items() if u not in kwargs
                        }
                    )
            except Exception:
                pass  # If we can't get hyperparameters, continue without root_log_dir

        # Now construct the wrapper and let it register into the ledger.
        wrapper = DataLoaderInterface(obj, **kwargs)

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy is not None else wrapper

    # Optimizer
    elif 'optimizer' in flag.lower() or (hasattr(obj, '__name__') and 'opt' in obj.__name__.lower()):
        # Determine registration name first
        reg_name = kwargs.get('name') or getattr(obj, '__name__', None) or getattr(obj, '__class__', type(obj)).__name__ or '_optimizer'
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_optimizer` will create a Proxy if
        # the name is not yet present.
        try:
            proxy = get_optimizer(reg_name)
        except Exception:
            proxy = None

        # Now construct the wrapper and let it register into the ledger.
        wrapper = OptimizerInterface(obj, **kwargs)

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy is not None else wrapper

    # Logger
    elif 'logger' in flag.lower() or (hasattr(obj, '__name__') and 'log' in obj.__name__.lower()):
        # Determine registration name for the logger (prefer explicit name)
        reg_name = kwargs.get('name') or getattr(obj, '__name__', None) or getattr(obj.__class__, '__name__', None) or 'main'
        # Ensure there's a proxy placeholder if callers already requested the logger
        try:
            proxy = get_logger(reg_name)
        except Exception:
            proxy = None

        # Register the logger into the ledger. This will update any proxy in-place.
        register_logger(reg_name, obj)

        # Return a stable handle (proxy) when available, otherwise the registered logger
        return proxy if proxy is not None else get_logger(reg_name)

    # Signals
    # # Loss
    elif 'loss' in flag.lower() or flag.lower() in ('criterion', 'signal', 'signals', 'watch'):
        # derive registration name from second part of flag if provided
        reg_name = kwargs.get('name') or flag
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
                register_signal(reg_name, obj)
            except Exception:
                pass

            # return proxy if exists else the object
            try:
                return get_signal(reg_name)
            except Exception:
                return obj
        except Exception:
            # fall back to hyperparams branch if something unexpected
            pass
    # # Metric
    elif 'metric' in flag.lower() or flag.lower() in ('signal', 'signals', 'watch'):
        # derive registration name from second part of flag if provided
        reg_name = kwargs.get('name') or flag

        # decide how to wrap: loss-like (forward) or metric-like (compute)
        # wrap forward
        try:
            if hasattr(obj, 'compute') and callable(getattr(obj, 'compute')):
                original_forward = obj.compute

                # New forward with logging and save stats
                @functools.wraps(original_forward)
                def new_forward(*a, **kw):
                    return wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw)
                obj.forward = new_forward

            elif hasattr(obj, 'forward') and callable(getattr(obj, 'forward')):
                original_forward = obj.forward

                # New forward with logging and save stats
                @functools.wraps(original_forward)
                def new_forward(*a, **kw):
                    return wrappered_fwd(original_forward, kwargs, reg_name, *a, **kw)
                obj.forward = new_forward

            # register wrapped signal in ledger
            try:
                register_signal(reg_name, obj)
            except Exception:
                pass

            # return proxy if exists else the object
            try:
                return get_signal(reg_name)
            except Exception:
                return obj
        except Exception:
            # fall back to hyperparams branch if something unexpected
            pass

    # Hyper parameters
    else:
        # Support hyperparameters/watchable parameter dicts or YAML paths.
        if flag is None:
            raise ValueError("Obj name should contains at least 'model', 'data', 'optimizer' or 'hp'.")

        fl = flag.lower()
        if fl in ('hp', 'hyperparams', 'params', 'hyperparameters', 'parameters'):
            # obj may be a dict of parameters or a path to a YAML file
            name = kwargs.get('name') or getattr(obj, '__name__', None) or 'hyperparams'
            # If obj is a string, treat as a file path and start watcher
            try:
                # Initialize CheckpointManagerV2 if we have a root dir (fallback to default root)
                root_log_dir = obj.get('root_log_dir') or os.path.join('.', 'root_log_dir')
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
                        _checkpoint_manager = CheckpointManagerV2(root_log_dir=root_log_dir)
                        try:
                            ledgers.register_checkpoint_manager('default', _checkpoint_manager)
                            logger.info("Registered new checkpoint manager in ledger")
                        except Exception:
                            pass
                except Exception:
                    _checkpoint_manager = None

                # Check if hyperparameters are available in checkpoint manager
                checkpoint_hp_loaded = False
                try:
                    chkpt_manager = get_checkpoint_manager()
                    if chkpt_manager is not None and not isinstance(chkpt_manager, ledgers.Proxy):
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
                                load_config=True,
                                force=True,
                                load_data=False
                            )
                            if checkpoint_data.get('config'):
                                config = checkpoint_data['config']
                                register_hyperparams(name, config)
                                logger.info(f"Loaded hyperparameters from checkpoint {latest_hash[:16]}")
                                checkpoint_hp_loaded = True
                except Exception:
                    pass  # If checkpoint loading fails, proceed with normal registration

                if not checkpoint_hp_loaded:
                    # Normal registration if no checkpoint hyperparameters were loaded
                    if isinstance(obj, str):
                        path = obj
                        # register empty/defaults if provided in kwargs
                        defaults = kwargs.get('defaults', None)
                        if defaults:
                            register_hyperparams(name, defaults)
                        # start ledger-managed watcher
                        watch_hyperparams_file(name, path, poll_interval=kwargs.get('poll_interval', 1.0))

                        # return the ledger handle (proxy or dict)
                        return get_hyperparams(name)
                    elif isinstance(obj, dict):
                        register_hyperparams(name, obj)

                        return get_hyperparams(name)
                    else:
                        # unsupported type for hp; attempt best-effort registration
                        try:
                            register_hyperparams(name, dict(obj))

                            return get_hyperparams(name)
                        except Exception:
                            raise ValueError('Unsupported hyperparams object; provide dict or YAML path')

                return get_hyperparams(name)
            except Exception:
                # bubble up original error
                raise

        raise ValueError(f"Obj name {obj} should contains at least 'model', 'data' or 'optimizer'.")


def serve(serving_ui: bool = False, serving_cli: bool = False, serving_grpc: bool = False, **kwargs) -> None:
    """ Serve the trainer services.

    Args:
        serving_ui (bool): Whether to serve the UI.
        serving_cli (bool): Whether to use the CLI.
        serving_grpc (bool): Whether to serve gRPC.
    """

    # Sanity check
    if serving_ui and not serving_grpc:
        logger.error("UI server requires gRPC server to be enabled.")
        sys.exit(1)

    if serving_grpc:
        grpc_serve(**kwargs)

    if serving_ui and serving_grpc:
        ui_serve(**kwargs)

    if serving_cli:
        cli_serve(**kwargs)


def keep_serving():
    """ Keep the main thread alive to allow background serving threads to run.
    """
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down WeightsLab services.")
