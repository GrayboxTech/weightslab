""" The Experiment class is the main class of the graybox package.
It is used to train and evaluate models. """

import sys
import time
import functools
import logging
import torch as th

from typing import Callable
from threading import Lock

from weightslab.backend.model_interface import ModelInterface
from weightslab.backend.dataloader_interface import DataLoaderInterface
from weightslab.backend.optimizer_interface import OptimizerInterface
from weightslab.backend.ledgers import get_model, get_dataloader, get_optimizer, register_hyperparams, watch_hyperparams_file, get_hyperparams, register_logger, get_logger, register_signal, get_signal
from weightslab.backend.cli import cli_serve
from weightslab.trainer.trainer_services import grpc_serve
from weightslab.ui.weightslab_ui import ui_serve


# Get global logger
logger = logging.getLogger(__name__)


def _save_data_statistics(
    model_age: int,
    batch_ids: th.Tensor,
    losses_batch: th.Tensor,
    preds: th.Tensor = None,
    lock: Lock = Lock()
):
    with lock:
        # Get batch data
        pred_np = preds.detach().cpu().numpy() if preds is not None else None
        batch_ids_np = batch_ids.detach().cpu().numpy()
        if not isinstance(losses_batch, dict):
            per_sample_loss_np = losses_batch.detach().cpu().numpy()
        else:
            for k in losses_batch:
                losses_batch[k] = losses_batch[k].detach().cpu().numpy()
            per_sample_loss_np = losses_batch

        # Update batch sample stats
        # Check if model is in training mode with grad enabled
        is_training = th.is_grad_enabled()

        name = 'train_loader' if is_training else 'test_loader'
        try:
            get_dataloader(name).tracked_dataset.update_batch_sample_stats(
                model_age,
                batch_ids_np,
                per_sample_loss_np,
                pred_np
            )
            get_dataloader(name).tracked_dataset.update_sample_stats_ex_batch(
                batch_ids_np,
                {
                    "loss/combined": per_sample_loss_np,
                    "pred": pred_np
                }
            )
        except AttributeError as e:
            logger.warning(
                "Warning: Could not save data statistics to " +
                f"tracked dataset: {e}. Please ensure the " +
                "dataloader has a tracked dataset with " +
                "`update_batch_sample_stats` and " +
                "`update_sample_stats_ex_batch` methods, and is_training set."
            )

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

    # Related functions
    if flag.lower() == 'model' or (hasattr(obj, '__name__') and 'model' in obj.__name__.lower()):
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

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy is not None else wrapper

    elif flag.lower() == 'data' or flag.lower() == 'dataset' or flag.lower() == 'dataloader' or (hasattr(obj, '__name__') and 'data' in obj.__name__.lower()):
        reg_name = kwargs.get('name') or getattr(getattr(obj, 'dataset', obj), '__name__', None) or getattr(getattr(obj, 'dataset', obj), '__class__', type(getattr(obj, 'dataset', obj))).__name__
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_dataloader` will create a Proxy if
        # the name is not yet present.
        try:
            proxy = get_dataloader(reg_name)
        except Exception:
            proxy = None

        # Auto-inject root_log_dir from hyperparameters if not provided
        if 'root_log_dir' not in kwargs:
            try:
                from weightslab.backend.ledgers import list_hyperparams
                hps = list_hyperparams()
                if hps:
                    # Get the first hyperparameters object (usually there's only one)
                    if isinstance(hps, dict):
                        hp_name = list(hps.keys())[0]
                        hp_dict = hps[hp_name]
                    else:
                        hp_name = hps[0]
                        hp_dict = get_hyperparams(hp_name)
                    if isinstance(hp_dict, dict) and 'root_log_dir' in hp_dict:
                        kwargs['root_log_dir'] = hp_dict['root_log_dir']
            except Exception:
                pass  # If we can't get hyperparameters, continue without root_log_dir

        # Now construct the wrapper and let it register into the ledger.
        wrapper = DataLoaderInterface(obj, **kwargs)

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy is not None else wrapper

    elif flag.lower() == 'optimizer' or (hasattr(obj, '__name__') and 'opt' in obj.__name__.lower()):
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

    elif flag.lower() == 'logger' or (hasattr(obj, '__name__') and 'log' in obj.__name__.lower()):
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

    # Signals: metrics / losses / custom monitors
    elif 'loss' in flag.lower() or flag.lower() in ('criterion', 'signal', 'signals', 'watch'):
        # derive registration name from second part of flag if provided
        reg_name = kwargs.get('name') or flag

        # decide how to wrap: loss-like (forward) or metric-like (compute)
        # wrap forward
        try:
            if hasattr(obj, 'forward') and callable(getattr(obj, 'forward')):
                original_forward = obj.forward

                # New forward with logging and save stats
                @functools.wraps(original_forward)
                def new_forward(*a, **kw):
                    # Remove parameters
                    _ = kw.pop('flag', None)
                    ids = kw.pop('batch_ids', None)
                    model_age = kw.pop('model_age', None)
                    preds = kw.pop('preds', None)
                    manual_losses_batch = kw.pop('losses_batch', None)

                    # Original forward
                    out = original_forward(*a, **kw)

                    if kwargs.get('per_sample', False):
                        out = out.flatten(1).mean(dim=1)  # Works for any shape [B, ...]

                    # extract scalar
                    batch_scalar = manual_losses_batch
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
                                    m = get_model()
                                    step = int(m.get_age())
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
                    if batch_scalar is not None and ids is not None and model_age is not None:
                        _save_data_statistics(
                            model_age=model_age,
                            batch_ids=ids,
                            losses_batch=batch_scalar,
                            preds=preds
                        )
                    return out

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


    elif 'metric' in flag.lower() or flag.lower() in ('signal', 'signals', 'watch'):
        # derive registration name from second part of flag if provided
        reg_name = kwargs.get('name') or flag

        # decide how to wrap: loss-like (forward) or metric-like (compute)
        # wrap forward
        try:
            if hasattr(obj, 'compute') and callable(getattr(obj, 'compute')):
                original_compute = obj.compute

                @functools.wraps(original_compute)
                def new_compute(*a, **kw):
                    out = original_compute(*a, **kw)
                    if kwargs.get('per_sample', False):
                        out = out.flatten(1).mean(dim=1)  # Works for any shape [B, ...]
                    # extract scalar
                    try:
                        if isinstance(out, th.Tensor):
                            scalar = float(out.detach().cpu().mean().item())
                        else:
                            try:
                                import numpy as _np
                                scalar = float(_np.array(out).mean())
                            except Exception:
                                scalar = None
                    except Exception:
                        scalar = None

                    if kwargs.get('log', False) and scalar is not None:
                        try:
                            logger = None
                            try:
                                logger = get_logger()
                            except Exception:
                                try:
                                    logger = get_logger('main')
                                except Exception:
                                    logger = None
                            if logger is not None and hasattr(logger, 'add_scalars'):
                                step = 0
                                try:
                                    m = get_model()
                                    step = int(m.get_age())
                                except Exception:
                                    step = 0
                                logger.add_scalars(reg_name, {reg_name: scalar}, global_step=step)
                        except Exception:
                            pass

                    return out

                obj.compute = new_compute

            elif hasattr(obj, 'forward') and callable(getattr(obj, 'forward')):
                original_forward = obj.forward

                @functools.wraps(original_forward)
                def new_forward(*a, **kw):
                    out = original_forward(*a, **kw)

                    # extract scalar
                    try:
                        if isinstance(out, th.Tensor):
                            scalar = float(out.detach().cpu().mean().item())
                        else:
                            try:
                                import numpy as _np
                                scalar = float(_np.array(out).mean())
                            except Exception:
                                scalar = None
                    except Exception:
                        scalar = None

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
                                    m = get_model()
                                    step = int(m.get_age())
                                except Exception:
                                    step = 0
                                logger.add_scalars(reg_name, {reg_name: scalar}, global_step=step)
                        except Exception:
                            pass

                    return out

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
            except Exception:
                # bubble up original error
                raise

        raise ValueError(f"Obj name {obj} should contains at least 'model', 'data' or 'optimizer'.")


def serve(serving_ui: bool = False, serving_cli: bool = False, serving_grpc: bool = False, **kwargs) -> None:
    """ Serve the trainer services.

    Args:
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
