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

from weightslab.backend.dataloader_interface import DataLoaderInterface
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.backend.optimizer_interface import OptimizerInterface
from weightslab.backend.ledgers import DEFAULT_NAME, get_checkpoint_manager, get_model, get_dataloader, get_dataframe, get_optimizer, register_hyperparams, watch_hyperparams_file, get_hyperparams, register_logger, get_logger, register_signal, get_signal, list_models
from weightslab.backend.model_interface import ModelInterface
from weightslab.trainer.trainer_services import grpc_serve
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.utils.logs import set_log_directory
from weightslab.ui.weightslab_ui import ui_serve
from weightslab.utils.logger import LoggerQueue
from weightslab.backend.cli import cli_serve
from weightslab.backend import ledgers
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.backend.ledgers import register_signal
from weightslab.backend.ledgers import list_models, get_model as _gm
from tqdm import tqdm


# Get global logger
logger = logging.getLogger(__name__)
# Get global dataframe proxy (auto-updated when ledger registers real manager)
DATAFRAME_M = None
# Global registry for custom signals
_REGISTERED_SIGNALS = {}


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


def _get_step(step: int | None = None) -> int:
    """
        Attempt to get the current training step from the model in the ledger, if available. This is used for logging signals with the correct global step.
        The function will try multiple approaches to find a valid step value:
        1. Try to get the model from the ledger and access `get_age()` if
              available (preferred for models that track age/step internally).
        2. If `get_age()` is not available, try to access `current_step` attribute on the model (common pattern for external checkpoint managers).
        3. If neither is available, fall back to 0 or the provided step argument
                and add a `current_step` attribute to the model for future tracking.
    """
    # Fallback: if get_model() failed (e.g. ambiguity), try to find a valid model
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
            val = m.get_age()
            if val is not None:
                step = int(val) -1
        elif hasattr(m, 'current_step'):
            val = m.current_step
            if val is not None:
                step = int(val) -1
            elif step is not None:
                # step = step # fallback to provided step
                m.current_step = step  # add current_step attribute to model for future tracking
        elif step is not None:
            # If model doesn't have current_step, force it to 0 or try to infer from checkpoint manager
            m.current_step = step  # add current_step attribute to model for future tracking

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
            else:
                try:
                    import numpy as _np
                    batch_scalar = _np.array(batch_scalar)
                    scalar = float(batch_scalar.mean())
                except Exception:
                    pass
            # Merged batch scalar with ids
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
                    signal_per_sample=signal_per_sample
                )
        except Exception:
            pass


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
    batch_scalar = kw.pop('signals', None)
    preds_raw = a[0] if len(a) > 0 else None
    targets = a[1] if len(a) > 1 else None

    # Original forward of the signal
    out = original_forward(*a, **kw)
    if kwargs.get('per_sample', False):
        if out.ndim > 1:
            out = out.mean(dim=tuple(range(1, out.ndim)))  # Reduce to [B,]0

    # Extract scalar from tensor
    scalar, batch_scalar = _extract_scalar_from_tensor(batch_scalar, out, ids)

    # Log if requested
    step = _get_step(None)
    _log_signal(scalar, batch_scalar, reg_name, step=step, **kwargs)

    # CHECK FOR SUBSCRIBERS (Dynamic Signals)
    # Allows @wl.signal(subscribe_to="metric_name")
    dynamic_updates = {}
    ids_np = None
    if ids is not None:
        ids_np = ids.detach().cpu().numpy() if hasattr(ids, 'detach') else np.asarray(ids)

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
                         batch_res = []
                         for i, uid in enumerate(ids_np):
                             # Generic 'value' argument
                             val = float(val_vec[i])
                             # Inject dataframe so user function is pure-ish
                             res = func(sample_id=int(uid), value=val, dataframe=df_proxy)
                             batch_res.append(res)
                         dynamic_updates[name] = np.array(batch_res)
                     except Exception:
                         pass # User function error, skip
    
    # Save statistics if requested and applicable
    if (batch_scalar is not None and ids is not None) or dynamic_updates:
        signals = {
            reg_name: list(batch_scalar.values()) if isinstance(batch_scalar, dict) else batch_scalar.detach().cpu().tolist()
        } if batch_scalar is not None else {}
        signals.update(dynamic_updates) # Merge dynamic signals
        
        save_signals(
            batch_ids=ids,
            preds=preds,
            preds_raw=preds_raw,
            signals=signals,
            targets=targets,
            log=False  # Already logged above, no need to log again in save_signals; set to False to avoid duplicate logging if save_signals is called separately without logging
        )
    return out


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
    

# ##############################################################################################################
# USER FUNCTION EXPOSED TO SERVE SIGNALS, TAG SAMPLES, ETC. (can be called from training script to manually set)
# ##############################################################################################################

def watch_or_edit(obj: Callable, obj_name: str = None, flag: str = None, **kwargs) -> None:
    """
    Watch or edit the given object.

    Args:
        obj (Callable): The object to watch or edit.
        flag (str): The flag specifying the type of object to watch or
        edit.
        kwargs (Any): Additional keyword arguments to pass.
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

    # Model
    if 'model' in flag.lower() or (hasattr(obj, '__name__') and 'model' in obj.__name__.lower()):
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_model` will create a Proxy if
        # the name is not yet present.
        proxy = get_model()

        # Architecture operations require dependencies to be available.
        # Keep backward-compatible behavior by enabling dependency computation
        # unless the caller explicitly disables it.
        kwargs.setdefault('compute_dependencies', True)

        # Now construct the wrapper and let it register into the ledger.
        wrapper = ModelInterface(obj, **kwargs)

        # Register logger in backend for model training
        LoggerQueue()

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy != None else wrapper

    # DataLoader
    elif 'data' in flag.lower() or flag.lower() == 'dataset' or flag.lower() == 'dataloader' or (hasattr(obj, '__name__') and 'data' in obj.__name__.lower()):
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_dataloader` will create a Proxy if
        # the name is not yet present.\[]
        try:
            proxy = get_dataloader(kwargs.get('loader_name', DEFAULT_NAME))
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
        proxy.__pl_saved_kwargs = kwargs  # Force pytorch lightning compatibility
        
        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy != None else wrapper

    # Optimizer
    elif 'optimizer' in flag.lower() or (hasattr(obj, '__name__') and 'opt' in obj.__name__.lower()):
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_optimizer` will create a Proxy if
        # the name is not yet present.
        try:
            proxy = get_optimizer()
        except Exception:
            proxy = None

        # Now construct the wrapper and let it register into the ledger.
        wrapper = OptimizerInterface(obj, **kwargs)

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy != None else wrapper

    # Logger
    elif 'logger' in flag.lower() or (hasattr(obj, '__name__') and 'log' in obj.__name__.lower()):
        # Ensure there's a proxy placeholder if callers already requested the logger
        try:
            proxy = get_logger()
        except Exception:
            proxy = None

        # Register the logger into the ledger. This will update any proxy in-place.
        register_logger(obj)

        # Return a stable handle (proxy) when available, otherwise the registered logger
        return proxy if proxy != None else obj

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
            # If obj is a string, treat as a file path and start watcher
            try:
                # Initialize CheckpointManager if we have a root dir (fallback to default root)
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
                    if chkpt_manager != None and not isinstance(chkpt_manager, ledgers.Proxy):
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
                        return get_hyperparams()
                    
                    elif isinstance(obj, dict):
                        register_hyperparams(obj)
                        
                        # Update log directory if root_log_dir provided in hyperparameters or defaults
                        new_log_dir = None
                        if defaults and 'root_log_dir' in defaults:
                            new_log_dir = defaults['root_log_dir']
                        elif 'root_log_dir' in kwargs:
                            new_log_dir = kwargs['root_log_dir']
                        if new_log_dir:
                            _update_log_directory(new_log_dir)

                        return get_hyperparams()
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

                            return get_hyperparams()
                        except Exception:
                            raise ValueError('Unsupported hyperparams object; provide dict or YAML path')

                return get_hyperparams()
            except Exception:
                # bubble up original error
                raise

        raise ValueError(f"Obj name {obj} should contains at least 'model', 'data' or 'optimizer'.")


# ##############################################################################################################
# USER FUNCTION EXPOSED TO SERVE SIGNALS, TAG SAMPLES, ETC. (can be called from training script to manually set)
# ##############################################################################################################

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


def keep_serving(timeout: int = None) -> None:
    """ Keep the main thread alive to allow background serving threads to run.
    """
    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.info("Timeout reached, stopping WeightsLab services.")
                break
    except KeyboardInterrupt:
        logger.info("Shutting down WeightsLab services.")

#  Signal Definition & Computation Helpers
def signal(name: str = None, subscribe_to: str = None, compute_every_n_steps: int = 1, **kwargs):
    """
    Decorator to register a custom signal function.
    
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
        
        # Register in global ledger for visibility by backend services
        try:
            register_signal(func, name=reg_name)
        except Exception as e:
            logger.warning(f"Failed to register signal '{reg_name}' in ledger: {e}")
            
        return func
    return decorator


def compute_signals(dataset_or_loader, origin: str = None, signals: list[str] = None):
    """
    Execute registered signals on a dataset and update the central DataFrame.

    Args:
        dataset_or_loader: The dataset or dataloader to process.
        origin: The split name (e.g. 'train', 'val'). Auto-detected if possible.
        signals: List of signal names to run. If None, runs all registered signals.
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
        else:
            logger.warning(f"Signal '{name}' not found in registry.")

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
                    # Pass the input data (usually the image)
                    val = sig_func(input_data)
                    # Prefix 'signals_' if not already present to group in UI
                    key = sig_name if sig_name.startswith("signals") else f"signals_{sig_name}"
                    row[key] = val
                except Exception as e:
                    pass # Fail silently for single signal failures
            
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
        
        df_update = pd.DataFrame(rows).set_index("sample_id")
        df_manager.upsert_df(df_update, force_flush=True)
        
        logger.info(f"Tagged {len(sample_ids)} samples with '{tag}' (mode={mode})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to tag samples: {e}", exc_info=True)
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
        
        df_update = pd.DataFrame(rows).set_index("sample_id")
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
    batch_ids: th.Tensor,
    preds_raw: th.Tensor = None,
    targets: th.Tensor = None,
    preds: th.Tensor = None,
    step: int | None = None,
    log: bool = True
):
    """
        Save data statistics to the tracked dataset.
        
        Args:
            signals (th.Tensor): The batch losses.
            preds_raw (th.Tensor, optional): The raw batch predictions. Defaults to None.
            targets (th.Tensor, optional): The batch targets. Defaults to None.
            batch_ids (th.Tensor, optional): The batch ids. Defaults to None.
            preds (th.Tensor, optional): The batch predictions. Defaults to None.
            step (int, optional): The current training step. Defaults to 0.
            log (bool, optional): Whether to log the signals. Defaults to True.
        
        Examples:
            >>> # In your training loop, after computing losses and predictions:
            >>> for batch in train_loader:
            ...     inputs, targets = batch
            ...     preds = model(inputs)
            ...     loss = loss_fn(preds, targets)
            ...     batch_ids = batch['sample_id']  # Assuming your dataloader provides sample IDs
            ...     wl.save_signals(
            ...         signals={'train_loss': loss},
            ...         batch_ids=batch_ids,
            ...         preds_raw=preds,  # Optionally save raw predictions. Not useful if already saved by loss wrapper.
            ...         targets=targets,  # Optionally save target predictions. Not useful if already saved by loss wrapper.
            ...         step=current_step,  # Optionally save step. If not provided, will attempt to infer from training loop context.
            ...         log=True  # Should we log the signals also or only save to sample metadata.
            ...     )
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
    batch_ids_np = batch_ids.detach().cpu().numpy().astype(int) if isinstance(batch_ids, th.Tensor) else np.asarray(batch_ids).astype(int)
    if preds is not None:
        pred_np = preds.detach().cpu().numpy() if isinstance(preds, th.Tensor) else np.asarray(preds)
        if np.issubdtype(pred_np.dtype, np.floating):
             pred_np = pred_np.astype(np.float32)
        else:
             pred_np = pred_np.astype(np.uint16)
    else:
        pred_np = None

    if preds_raw is not None:
        pred_raw_np = preds_raw.detach().cpu().numpy() if isinstance(preds_raw, th.Tensor) else np.asarray(preds_raw) 
    else: 
        pred_raw_np = None

    if targets is not None:
        target_np = targets.detach().cpu().numpy() if isinstance(targets, th.Tensor) else np.asarray(targets)
        if np.issubdtype(target_np.dtype, np.floating):
             target_np = target_np.astype(np.float32)
        else:
             target_np = target_np.astype(np.uint16)
    else: 
        target_np = None

    # Processing
    # # Process signals
    if isinstance(signals, dict):
        losses_data = {\
            # Convert losses map of shape (B, ...) to (B,) by averaging all axes except batch (axis 0)
            'signals//' + k: (lambda arr: arr.mean(axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else arr)(
                v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
            )
            for k, v in signals.items()
        }
    elif signals is not None and isinstance(signals, (th.Tensor, np.ndarray, list)):
        losses_data = {
            "signals//default": (lambda arr: arr.mean(axis=tuple(range(1, arr.ndim))) if arr.ndim > 1 else arr)(
                signals.detach().cpu().numpy() if hasattr(signals, 'detach') else np.asarray(signals)
            )
        }
    else:
        losses_data = None
    # # Process targets
    if target_np is not None and target_np.ndim == 1:
        target_np = target_np[:, np.newaxis]
    if pred_np is not None and pred_np.ndim == 1:
        pred_np = pred_np[:, np.newaxis]
    if pred_raw_np is not None and pred_raw_np.ndim == 1:
        pred_raw_np = pred_raw_np[:, np.newaxis]

    # Enqueue to dataframe manager buffer for efficientcy
    DATAFRAME_M.enqueue_batch(
        sample_ids=batch_ids_np,
        preds_raw=pred_raw_np,
        preds=pred_np,
        targets=target_np,
        losses=losses_data,
        step=step
    )


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
    print(f"Found {len(difficult_ids)} difficult samples")

    # Remove tag after review
    tag_samples([5], 'outlier', mode='remove')

    # Keep script alive
    keep_serving(timeout=60)
