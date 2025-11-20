""" The Experiment class is the main class of the graybox package.
It is used to train and evaluate models. """

import time
import functools
import torch as th

from tqdm import trange
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Callable, List
from collections import namedtuple
from threading import Lock, RLock

from weightslab.components.checkpoint import CheckpointManager
from weightslab.data.data_samples_with_ops import \
    DataSampleTrackingWrapper
from weightslab.backend.model_interface import ModelInterface
from weightslab.backend.data_loader_interface import DataLoaderInterface
from weightslab.backend.optimizer_interface import OptimizerInterface
from weightslab.ledgers import get_model, get_dataloader, get_optimizer
from weightslab.utils.logs import print
from weightslab.components.global_monitoring import GuardContext


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
        if obj_name is None:
            try:
                obj.__name__ = type(obj).__name__
            except Exception:
                obj.__name__ = str(time.time())
            print(
                "Warning: Watching or editing anonymous object '" +
                f"{obj.__name__}'."
            )
            print(
                "Please add a 'name' attribute to the object."
            )
        else:
            obj.__name__ = obj_name

    # Related functions
    if flag.lower() == 'model' or 'model' in obj.__name__.lower():
        reg_name = kwargs.get('name') or getattr(obj, '__name__', None) or obj.__class__.__name__
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
    elif flag.lower() == 'data' or 'data' in obj.__name__.lower():
        reg_name = kwargs.get('name') or getattr(getattr(obj, 'dataset', obj), '__name__', None) or getattr(getattr(obj, 'dataset', obj), '__class__', type(getattr(obj, 'dataset', obj))).__name__
        # Ensure ledger has a placeholder (Proxy) for this name so callers
        # receive a stable handle that will be updated in-place when the
        # real wrapper is registered. `get_dataloader` will create a Proxy if
        # the name is not yet present.
        try:
            proxy = get_dataloader(reg_name)
        except Exception:
            proxy = None

        # Now construct the wrapper and let it register into the ledger.
        wrapper = DataLoaderInterface(obj, **kwargs)

        # Prefer returning the proxy (if one exists) so external callers hold
        # a stable reference that will see updates. If no proxy was
        # obtainable, return the wrapper itself.
        return proxy if proxy is not None else wrapper
    elif flag.lower() == 'optimizer' or 'opt' in obj.__name__.lower():
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
    else:
        raise ValueError("Obj name should contains at least 'model', 'data' or 'optimizer'.")
