"""Global Ledgers for sharing objects across threads.

Provide a simple, thread-safe registry for models, dataloaders and
optimizers so different threads can access and update the same objects by
name. The ledger supports returning placeholder proxies for objects that are
not yet registered; those proxies can be updated in-place when the real
object is registered later, which enables the "import placeholder then
update" workflow described by the user.
"""

from __future__ import annotations

import threading
import weakref
import logging
import os
import time
import yaml
from collections.abc import MutableMapping

from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# Default registry name for single-experiment workflows.
# TODO: For parallel experiments (future), implement dynamic naming or experiment_id param.
DEFAULT_NAME = "main"


class Proxy:
    """A small forwarding proxy that holds a mutable reference to an object.

    Attribute access is forwarded to the underlying object once set. Until
    then, attempting to access attributes raises AttributeError.
    """

    def __init__(self, obj: Any = None):
        self._obj = obj

    @staticmethod
    def is_proxy(obj: Any) -> bool:
        """Return True when obj is a Proxy instance."""
        return isinstance(obj, Proxy)

    @property
    def __class__(self):
        """Report the class of the wrapped object to make isinstance checks work.

        This allows isinstance(proxy, dict) to return True when proxy wraps a dict.
        """
        if self._obj is not None:
            return type(self._obj)
        return type(self)

    def set(self, obj: Any) -> None:
        if isinstance(obj, Proxy):
            obj = obj.get()
        self._obj = obj
        # invalidate any cached iterator when target changes
        if hasattr(self, '_iterator'):
            try:
                del self._iterator
            except Exception:
                pass

    def get(self, ref=None, default=None) -> Any:
        if ref is not None:
            return self._obj.get(ref, default)
        return self._obj if self._obj is not None else default

    def __getattr__(self, item):
        # Use object.__getattribute__ to avoid infinite recursion during unpickling
        try:
            obj = object.__getattribute__(self, '_obj')
        except AttributeError:
            raise AttributeError("Proxy target not set")

        if obj is None:
            raise AttributeError("Proxy target not set")
        try:
            return getattr(obj, item)
        except AttributeError:
            return None

    # Special method forwarding for common container/iterable operations.
    # CPython looks up special methods on the type, so we must implement
    # them here to allow `for x in proxy` and `len(proxy)` to work.
    def __iter__(self):
        # Return a small iterator wrapper that delegates to the underlying
        # object's iterator. We return a fresh wrapper each call so multiple
        # concurrent iterations can proceed independently.
        if self._obj is None:
            raise TypeError("Proxy target not set")
        underlying_iter = iter(self._obj)

        class _ProxyIterator:
            def __init__(self, it):
                self._it = it

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._it)

        return _ProxyIterator(underlying_iter)

    def __len__(self):
        if self._obj is None:
            raise TypeError("Proxy target not set")
        return len(self._obj)

    def __getitem__(self, idx):
        if self._obj is None:
            raise TypeError("Proxy target not set")
        return self._obj[idx]

    def __setitem__(self, key, value):
        """Support item assignment: proxy[key] = value"""
        if self._obj is None:
            raise TypeError("Proxy target not set")
        self._obj[key] = value

    def __delitem__(self, key):
        """Support item deletion: del proxy[key]"""
        if self._obj is None:
            raise TypeError("Proxy target not set")
        del self._obj[key]

    def __contains__(self, item):
        """Support 'in' operator: key in proxy"""
        if self._obj is None:
            return False
        return item in self._obj

    def keys(self):
        """Support dict.keys() method"""
        if self._obj is None:
            raise TypeError("Proxy target not set")
        return self._obj.keys() if hasattr(self._obj, 'keys') else []

    def values(self):
        """Support dict.values() method"""
        if self._obj is None:
            raise TypeError("Proxy target not set")
        return self._obj.values() if hasattr(self._obj, 'values') else []

    def items(self):
        """Support dict.items() method"""
        if self._obj is None:
            raise TypeError("Proxy target not set")
        return self._obj.items() if hasattr(self._obj, 'items') else []

    def __call__(self, *args, **kwargs):
        """Forward callable invocation to the wrapped object.

        This allows code that receives a ledger Proxy for a callable
        (e.g., a model or function) to call it directly: `proxy(x)`.
        """
        if self._obj is None:
            raise TypeError("Proxy target not set")
        target = self._obj

        # Perform call outside lock to avoid deadlocks if target itself
        # acquires locks and calls back into ledger.
        return target(*args, **kwargs)

    def __repr__(self):
        return f"Proxy({repr(self._obj)})"

    def __eq__(self, other):
        """Enable equality comparison with the wrapped object.

        This allows `proxy is None` to return True when the wrapped object is None.
        
        IMPORTANT: Use `proxy is None` or `proxy is not None` for None checks.
        Python's `is` operator cannot be overridden and `proxy is None` will always be False.
        """
        if other is None:
            return self._obj is None
        if isinstance(other, Proxy):
            return self._obj is other._obj
        return self._obj == other

    def __ne__(self, other):
        """Enable inequality comparison with the wrapped object.
        
        This allows `proxy is not None` to return False when the wrapped object is None.
        """
        return not self.__eq__(other)

    def __bool__(self):
        """Enable boolean evaluation of the proxy based on the wrapped object.

        This allows `bool(Proxy(None))` to return False and
        `if not proxy:` to work correctly when proxy wraps None.
        """
        if self._obj is None:
            return False
        return bool(self._obj)

    def __hash__(self):
        """Make Proxy hashable by hashing the wrapped object's identity.
        
        This allows Proxy objects to be used in sets and as dictionary keys.
        Uses id() for consistent hashing regardless of object's __hash__ implementation.
        """
        if self._obj is None:
            return hash(None)
        # Use id() to ensure consistent hashing even for unhashable wrapped objects
        return hash(id(self._obj))

    def __next__(self):
        """Allow the Proxy itself to act as an iterator when `next(proxy)` is
        called. We cache an internal iterator per-proxy so successive calls to
        `next(proxy)` advance through the wrapped object. The iterator is
        invalidated when `set()` is called.
        """
        try:
            return next(self._obj)
        except Exception:
            # clear cached iterator so future next(proxy) restarts
            try:
                delattr(self, '_iterator')
            except Exception:
                pass
        raise StopIteration

    # Context manager support so `with proxy as x:` works when the proxy
    # wraps an object that implements the context manager protocol. If the
    # wrapped object does not implement __enter__/__exit__, the proxy will
    # simply return the wrapped object from __enter__ and do nothing on exit.
    def __enter__(self):
        if self._obj is None:
            raise TypeError("Proxy target not set")
        target = self._obj

        enter = getattr(target, '__enter__', None)
        if callable(enter):
            # call __enter__ on the underlying object
            return enter()
        # fallback: return the underlying object itself
        return target

    def __exit__(self, exc_type, exc, tb):
        if self._obj is None:
            raise TypeError("Proxy target not set")
        target = self._obj

        exit_fn = getattr(target, '__exit__', None)
        if callable(exit_fn):
            return exit_fn(exc_type, exc, tb)
        # if underlying object had no __exit__, just return False
        return False


# Register Proxy as a virtual subclass of MutableMapping so isinstance(proxy, dict) works
# Note: This makes Proxy compatible with dict-like checks but doesn't inherit from dict
MutableMapping.register(Proxy)


class Ledger:
    """Thread-safe ledger storing named registries for different object types.

    The ledger stores strong references by default and also supports weak
    registrations. If an object is requested via `get_*` and not present a
    `Proxy` placeholder is created, stored, and returned; calling
    `register_*` with the same name will update the proxy in-place.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # strong refs
        self._models: Dict[str, Any] = {}
        self._dataloaders: Dict[str, Any] = {}
        self._optimizers: Dict[str, Any] = {}
        self._dataframes: Dict[str, Any] = {}
        self._checkpoint_managers: Dict[str, Any] = {}
        # weak refs
        self._models_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        self._dataloaders_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        self._optimizers_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        self._dataframes_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        self._checkpoint_managers_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        self._hyperparams_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        self._loggers_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        self._signals_weak: "weakref.WeakValueDictionary[str, Any]" = weakref.WeakValueDictionary()
        # proxies mapping name -> Proxy for placeholders
        self._proxies_models: Dict[str, Proxy] = {}
        self._proxies_dataloaders: Dict[str, Proxy] = {}
        self._proxies_optimizers: Dict[str, Proxy] = {}
        self._proxies_dataframes: Dict[str, Proxy] = {}
        self._proxies_checkpoint_managers: Dict[str, Proxy] = {}
        # hyperparameters registry (name -> dict)
        self._hyperparams: Dict[str, Dict[str, Any]] = {}
        self._proxies_hyperparams: Dict[str, Proxy] = {}
        # hyperparam file watchers: name -> dict(path, thread, stop_event)
        self._hp_watchers: Dict[str, Dict[str, Any]] = {}
        # loggers registry
        self._loggers: Dict[str, Any] = {}
        self._proxies_loggers: Dict[str, Proxy] = {}
        # signals registry (metrics, losses, etc.)
        self._signals: Dict[str, Any] = {}
        self._proxies_signals: Dict[str, Proxy] = {}

    # Generic helpers
    def _register(self, registry: Dict[str, Any], registry_weak: weakref.WeakValueDictionary, proxies: Dict[str, Proxy], name: str, obj: Any, weak: bool = False) -> None:
        with self._lock:
            if weak:
                registry.pop(name, None)
                registry_weak[name] = obj
                return registry_weak[name]
            else:
                proxy = proxies.get(name)
                if proxy is not None:
                    # update proxy in-place and keep the proxy as the public handle
                    proxy.set(obj)
                    registry[name] = proxy
                else:
                    registry[name] = obj if obj is not None else Proxy(None)
                if name in registry_weak:
                    try:
                        del registry_weak[name]
                    except KeyError:
                        pass
                return registry[name]

    def _get(self, registry: Dict[str, Any], registry_weak: weakref.WeakValueDictionary, proxies: Dict[str, Proxy], name: Optional[str] = None) -> Any:
        with self._lock:
            if name is not None:
                if name in registry:
                    return registry[name]
                if name in registry_weak:
                    return registry_weak[name]
                # create a placeholder proxy, store it strongly and return it
                proxy = Proxy(None)
                registry[name] = proxy
                proxies[name] = proxy
                return proxy

            # if name is None and exactly one total item exists, return it
            keys = set(registry.keys()) | set(registry_weak.keys())
            if len(keys) == 1:
                k = next(iter(keys))
                return registry.get(k, registry_weak.get(k))
            raise KeyError("multiple entries present, specify a name")

    def _list(self, registry: Dict[str, Any], registry_weak: weakref.WeakValueDictionary) -> List[str]:
        with self._lock:
            # combine keys from strong and weak registries
            keys = list(dict.fromkeys(list(registry.keys()) + list(registry_weak.keys())))
            return keys

    def _unregister(self, registry: Dict[str, Any], registry_weak: weakref.WeakValueDictionary, proxies: Dict[str, Proxy], name: str) -> None:
        with self._lock:
            registry.pop(name, None)
            try:
                del registry_weak[name]
            except KeyError:
                pass
            proxies.pop(name, None)

    # ===========================
    # Hyperparameters
    # ===========================
    def register_hyperparams(self, params: Dict[str, Any] = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
        """Register a dict of hyperparameters under `name`. Overwrites any
        existing entry. If a Proxy placeholder exists for this name it is
        updated in-place (so external references continue to work).
        """
        with self._lock:
            proxy = self._proxies_hyperparams.get(name)
            if proxy is not None:
                proxy.set(params)
                self._hyperparams[name] = proxy
            else:
                self._hyperparams[name] = params
        return self._register(self._hyperparams, self._hyperparams_weak, self._proxies_hyperparams, name, params, weak=weak)

    def get_hyperparams(self, name: str = DEFAULT_NAME) -> Any:
        """Get hyperparams by name. Creates a placeholder Proxy(None) if not yet registered."""
        with self._lock:
            if name in self._hyperparams:
                return self._hyperparams[name]
            # create placeholder proxy
            proxy = Proxy({})
            self._hyperparams[name] = proxy
            self._proxies_hyperparams[name] = proxy
            return proxy

    def list_hyperparams(self) -> List[str]:
        with self._lock:
            return list(self._hyperparams.keys())

    def set_hyperparam(self, key_path: str, value: Any, name: str = DEFAULT_NAME) -> None:
        """Set a nested hyperparameter using dot-separated `key_path`.
        Example: set_hyperparam('exp', 'data.train.batch_size', 128)
        """
        with self._lock:
            if name is None:
                from weightslab.backend.ledgers import resolve_hp_name
                name = resolve_hp_name()
            if name is None or name not in self._hyperparams:
                # If still None and we have sets, take the first as ultimate fallback
                keys = list(self._hyperparams.keys())
                if not name and keys:
                    name = keys[0]

            hp = self._hyperparams.get(name, Proxy({}))
            parts = key_path.split('.') if key_path else []
            cur = hp
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value

    def watch_hyperparams_file(self, path: str, poll_interval: float = 1.0, name: str = DEFAULT_NAME) -> None:
        """Start (or restart) a background watcher that loads the YAML at
        `path` into the hyperparams registry under `name`. The file is polled
        every `poll_interval` seconds. If a watcher already exists for `name`
        it will be stopped and replaced.
        """
        with self._lock:
            # stop existing watcher if present
            existing = self._hp_watchers.get(name)
            if existing is not None:
                try:
                    existing['stop_event'].set()
                    existing['thread'].join(timeout=1.0)
                except Exception:
                    pass

            stop_event = threading.Event()

            def _watcher():
                last_mtime = None
                # initial load if present
                while not stop_event.is_set():
                    try:
                        if os.path.exists(path):
                            mtime = os.path.getmtime(path)
                            if last_mtime is None or mtime != last_mtime:
                                with open(path, 'r', encoding='utf-8') as f:
                                    data = yaml.safe_load(f)
                                if data is None:
                                    data = {}
                                if not isinstance(data, dict):
                                    # ignore invalid top-level content
                                    last_mtime = mtime
                                else:
                                    self.register_hyperparams(name, data)
                                    last_mtime = mtime
                        # sleep with small increments to be responsive to stop_event
                        for _ in range(int(max(1, poll_interval * 10))):
                            if stop_event.is_set():
                                break
                            time.sleep(poll_interval / 10.0)
                    except Exception:
                        # swallow errors to keep watcher alive; user can inspect file
                        time.sleep(poll_interval)

            th = threading.Thread(target=_watcher, name=f"WL-HP_Watcher_{name}", daemon=True)
            self._hp_watchers[name] = {'path': path, 'thread': th, 'stop_event': stop_event}
            th.start()

    def unwatch_hyperparams_file(self, name: str = DEFAULT_NAME) -> None:
        """Stop a running hyperparams file watcher for `name` if present."""
        with self._lock:
            existing = self._hp_watchers.pop(name, None)
            if existing is None:
                return
            try:
                existing['stop_event'].set()
                existing['thread'].join(timeout=1.0)
            except Exception:
                pass

    # ===========================
    # Loggers
    # ===========================
    def register_logger(self, logger: Any = None, name: str = DEFAULT_NAME, weak: bool = False) -> None:
        return self._register(self._loggers, self._loggers_weak, self._proxies_loggers, name, logger, weak=weak)

    def get_logger(self, name: str = DEFAULT_NAME) -> Any:
        with self._lock:
            if name in self._loggers:
                return self._loggers[name]
            proxy = Proxy(None)
            self._loggers[name] = proxy
            self._proxies_loggers[name] = proxy
            return proxy

    def list_loggers(self) -> List[str]:
        with self._lock:
            return list(self._loggers.keys())

    def unregister_logger(self, name: str = DEFAULT_NAME) -> None:
        with self._lock:
            self._loggers.pop(name, None)
            self._proxies_loggers.pop(name, None)

    # ===========================
    # Signals (metrics, loss functions, monitors)
    # ===========================
    def register_signal(self, signal: Any = None, name: str = DEFAULT_NAME, weak: bool = False) -> None:
        return self._register(self._signals, self._signals_weak, self._proxies_signals, name, signal, weak=weak)

    def get_signal(self, name: str = DEFAULT_NAME) -> Any:
        with self._lock:
            if name in self._signals:
                return self._signals[name]
            proxy = Proxy(None)
            self._signals[name] = proxy
            self._proxies_signals[name] = proxy
            return proxy

    def list_signals(self) -> List[str]:
        with self._lock:
            return list(self._signals.keys())

    def unregister_signal(self, name: str = DEFAULT_NAME) -> None:
        with self._lock:
            self._signals.pop(name, None)
            self._proxies_signals.pop(name, None)

    # ===========================
    # Checkpoint managers
    # ===========================
    def register_checkpoint_manager(self, manager: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> Any:
        return self._register(self._checkpoint_managers, self._checkpoint_managers_weak, self._proxies_checkpoint_managers, name, manager, weak=weak)

    def get_checkpoint_manager(self, name: str = DEFAULT_NAME) -> Any:
        return self._get(self._checkpoint_managers, self._checkpoint_managers_weak, self._proxies_checkpoint_managers, name)

    def list_checkpoint_managers(self) -> List[str]:
        return self._list(self._checkpoint_managers, self._checkpoint_managers_weak)

    def unregister_checkpoint_manager(self, name: str = DEFAULT_NAME) -> None:
        self._unregister(self._checkpoint_managers, self._checkpoint_managers_weak, self._proxies_checkpoint_managers, name)

    # ===========================
    # DataFrames (e.g., shared sample stats managers)
    # ===========================
    def register_dataframe(self, dataframe: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
        return self._register(self._dataframes, self._dataframes_weak, self._proxies_dataframes, name, dataframe, weak=weak)

    def get_dataframe(self, name: str = DEFAULT_NAME) -> Any:
        return self._get(self._dataframes, self._dataframes_weak, self._proxies_dataframes, name)

    def list_dataframes(self) -> List[str]:
        return self._list(self._dataframes, self._dataframes_weak)

    def unregister_dataframe(self, name: str = DEFAULT_NAME) -> None:
        self._unregister(self._dataframes, self._dataframes_weak, self._proxies_dataframes, name)

    # ===========================
    # Models
    # ===========================
    def register_model(self, model: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
        self._register(self._models, self._models_weak, self._proxies_models, name, model, weak=weak)

    def get_model(self, name: str = DEFAULT_NAME) -> Any:
        return self._get(self._models, self._models_weak, self._proxies_models, name)

    def list_models(self) -> List[str]:
        return self._list(self._models, self._models_weak)

    def unregister_model(self, name: str = DEFAULT_NAME) -> None:
        self._unregister(self._models, self._models_weak, self._proxies_models, name)

    # ===========================
    # Dataloaders
    # ===========================
    def register_dataloader(self, dataloader: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
        self._register(self._dataloaders, self._dataloaders_weak, self._proxies_dataloaders, name, dataloader, weak=weak)

    def get_dataloader(self, name: str = DEFAULT_NAME) -> Any:
        if name is None:
            return Proxy({})
        return self._get(self._dataloaders, self._dataloaders_weak, self._proxies_dataloaders, name)

    def list_dataloaders(self) -> List[str]:
        return self._list(self._dataloaders, self._dataloaders_weak)

    def unregister_dataloader(self, name: str = DEFAULT_NAME) -> None:
        self._unregister(self._dataloaders, self._dataloaders_weak, self._proxies_dataloaders, name)

    def register_dataloaders_dict(self, dataloaders: Dict[str, Any], weak: bool = False) -> None:
        """Register multiple dataloaders from a dict, e.g., {'train': train_loader, 'val': val_loader}."""
        for name, loader in dataloaders.items():
            self.register_dataloader(loader, weak=weak, name=name)

    def get_dataloaders_dict(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get multiple dataloaders as a dict. If names is None, returns common default names.

        Args:
            names: List of dataloader names to retrieve. If None, uses ['train', 'val', 'test'].

        Returns:
            Dict mapping each name to its Proxy (creates Proxy(None) if not yet registered).
        """
        if names is None:
            names = self.list_dataloaders()

        result = {}
        for name in names:
            result[name] = self.get_dataloader(name)
        return result

    # ===========================
    # Optimizers
    # ===========================
    def register_optimizer(self, optimizer: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
        self._register(self._optimizers, self._optimizers_weak, self._proxies_optimizers, name, optimizer, weak=weak)

    def get_optimizer(self, name: str = DEFAULT_NAME) -> Any:
        return self._get(self._optimizers, self._optimizers_weak, self._proxies_optimizers, name)

    def list_optimizers(self) -> List[str]:
        return self._list(self._optimizers, self._optimizers_weak)

    def unregister_optimizer(self, name: str = DEFAULT_NAME) -> None:
        self._unregister(self._optimizers, self._optimizers_weak, self._proxies_optimizers, name)

    # ===========================
    # Convenience
    # ===========================
    def clear(self) -> None:
        """Clear all registries."""
        with self._lock:
            self._models.clear()
            self._dataloaders.clear()
            self._optimizers.clear()
            self._dataframes.clear()
            self._checkpoint_managers.clear()
            self._hyperparams.clear()
            self._loggers.clear()
            self._signals.clear()
            self._models_weak.clear()
            self._dataloaders_weak.clear()
            self._optimizers_weak.clear()
            self._dataframes_weak.clear()
            self._checkpoint_managers_weak.clear()
            self._hyperparams_weak.clear()
            self._loggers_weak.clear()
            self._signals_weak.clear()
            self._proxies_models.clear()
            self._proxies_dataloaders.clear()
            self._proxies_optimizers.clear()
            self._proxies_dataframes.clear()
            self._proxies_checkpoint_managers.clear()
            self._proxies_hyperparams.clear()
            self._proxies_loggers.clear()
            self._proxies_signals.clear()

    def snapshot(self) -> Dict[str, List[str]]:
        """Return the current keys for all registries (a lightweight snapshot)."""
        with self._lock:
            return {
                "models": list(self._models.keys()),
                "dataloaders": list(self._dataloaders.keys()),
                "optimizers": list(self._optimizers.keys()),
                "dataframes": list(self._dataframes.keys()),
                "hyperparams": list(self._hyperparams.keys()),
                "loggers": list(self._loggers.keys()),
                "checkpoint_managers": list(self._checkpoint_managers.keys()),
            }

    def __repr__(self) -> str:
        s = self.snapshot()
        return str(s)


# Module-level singleton
GLOBAL_LEDGER = Ledger()


# Model
def list_models() -> List[str]:
    return GLOBAL_LEDGER.list_models()

def register_model(model: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
    name = DEFAULT_NAME if name is None else name
    GLOBAL_LEDGER.register_model(model, weak=weak, name=name)

def get_model(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_model(name)

def get_models() -> List[str]:
    return GLOBAL_LEDGER.list_models()


# Dataloaders
def list_dataloaders() -> List[str]:
    return GLOBAL_LEDGER.list_dataloaders()

def register_dataloader(dataloader: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
    name = DEFAULT_NAME if name is None else name
    GLOBAL_LEDGER.register_dataloader(dataloader, weak=weak, name=name)

def get_dataloader(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_dataloader(name)

def register_dataloaders(dataloaders: Dict[str, Any], weak: bool = False) -> None:
    """Register multiple dataloaders from a dict, e.g., {'train': train_loader, 'val': val_loader}."""
    GLOBAL_LEDGER.register_dataloaders_dict(dataloaders, weak=weak)

def get_dataloaders(names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get multiple dataloaders as a dict. If names is None, uses ['train', 'val', 'test'].

    Returns dict mapping each name to its Proxy (creates Proxy(None) if not yet registered).
    """
    return GLOBAL_LEDGER.get_dataloaders_dict(names)


# Optimizer
def list_optimizers() -> List[str]:
    return GLOBAL_LEDGER.list_optimizers()

def register_optimizer(optimizer: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
    name = DEFAULT_NAME if name is None else name
    GLOBAL_LEDGER.register_optimizer(optimizer, weak=weak, name=name)

def get_optimizer(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_optimizer(name)

def get_optimizers() -> List[str]:
    return GLOBAL_LEDGER.list_optimizers()


# Hyperparameters
def register_hyperparams(params: Dict[str, Any] = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
    name = DEFAULT_NAME if name is None else name
    GLOBAL_LEDGER.register_hyperparams(params, weak=weak, name=name)

def get_hyperparams(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_hyperparams(name)

def list_hyperparams() -> List[str]:
    return GLOBAL_LEDGER.list_hyperparams()

def resolve_hp_name() -> str | None:
    """Resolve a sensible hyperparam set name from the ledger.
    Checks for 'main', then 'experiment', then falls back to the first registered name.
    """
    names = list_hyperparams()
    if not names:
        return 'unknown'
    if 'main' in names:
        return 'main'
    if 'experiment' in names:
        return 'experiment'
    # If we have any names at all, returning the first one is better than returning None
    # and causing a "Cannot resolve hyperparams name" error in the UI.
    return names[-1]  # first is empty proxy parameters generated at init

def set_hyperparam(key_path: str, value: Any, name: str = DEFAULT_NAME) -> None:
    try:
        return GLOBAL_LEDGER.set_hyperparam(key_path, value, name=name)
    except IndexError:
        logger.error(f'no hyperparams registered under {name}')

def watch_hyperparams_file(path: str, poll_interval: float = 1.0, name: str = DEFAULT_NAME) -> None:
    return GLOBAL_LEDGER.watch_hyperparams_file(name, path, poll_interval=poll_interval)

def unwatch_hyperparams_file(name: str = DEFAULT_NAME) -> None:
    return GLOBAL_LEDGER.unwatch_hyperparams_file(name)


# Logger
def register_logger(logger: Any = None, name: str = DEFAULT_NAME) -> None:
    name = DEFAULT_NAME if name is None else name
    GLOBAL_LEDGER.register_logger(logger, name=name)

def get_logger(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_logger(name)

def list_loggers() -> List[str]:
    return GLOBAL_LEDGER.list_loggers()

def unregister_logger(name: str = DEFAULT_NAME) -> None:
    return GLOBAL_LEDGER.unregister_logger(name)


# Signals
def register_signal(signal: Any = None, name: str = DEFAULT_NAME) -> None:
    name = DEFAULT_NAME if name is None else name
    GLOBAL_LEDGER.register_signal(signal, name=name)

def get_signal(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_signal(name)

def list_signals() -> List[str]:
    return GLOBAL_LEDGER.list_signals()

def unregister_signal(name: str = DEFAULT_NAME) -> None:
    return GLOBAL_LEDGER.unregister_signal(name)


# Checkpoint managers
def register_checkpoint_manager(manager: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> Any:
    name = DEFAULT_NAME if name is None else name
    return GLOBAL_LEDGER.register_checkpoint_manager(manager, weak=weak, name=name)

def get_checkpoint_manager(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_checkpoint_manager(name)

def list_checkpoint_managers() -> List[str]:
    return GLOBAL_LEDGER.list_checkpoint_managers()

def unregister_checkpoint_manager(name: str = DEFAULT_NAME) -> None:
    GLOBAL_LEDGER.unregister_checkpoint_manager(name)


# DataFrames
def register_dataframe(dataframe: Any = None, weak: bool = False, name: str = DEFAULT_NAME) -> None:
    name = DEFAULT_NAME if name is None else name
    return GLOBAL_LEDGER.register_dataframe(dataframe, weak=weak, name=name)

def get_dataframe(name: str = DEFAULT_NAME) -> Any:
    return GLOBAL_LEDGER.get_dataframe(name)

def list_dataframes() -> List[str]:
    return GLOBAL_LEDGER.list_dataframes()

def unregister_dataframe(name: str = DEFAULT_NAME) -> None:
    return GLOBAL_LEDGER.unregister_dataframe(name)


# Convenience
def clear_all() -> None:
    """Clear all registries (models, dataloaders, optimizers, dataframes, hyperparams, loggers, signals, etc.)"""
    return GLOBAL_LEDGER.clear()
