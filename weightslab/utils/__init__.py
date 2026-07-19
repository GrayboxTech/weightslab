"""weightslab.utils package.

The convenience re-exports from `.tools` are resolved lazily (PEP 562): `.tools`
imports torch/numpy, so eager re-export here would pull the heavy stack into
every `weightslab.utils.<x>` import (e.g. the torch-free `weightslab.utils.logs`
used by the UI/CLI). Names still import normally on first access, e.g.
``from weightslab.utils import filter_kwargs_for_callable``.
"""
import importlib

_LAZY_EXPORTS = {
    name: ".tools"
    for name in (
        "filter_kwargs_for_callable",
        "safe_call_with_kwargs",
        "capture_rng_state",
        "restore_rng_state",
        "recursive_update",
    )
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):  # PEP 562
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
