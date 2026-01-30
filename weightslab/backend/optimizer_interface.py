import torch
import tempfile
import weightslab as wl

from weightslab.backend.ledgers import register_optimizer


class OptimizerInterface:
    def __init__(self, optimizer_or_cls, params=None, register: bool = True, weak: bool = False, **kwargs):
        """Wrap a torch optimizer instance or instantiate one from a class.

        If `optimizer_or_cls` is an instance of `torch.optim.Optimizer` it is
        used directly. Otherwise it is expected to be a callable (optimizer
        class/factory) and `params` must be provided.
        """

        if isinstance(optimizer_or_cls, torch.optim.Optimizer):
            self.optimizer = optimizer_or_cls
            self._constructed = False
        else:
            if params is None:
                raise ValueError("When passing an optimizer class you must provide 'params'.")
            self.optimizer = optimizer_or_cls(params, **kwargs)
            self._constructed = True

        # Expose attributes and methods from the wrapped optimizer
        self.init_attributes(self.optimizer)

        # Optionally register this wrapper into the global ledger so other
        # threads / modules can access it by name. Prefer explicit `name`,
        # else infer from the optimizer class or fall back to '_optimizer'.
        if register:
            register_optimizer(self, weak=weak)

    def init_attributes(self, obj):
        """Expose attributes and methods from the wrapped `obj`.

        Implementation strategy (direct iteration):
        - Iterate over `vars(obj)` to obtain instance attributes and
          create class-level properties that forward to `obj.<attr>`.
        - Iterate over `vars(obj.__class__)` to find callables (methods)
          and bind the model's bound method to this wrapper instance so
          calling `mi.method()` invokes `mi.model.method()`.

        This avoids using `dir()` and directly inspects the object's
        own dictionaries. Existing attributes on `ModelInterface` are
        preserved and not overwritten.
        """
        # Existing names on the wrapper instance/class to avoid overwriting
        existing_instance_names = set(self.__dict__.keys())
        existing_class_names = set(getattr(self.__class__, '__dict__', {}).keys())

        # 1) Expose model instance attributes as properties on the wrapper class
        optimizer_vars = getattr(obj, '__dict__', {})
        for name, _ in optimizer_vars.items():
            if name.startswith('_'):
                continue
            if name in existing_instance_names or name in existing_class_names:
                continue

            # Create a property on the ModelInterface class that forwards to
            # the underlying model attribute. Using a property keeps the
            # attribute live (reads reflect model changes).
            try:
                def _make_getter(n):
                    return lambda inst: getattr(inst.optimizer, n)

                getter = _make_getter(name)
                prop = property(fget=getter)
                setattr(self.__class__, name, prop)
            except Exception:
                # Best-effort: skip if we cannot set the property
                continue

        # 2) Bind model class-level callables (methods) to this instance
        optimizer_cls_vars = getattr(obj.__class__, '__dict__', {})
        for name, member in optimizer_cls_vars.items():
            if name.startswith('_'):
                continue
            if name in existing_instance_names or name in existing_class_names:
                continue

            # Only consider callables defined on the class (functions/descriptors)
            if callable(member):
                try:
                    # getattr(obj, name) returns the bound method
                    bound = getattr(obj, name)
                    # Attach the bound method to the wrapper instance so that
                    # calling mi.name(...) calls model.name(...)
                    setattr(self, name, bound)
                except Exception:
                    # If we cannot bind, skip gracefully
                    continue

    def step(self, closure=None):
        return self.optimizer.step(closure) if closure is not None else self.optimizer.step()

    def zero_grad(self, set_to_none=False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        return [g.get('lr') for g in self.optimizer.param_groups]

    def set_lr(self, lr, group_idx=None):
        if group_idx is None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr
        else:
            self.optimizer.param_groups[group_idx]['lr'] = lr

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def __repr__(self):
        cls = self.optimizer.__class__
        return f"OptimizerInterface({cls.__module__}.{cls.__name__}, lrs={self.get_lr()})"


if __name__ == "__main__":
    print('Hello World')
    from weightslab.baseline_models.pytorch.models import FashionCNN

    # 0. Init. variables
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TMP_DIR = tempfile.mkdtemp()

    # 1. Define the model
    model = FashionCNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2
    )

    # WatchOrEdit
    model = wl.watch_or_edit(model, flag='model')
    optim = wl.watch_or_edit(optimizer, flag='optimizer')
