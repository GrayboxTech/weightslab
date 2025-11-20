import os
import torch
import tempfile
import weightslab as wl

from torchvision import datasets, transforms

from weightslab.tests.torch_models import FashionCNN 
from weightslab.ledgers import register_optimizer


class OptimizerInterface:
    def __init__(self, optimizer_or_cls, params=None, name: str = None, register: bool = True, weak: bool = False, _kill: bool = False, **kwargs):
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

        # Optionally register this wrapper into the global ledger so other
        # threads / modules can access it by name. Prefer explicit `name`,
        # else infer from the optimizer class or fall back to '_optimizer'.
        self._ledger_name = name
        if register:
            reg_name = name or getattr(optimizer_or_cls, "__name__", None) or "_optimizer"
            try:
                register_optimizer(reg_name, self, weak=weak)
            except Exception:
                # avoid failing construction due to ledger issues
                pass

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

    def get_param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def __repr__(self):
        cls = self.optimizer.__class__
        return f"OptimizerInterface({cls.__module__}.{cls.__name__}, lrs={self.get_lr()})"


if __name__ == "__main__":
    print('Hello World')

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
