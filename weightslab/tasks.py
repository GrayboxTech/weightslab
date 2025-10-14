# tasks.py
from typing import Dict, Any, Optional, Callable
import torch as th

TargetFn = Callable[[th.Tensor], th.Tensor]  # input -> targets

class Task:
    """
    One Task == one 'head' (and its loss + metrics) over a shared backbone.
    """
    def __init__(
        self,
        name: str,
        model,
        criterion,
        metrics: Optional[Dict[str, Any]] = None,
        loss_weight: float = 1.0,
        target_fn: Optional[TargetFn] = None,  # NEW
    ):
        self.name = name
        self.model = model
        self.criterion = criterion             # must return per-sample loss (reduction='none')
        self.metrics = metrics or {}
        self.loss_weight = float(loss_weight)
        # by default use classification labels that your dataset wrapper attaches
        self.target_fn = target_fn or (lambda inp: inp.label_batch)

    def get_targets(self, inp: th.Tensor) -> th.Tensor:
        return self.target_fn(inp)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # model must implement: forward_head(task_name, x)
        return self.model.forward_head(self.name, x)

    def compute_loss(self, outputs: th.Tensor, labels: th.Tensor) -> th.Tensor:
        """
        Returns per-sample loss [N]. If criterion returns [N,...] we reduce
        over non-batch dims (mean) to [N].
        """
        losses = self.criterion(outputs, labels)
        if losses.ndim > 1:
            losses = losses.view(losses.shape[0], -1).mean(dim=1)
        return losses

    @th.no_grad()
    def infer_pred(self, outputs: th.Tensor):
        """
        Convert logits to discrete predictions for logging/inspection.
        """
        if outputs.ndim == 1:
            return (outputs > 0).long()
        if outputs.ndim >= 2 and outputs.shape[1] > 1:
            return outputs.argmax(dim=1)
        # dense outputs (recon/seg): return raw tensor
        return outputs


'''typical construction'''

# from tasks import Task
# import torch.nn as nn

# cls_task = Task(
#     name="class",
#     model=unet,  # exposes forward_head("class", x)
#     criterion=nn.CrossEntropyLoss(reduction="none"),
#     loss_weight=1.0,
#     target_fn=lambda inp: inp.label_batch,   # default anyway
# )

# recon_task = Task(
#     name="recon",
#     model=unet,  # exposes forward_head("recon", x)
#     criterion=nn.MSELoss(reduction="none"),
#     loss_weight=0.3,
#     target_fn=lambda inp: inp,               # targets are the input (reconstruction)
# )
