from typing import Dict, Any, Optional
import torch as th

class Task:
    """
    A thin contract the trainer can call uniformly.
    One Task == one 'head' (or composite loss) over the shared backbone.
    """
    def __init__(
        self,
        name: str,
        model,
        criterion,
        metrics: Optional[Dict[str, Any]] = None,
        loss_weight: float = 1.0,
    ):
        self.name = name
        self.model = model
        self.criterion = criterion             # must output per-sample loss (reduction='none')
        self.metrics = metrics or {}
        self.loss_weight = float(loss_weight)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model.forward_head(self.name, x)

    def compute_loss(self, outputs: th.Tensor, labels: th.Tensor) -> th.Tensor:
        """
        Returns per-sample loss (shape [N]).
        If criterion returns [N, ...], reduce to [N] by mean over non-batch dims.
        """
        losses = self.criterion(outputs, labels)
        if losses.ndim > 1:
            losses = losses.view(losses.shape[0], -1).mean(dim=1)
        return losses

    @th.no_grad()
    def infer_pred(self, outputs: th.Tensor):
        """
        convert logits to discrete predictions for logging.
        - Binary: [N] -> {0,1}
        - Multi-class: argmax channel
        - Dense (reconstruction/segmentation): return raw tensor
        """
        if outputs.ndim == 1:
            return (outputs > 0).long()
        if outputs.ndim >= 2 and outputs.shape[1] > 1:
            return outputs.argmax(dim=1)
        return outputs
