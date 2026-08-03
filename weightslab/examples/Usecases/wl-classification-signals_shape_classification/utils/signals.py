# =============================================================================
# Per-sample @wl.signal chain
# =============================================================================
# Per-step user code is just the watched loss. Everything here is reactive:
#   sig/entropy    from the logits when the watched loss fires
#   sig/loss_norm  batch-normalized loss (loss / mean(loss))
#   sig/hardness   loss * entropy
#
# The loss-SHAPE part is no longer a live signal here: it's a custom
# ``@wl.signal_classifier`` (see utils/criterions.py) that labels each sample's
# loss trajectory monotonic / not_monotonic and lands in a categorical
# ``tag:loss_shape`` column via the background auto-tagger.
import numpy as np
import torch

import weightslab as wl


def register_signals(loss_name):
    """Define and register the per-sample signal chain on the watched loss.

    Defining a ``@wl.signal`` registers it globally, so this must be called
    before ``wl.serve`` / ``wl.start_training``.

    Args:
        loss_name: name of the watched per-sample loss signal to subscribe to.
    """

    @wl.signal(name="sig/entropy", subscribe_to=loss_name, batched=True)
    def entropy(b):
        p = torch.softmax(b.logits, 1)
        return (-(p * (p + 1e-12).log()).sum(1)).detach().cpu().numpy()

    @wl.signal(name="sig/loss_norm", inputs=[loss_name], batched=True)
    def loss_norm(b):
        return b.inputs[loss_name] / (float(np.mean(b.inputs[loss_name])) + 1e-8)

    @wl.signal(name="sig/hardness", inputs=[loss_name, "sig/entropy"], batched=True)
    def hardness(b):
        return b.inputs[loss_name] * b.inputs["sig/entropy"]
