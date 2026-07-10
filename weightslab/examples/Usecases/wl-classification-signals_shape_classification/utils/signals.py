# =============================================================================
# Per-sample @wl.signal chain
# =============================================================================
# Per-step user code is just the watched loss. Everything here is reactive:
#   sig/entropy    from the logits when the watched loss fires
#   sig/loss_norm  batch-normalized loss (loss / mean(loss))
#   sig/hardness   loss * entropy
#   sig/loss_shape classifies each sample's loss trajectory (live signal, not an
#                  end-of-run function). Reads history, so it's throttled by
#                  shape_every (1 = every step; higher = cheaper, sparser).
import numpy as np
import torch

import weightslab as wl

from utils.criterions import classify_shape


def register_signals(loss_name, shape_every, min_step: int = 0):
    """Define and register the per-sample signal chain on the watched loss.

    Defining a ``@wl.signal`` registers it globally, so this must be called
    before ``wl.serve`` / ``wl.start_training``.

    Args:
        loss_name: name of the watched per-sample loss signal to subscribe to.
        shape_every: compute sig/loss_shape every N steps (throttle).
        min_step: minimum step to start computing sig/loss_shape.
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

    @wl.signal(name="sig/loss_shape", inputs=[loss_name], batched=True,
               compute_every_n_steps=shape_every, min_step=min_step)
    def loss_shape(b):
        hist = b.history(loss_name)   # {uid: [loss values in step order]}
        return np.array([classify_shape(hist[s]) for s in b.sample_ids], dtype=float)
