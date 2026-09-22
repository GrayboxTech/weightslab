"""Parametric UMAP: a live 3-D projection of what the model is learning.

A small encoder network ``features -> R^3`` is trained *alongside* your model,
on the UMAP cross-entropy, from the activations of one layer. Unlike
``umap-learn``'s ``fit_transform`` this is **parametric**: the projection is a
function, so a sample's coordinates come from one forward pass and the
embedding stays comparable from step to step instead of being re-laid-out from
scratch on every refit.

Why parametric, here:
  * it runs online during training, on batches the model already computed, so
    there is no separate embedding pass over the dataset;
  * new/unseen samples project without refitting;
  * the embedding moves smoothly as the representation moves, which is the
    thing worth watching -- a fresh UMAP per epoch mostly shows you UMAP's own
    initialisation noise.

It NEVER perturbs training: features are detached at the hook and the encoder
carries its own optimizer. Turning it off (see :func:`projection_enabled`)
removes the hook entirely.

Algorithm (standard UMAP objective, computed densely per batch):
  1. kNN graph on the batch's features, with UMAP's per-point ``rho``
     (distance to nearest neighbour) and ``sigma`` (binary-searched so the
     row's memberships sum to ``log2(k)``).
  2. High-dim membership ``mu_ij = exp(-(d_ij - rho_i) / sigma_i)``,
     symmetrised as ``mu + mu^T - mu*mu^T``.
  3. Low-dim membership ``nu_ij = 1 / (1 + a * d_ij^(2b))``, with ``(a, b)``
     fitted from ``spread``/``min_dist`` as umap-learn does.
  4. Cross-entropy between the two: attractive ``-mu*log(nu)`` on the kNN
     edges, repulsive ``-(1-mu)*log(1-nu)`` on the rest.

Batch-level density (B up to a few hundred) keeps this a handful of B x B
matmuls -- cheap next to the model's own step, and it only runs every
``every_n_steps`` steps.
"""

from __future__ import annotations

import logging
import os
import re
import threading

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Env switches. The feature is ON by default; set WEIGHTSLAB_PROJECTION to one
# of 0/false/no/off to remove it completely (no hook, no encoder, no cost).
ENV_ENABLED = "WEIGHTSLAB_PROJECTION"
ENV_EVERY = "WEIGHTSLAB_PROJECTION_EVERY"
ENV_DIM = "WEIGHTSLAB_PROJECTION_DIM"
ENV_NEIGHBORS = "WEIGHTSLAB_PROJECTION_NEIGHBORS"
ENV_GRAPH = "WEIGHTSLAB_PROJECTION_GRAPH"

_OFF_VALUES = {"0", "false", "no", "off"}

# How many fit-visits with no captured features before we say so out loud.
_MISS_WARN_AFTER = 3
# Live encoder is checkpointed this often (in fits).
_SAVE_EVERY_FITS = 20


def projection_enabled() -> bool:
    """Whether live projection is on. Default ON; ``WEIGHTSLAB_PROJECTION=0``
    (or false/no/off, any case) turns it off."""
    raw = os.environ.get(ENV_ENABLED)
    if raw is None or str(raw).strip() == "":
        return True
    return str(raw).strip().lower() not in _OFF_VALUES


def _env_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def find_ab_params(spread: float = 1.0, min_dist: float = 0.1,
                   iters: int = 400) -> tuple[float, float]:
    """Fit ``(a, b)`` so ``1/(1 + a*d^(2b))`` approximates UMAP's piecewise
    target curve -- the same fit umap-learn does with ``scipy.curve_fit``, done
    here with a few hundred Adam steps so this module needs no scipy.

    Falls back to the well-known ``spread=1.0, min_dist=0.1`` solution if the
    fit misbehaves, so a bad input can never leave the caller without params.
    """
    try:
        xv = th.linspace(0.0, spread * 3.0, 300)
        yv = th.where(xv < min_dist, th.ones_like(xv),
                      th.exp(-(xv - min_dist) / spread))
        log_a = th.zeros(1, requires_grad=True)
        log_b = th.zeros(1, requires_grad=True)
        opt = th.optim.Adam([log_a, log_b], lr=0.05)
        for _ in range(iters):
            opt.zero_grad()
            pred = 1.0 / (1.0 + log_a.exp() * xv.pow(2.0 * log_b.exp()))
            loss = ((pred - yv) ** 2).mean()
            loss.backward()
            opt.step()
        a = float(log_a.exp().item())
        b = float(log_b.exp().item())
        if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
            raise ValueError("degenerate fit")
        return a, b
    except Exception:
        # umap-learn's values for spread=1.0 / min_dist=0.1.
        return 1.5769434, 0.8950608


class ProjectionEncoder(nn.Module):
    """The parametric half of parametric UMAP: ``features -> R^out_dim``.

    Deliberately small (two hidden layers). It has to keep up with a moving
    representation on a per-batch budget, and a bigger encoder mostly buys
    overfitting to whatever the last few batches looked like.
    """

    def __init__(self, in_dim: int, out_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


def membership_high_dim(feats: th.Tensor, n_neighbors: int,
                        bisect_steps: int = 32) -> th.Tensor:
    """UMAP's fuzzy simplicial set for one batch, as a dense ``(B, B)`` matrix.

    ``rho_i`` is i's distance to its nearest neighbour and ``sigma_i`` is
    binary-searched so that ``sum_j exp(-(d_ij - rho_i)/sigma_i) == log2(k)``.
    That per-point normalisation is the whole reason UMAP tolerates wildly
    varying local density -- without it, dense regions dominate the loss.
    """
    n = feats.shape[0]
    k = max(2, min(int(n_neighbors), n - 1))

    dist = th.cdist(feats, feats)
    dist = dist + th.eye(n, device=feats.device) * 1e12  # exclude self

    knn_d, knn_i = th.topk(dist, k, dim=1, largest=False)
    rho = knn_d[:, 0:1]                                   # (B,1)
    centered = (knn_d - rho).clamp(min=0.0)

    target = float(np.log2(k))
    lo = th.full((n, 1), 1e-6, device=feats.device)
    hi = th.full((n, 1), 1e4, device=feats.device)
    for _ in range(bisect_steps):
        mid = (lo + hi) / 2.0
        total = th.exp(-centered / mid).sum(dim=1, keepdim=True)
        too_big = total > target
        hi = th.where(too_big, mid, hi)
        lo = th.where(too_big, lo, mid)
    sigma = ((lo + hi) / 2.0).clamp(min=1e-6)

    mu = th.zeros(n, n, device=feats.device)
    mu.scatter_(1, knn_i, th.exp(-centered / sigma))
    # Fuzzy union: an edge counts if EITHER endpoint considers it a neighbour.
    return mu + mu.t() - mu * mu.t()


def first_tensor(payload):
    """The first tensor inside whatever a module handed us.

    Real models return far more than a bare tensor: tuples
    (``(logits, hidden)``), HuggingFace ``ModelOutput`` dicts, detection heads
    returning lists of per-level maps. Reaching the tensor is the difference
    between "works on a toy CNN" and "works on the model the user actually has",
    so unwrap one level of the usual containers rather than giving up.
    """
    if isinstance(payload, th.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in ("last_hidden_state", "hidden_states", "features", "logits"):
            value = payload.get(key)
            if isinstance(value, th.Tensor):
                return value
        payload = list(payload.values())
    if isinstance(payload, (tuple, list)):
        for item in payload:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def flatten_features(tensor: th.Tensor, feature_last: bool) -> th.Tensor:
    """Reduce an activation of any rank to ``(B, F)``.

    WHICH axis holds the features depends on what was hooked, and getting this
    backwards silently projects the wrong thing:

    * ``feature_last=True`` -- the INPUT of an ``nn.Linear``. Linear always acts
      on the last axis, so that axis is the feature axis whatever the rank:
      ``(B, T, C)`` from a transformer pools over the tokens ``T`` and keeps the
      model dim ``C``. Pooling the other way round would hand the projection a
      per-token scalar and call it an embedding.
    * ``feature_last=False`` -- the OUTPUT of a conv/norm/pool, which is
      channels-first: ``(B, C, H, W)`` pools over the spatial axes and keeps
      ``C``.

    Both collapse to ``(B, F)``; a tensor already 2-D is returned unchanged.
    """
    if tensor.ndim < 2:
        return tensor
    if tensor.ndim == 2:
        return tensor
    if feature_last:
        return tensor.mean(dim=tuple(range(1, tensor.ndim - 1)))
    return tensor.mean(dim=tuple(range(2, tensor.ndim)))


class ProjectionTracker:
    """Owns the encoder, the feature hook and the write-back of coordinates.

    Lifecycle, per tracked step::

        forward hook (detached features)
            -> observe_batch(ids, step)   # from the loss wrapper, where the
                                          # sample ids for that same forward
                                          # pass are known
            -> fit the encoder on the batch
            -> write umap_x / umap_y / umap_z per sample

    Pairing features with ids positionally at the loss is the same trick
    ``ctx.logits`` already uses: the hook cannot see sample ids, the loss cannot
    see intermediate activations, and both describe the same batch in the same
    order.
    """

    def __init__(self, out_dim: int = 3, n_neighbors: int = 15,
                 every_n_steps: int = 50, inner_steps: int = 4,
                 lr: float = 1e-3, min_dist: float = 0.1, spread: float = 1.0,
                 repulsion_strength: float = 1.0, signal_prefix: str = "umap",
                 graph_size: int | None = None):
        self.out_dim = int(out_dim)
        self.n_neighbors = int(n_neighbors)
        self.every_n_steps = max(1, int(every_n_steps))
        self.inner_steps = max(1, int(inner_steps))
        self.lr = float(lr)
        self.repulsion_strength = float(repulsion_strength)
        self.signal_prefix = str(signal_prefix)

        self.a, self.b = find_ab_params(spread=spread, min_dist=min_dist)

        # Rolling buffer of recent (features, ids), fitted on instead of the
        # bare batch.
        #
        # This is not an optimisation, it is a correctness fix. UMAP's graph is
        # a k-NEAREST-neighbour graph, so with a batch of 16 and k=15 every
        # sample is every other sample's neighbour: the graph is complete, the
        # attractive term pulls on all pairs equally and the projection has no
        # local structure to preserve. Measured on clustered data, batch 16
        # separates within- from between-cluster membership by 3x; batch 512
        # by ~10^11. Buffering decouples graph quality from whatever batch size
        # the user's training loop happens to use.
        self.graph_size = int(graph_size or _env_int(ENV_GRAPH, 512))
        self._buffer_feats: list = []
        self._buffer_ids: list = []

        self._encoder: ProjectionEncoder | None = None
        self._optimizer: th.optim.Optimizer | None = None
        # Feature width the encoder was built for. Compared on every fit so a
        # live architecture edit rebuilds it instead of raising forever.
        self._encoder_dim: int | None = None
        self._previous_dim: int | None = None
        self.rebuilds = 0
        self._handle = None
        self._pending: th.Tensor | None = None
        # Whether the hooked module was in train mode when it produced
        # _pending. Eval batches are PLACED by the current encoder but never
        # FIT on: held-out samples belong in the picture, and letting them
        # shape the projection would leak the eval set into it.
        self._pending_training = True
        self._lock = threading.Lock()
        # save_signals is how coordinates get written, and save_signals also
        # calls back into observe_batch -- without this the write-back would
        # recurse into another fit.
        self._writing_back = False
        self._last_step_run = -1
        self._missed_batches = 0
        self.steps_trained = 0
        self.samples_written = 0
        # The fit count at the last encoder checkpoint (see _write_coords).
        self._last_saved_fit = 0
        self.last_loss: float | None = None

    # ------------------------------------------------------------------ hook
    def attach(self, module: nn.Module, use_input: bool = True,
               feature_last: bool | None = None):
        """Hook *module*.

        ``use_input`` captures what flows INTO it (the representation) rather
        than its output (usually logits). ``feature_last`` says which axis holds
        the features -- ``True`` for a Linear's input, ``False`` for a conv's
        channels-first tensor. It defaults to ``use_input`` only to keep the
        two-argument call working; :func:`pick_embedding_layer` always states it.
        """
        self.detach()
        if feature_last is None:
            feature_last = use_input

        def _hook(_mod, inputs, output):
            try:
                tensor = first_tensor(inputs if use_input else output)
                if tensor is None or tensor.ndim < 2:
                    return
                self._pending = flatten_features(
                    tensor.detach().float(), feature_last=feature_last)
                self._pending_training = bool(getattr(_mod, "training", True))
            except Exception as exc:  # never break a forward pass
                logger.debug(f"[projection] feature hook failed: {exc}")

        self._handle = module.register_forward_hook(_hook)
        return self._handle

    def detach(self) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None

    # ------------------------------------------------------------- training
    def _ensure_encoder(self, in_dim: int, device) -> None:
        """Build the encoder, and REBUILD it when the feature width changes.

        This is not a defensive nicety in WeightsLab: editing the architecture
        of a running model -- pruning neurons, growing a layer -- is a headline
        feature, and it changes the hooked layer's width underneath us. An
        encoder pinned to the old width then raises on every batch forever,
        which the caller's broad ``except`` turns into a projection that
        silently stops updating after the user's first edit.

        Rebuilding loses the learned layout, so the cloud visibly re-lays-out.
        That is the honest outcome -- the representation being projected really
        did change -- and it is logged at INFO so the jump has an explanation.
        """
        if self._encoder is not None and self._encoder_dim == in_dim:
            # The model may also have been moved (.cuda() after wrapping), or
            # the encoder may have been adopted from an offline fit that ran on
            # another device. Follow it rather than dying on a mismatch.
            if next(self._encoder.parameters()).device != device:
                self._encoder.to(device)
                # Adam's state (exp_avg, exp_avg_sq) does NOT follow .to(), so
                # step() would then mix devices and raise on every batch --
                # which the caller's broad except turns into a projection that
                # silently stops. Rebuild rather than migrate the state: a few
                # steps of lost momentum is nothing next to that.
                self._optimizer = th.optim.Adam(self._encoder.parameters(), lr=self.lr)
            return

        rebuilt = self._encoder is not None
        if rebuilt:
            # Buffered features describe the OLD layer width; mixing them into
            # the next graph would be comparing different spaces.
            self._clear_buffer()
        self._encoder = ProjectionEncoder(in_dim, self.out_dim).to(device)
        self._optimizer = th.optim.Adam(self._encoder.parameters(), lr=self.lr)
        self._encoder_dim = in_dim
        self.rebuilds += 1 if rebuilt else 0
        if rebuilt:
            logger.info(
                f"[projection] feature width changed {self._previous_dim} -> {in_dim} "
                f"(architecture edit?); encoder rebuilt, so the layout restarts"
            )
        else:
            logger.info(
                f"[projection] parametric UMAP encoder {in_dim} -> {self.out_dim} "
                f"(a={self.a:.4f}, b={self.b:.4f}, every {self.every_n_steps} steps)"
            )
        self._previous_dim = in_dim

    def _umap_loss(self, feats: th.Tensor, target_p: th.Tensor) -> th.Tensor:
        emb = self._encoder(feats)
        d2 = th.cdist(emb, emb).pow(2).clamp(min=1e-12)
        nu = 1.0 / (1.0 + self.a * d2.pow(self.b))
        nu = nu.clamp(1e-6, 1.0 - 1e-6)

        off_diag = 1.0 - th.eye(feats.shape[0], device=feats.device)
        attractive = -(target_p * th.log(nu)) * off_diag
        repulsive = -((1.0 - target_p) * th.log(1.0 - nu)) * off_diag
        return (attractive.sum()
                + self.repulsion_strength * repulsive.sum()) / off_diag.sum()

    def _push_buffer(self, feats: th.Tensor, ids: list) -> th.Tensor:
        """Add this batch to the rolling buffer and return what to fit on.

        Keeps the most recent ``graph_size`` samples. Recency matters: the
        representation is moving, so features from thousands of steps ago
        describe a model that no longer exists and would anchor the layout to
        the past.
        """
        self._buffer_feats.append(feats)
        self._buffer_ids.extend(ids)
        buffered = th.cat(self._buffer_feats, dim=0)
        if buffered.shape[0] > self.graph_size:
            buffered = buffered[-self.graph_size:]
            self._buffer_ids = self._buffer_ids[-self.graph_size:]
        self._buffer_feats = [buffered]
        return buffered

    def _clear_buffer(self) -> None:
        self._buffer_feats = []
        self._buffer_ids = []

    def observe_batch(self, batch_ids, step: int | None) -> bool:
        """Pair the hooked features with *batch_ids*, fit a little, write the
        coordinates. Returns True when this call actually ran a projection.

        Every failure mode here is swallowed: this is instrumentation living
        inside somebody's training loop, and it must not be able to take the
        run down.
        """
        if self._handle is None or batch_ids is None or self._writing_back:
            return False
        step_i = int(step or 0)
        # The every_n_steps gate is about FITTING, and only training batches
        # fit. Applying it to eval batches as well is why the cloud showed the
        # train split and nothing else: evaluation does not advance the step, so
        # every batch of a validation pass arrived carrying the same step
        # number -- the first was rejected unless that step happened to be a
        # multiple of every_n_steps, and all the rest were rejected by
        # `step_i == self._last_step_run`. A whole split ended up with no
        # coordinates at all. Placing an eval batch costs one encoder forward
        # and no gradient, so there is nothing to ration.
        if self._pending_training and (step_i == self._last_step_run
                                       or (step_i % self.every_n_steps) != 0):
            return False

        with self._lock:
            feats = self._pending
            training = self._pending_training
            self._pending = None
            if feats is None:
                # The auto-picked layer never ran (an auxiliary head, a branch
                # this forward skipped). Silence here means the board just never
                # appears and nobody knows why, so say it once -- the fix is a
                # one-line projection={"layer": ...}.
                self._missed_batches += 1
                if self._missed_batches == _MISS_WARN_AFTER:
                    logger.warning(
                        f"[projection] the hooked layer has not produced features "
                        f"in {_MISS_WARN_AFTER} visits -- it is probably not on this "
                        f"model's forward path. Point it somewhere else with "
                        f"wl.watch_or_edit(model, flag='model', "
                        f"projection={{'layer': '<named_modules() name>'}}), or "
                        f"set {ENV_ENABLED}=0 to turn the projection off."
                    )
                return False
            self._missed_batches = 0
            try:
                ids = [str(i) for i in (
                    batch_ids.detach().cpu().tolist()
                    if hasattr(batch_ids, "detach") else list(batch_ids))]
            except Exception:
                return False
            if len(ids) != feats.shape[0] or feats.shape[0] < 4:
                # Mismatched pairing (or a batch too small for a kNN graph):
                # writing it would attach one sample's features to another's id.
                return False

            # Only a fit claims the step. An eval batch that claimed it would
            # lock out the training batch arriving at the same step, which is
            # the very thing this guard exists to let through.
            if training:
                self._last_step_run = step_i
            try:
                self._ensure_encoder(feats.shape[1], feats.device)
                # Eval batches are placed, not learned from (see
                # _pending_training). An encoder that has never fit yet still
                # places them -- at random, but it will be refit within
                # every_n_steps and the coordinates overwritten.
                if training:
                    graph_feats = self._push_buffer(feats, ids)
                    target_p = membership_high_dim(graph_feats, self.n_neighbors)
                    loss = None
                    for _ in range(self.inner_steps):
                        self._optimizer.zero_grad(set_to_none=True)
                        loss = self._umap_loss(graph_feats, target_p)
                        loss.backward()
                        self._optimizer.step()
                    self.last_loss = float(loss.detach().item())
                    self.steps_trained += 1

                # Emit for everything in the buffer, not just this batch: the
                # encoder just moved, so every buffered sample's old
                # coordinates are stale, and re-encoding them costs one more
                # forward pass while multiplying the coverage each visit buys.
                with th.no_grad():
                    if training and self._buffer_feats:
                        emit = self._buffer_feats[0]
                        ids = list(self._buffer_ids)
                    else:
                        emit = feats
                    coords = self._encoder(emit).detach().cpu().numpy()
            except Exception as exc:
                logger.debug(f"[projection] fit failed at step {step_i}: {exc}")
                return False

        return self._write_coords(ids, coords, step_i)

    def _write_coords(self, ids: list, coords: np.ndarray, step: int) -> bool:
        """Persist coordinates as ordinary per-sample signals.

        Going through ``save_signals`` (rather than poking the dataframe) means
        the projection inherits everything the rest of the pipeline already
        has: H5 persistence, the metadata column list, sorting, histograms.
        """
        try:
            from weightslab.src import save_signals
            axes = ["x", "y", "z"][:coords.shape[1]]
            # Tensors, not numpy arrays: save_signals' own `normalize` handles
            # torch tensors and lists, and returns None for a bare ndarray --
            # which lands the column in the dataframe holding nothing at all.
            signals = {
                f"{self.signal_prefix}_{axis}":
                    th.from_numpy(np.ascontiguousarray(coords[:, i], dtype=np.float32))
                for i, axis in enumerate(axes)
            }
            self._writing_back = True
            try:
                save_signals(signals=signals, batch_ids=ids, step=step, log=False)
            finally:
                self._writing_back = False
            self.samples_written += len(ids)
            # Checkpoint occasionally, so a crash or a Ctrl-C does not cost the
            # layout. Cheap: a few hundred KB of encoder weights.
            #
            # Keyed on the fit that was last SAVED, not merely on the counter
            # being divisible. `steps_trained` only advances on a fit, and this
            # write-back now also runs for eval batches (they are placed, not
            # learned from) -- so with the counter sitting on a multiple of
            # _SAVE_EVERY_FITS, every batch of a full evaluation pass wrote the
            # encoder to disk again. A single pass produced a burst of ~16
            # saves a second apart, for one fit's worth of weights.
            if (self.steps_trained
                    and self.steps_trained % _SAVE_EVERY_FITS == 0
                    and self.steps_trained != self._last_saved_fit):
                self._last_saved_fit = self.steps_trained
                save_projection(self)
            return True
        except Exception as exc:
            logger.debug(f"[projection] coordinate write-back failed: {exc}")
            return False

    def stats(self) -> dict:
        return {
            "enabled": self._handle is not None,
            "out_dim": self.out_dim,
            "every_n_steps": self.every_n_steps,
            "graph_size": self.graph_size,
            "buffered": len(self._buffer_ids),
            "fits": self.steps_trained,
            "rebuilds": self.rebuilds,
            "feature_dim": self._encoder_dim,
            "samples_written": self.samples_written,
            "last_loss": self.last_loss,
            "columns": [f"signals//{self.signal_prefix}_{a}"
                        for a in ["x", "y", "z"][:self.out_dim]],
        }


# --------------------------------------------------------------------------
# Module-level singleton: one projection per process, mirroring how the model,
# logger and dataframe are single registered objects in the ledger.
# --------------------------------------------------------------------------
_TRACKER = None


def get_tracker():
    return _TRACKER


# `aux_classifier` (torchvision FCN/DeepLab), `aux_head`/`auxiliary_head`
# (mmsegmentation), `aux` on its own. Matched against the dotted module name.
_AUXILIARY_NAME = re.compile(r"(^|\.)(aux|auxiliary)(_|\.|$)", re.IGNORECASE)


def pick_embedding_layer(model: nn.Module):
    """Choose what to hook, for a model we know nothing about.

    The rule is one idea applied consistently: **hook the input of the last
    parameterised layer**. That layer is the model's classifier/head, so what
    flows INTO it is the representation -- the thing worth projecting -- while
    what comes OUT is class scores.

    Getting this wrong is quiet and expensive. Reading a segmentation head's
    *output* yields ``(B, num_classes)``, which is a perfectly well-shaped
    tensor that UMAP will happily lay out; you would just be looking at a map
    of class scores and believing it was the model's representation.

    Order of preference:

    1. last ``nn.Linear`` -> its input; features are on the LAST axis (Linear
       acts on that axis), so ``(B, T, C)`` keeps ``C``;
    2. last conv (``Conv1d/2d/3d``, the head of a fully-convolutional model like
       FCN/DeepLab/U-Net) -> its input; channels-first, so ``(B, C, H, W)``
       keeps ``C``;
    3. otherwise the last leaf module's output, channels-first -- a genuine
       last resort for a model with no parameterised layer to anchor on.

    AUXILIARY heads are skipped. torchvision's FCN and DeepLab register
    ``aux_classifier`` AFTER ``classifier``, so "the last conv" picked the
    auxiliary head -- which branches off an earlier backbone stage, so its
    input is a shallower representation than the one the model actually
    predicts from, and which torchvision does not even run outside training, so
    on an evaluation pass the hook produced nothing at all and the cloud simply
    stopped gaining test-split points. Both failures are silent.

    Returns ``(module, use_input, feature_last)``.
    """
    # main = anything not under an auxiliary head; aux = the fallback, used
    # only if a model turns out to be nothing BUT an auxiliary head.
    last_linear = {"main": None, "aux": None}
    last_conv = {"main": None, "aux": None}
    last_leaf = {"main": None, "aux": None}
    for name, module in model.named_modules():
        bucket = "aux" if _AUXILIARY_NAME.search(name) else "main"
        if isinstance(module, nn.Linear):
            last_linear[bucket] = module
        elif isinstance(module, nn.modules.conv._ConvNd):
            last_conv[bucket] = module
        if len(list(module.children())) == 0:
            last_leaf[bucket] = module

    linear = last_linear["main"] or last_linear["aux"]
    conv = last_conv["main"] or last_conv["aux"]
    leaf = last_leaf["main"] or last_leaf["aux"]
    if linear is not None:
        return linear, True, True
    if conv is not None:
        return conv, True, False
    return leaf, False, False


def attach_projection(model: nn.Module, layer=None, **kwargs):
    """Start projecting *model*'s features to 3-D. Returns the tracker, or
    ``None`` when the feature is disabled or no layer could be hooked.

    *layer* may be a module, a ``named_modules()`` name, or ``None`` to
    auto-pick (see :func:`pick_embedding_layer`).
    """
    global _TRACKER
    if not projection_enabled():
        logger.debug(f"[projection] disabled via {ENV_ENABLED}")
        return None

    def _describe(module):
        """Hook policy for a user-named layer: same rule as the auto-pick --
        read the INPUT of a parameterised layer, and know which axis is which."""
        if isinstance(module, nn.Linear):
            return module, True, True
        if isinstance(module, nn.modules.conv._ConvNd):
            return module, True, False
        return module, False, False

    target, use_input, feature_last = None, True, True
    if isinstance(layer, str):
        found = dict(model.named_modules()).get(layer)
        if found is None:
            logger.warning(f"[projection] layer {layer!r} not found; auto-picking")
        else:
            target, use_input, feature_last = _describe(found)
    elif isinstance(layer, nn.Module):
        target, use_input, feature_last = _describe(layer)
    if target is None:
        target, use_input, feature_last = pick_embedding_layer(model)
    if target is None:
        logger.debug("[projection] no hookable layer found")
        return None

    kwargs.setdefault("out_dim", _env_int(ENV_DIM, 3))
    kwargs.setdefault("every_n_steps", _env_int(ENV_EVERY, 50))
    kwargs.setdefault("n_neighbors", _env_int(ENV_NEIGHBORS, 15))

    tracker = ProjectionTracker(**kwargs)
    tracker.attach(target, use_input=use_input, feature_last=feature_last)
    _TRACKER = tracker
    # Continue a previous run's layout instead of re-laying it out from a
    # random initialisation -- otherwise the coordinates already in the
    # dataframe describe a different embedding than the one about to overwrite
    # them, and the cloud is two layouts at once.
    load_projection(tracker)
    logger.info(
        f"[projection] attached to {type(target).__name__} "
        f"({'input' if use_input else 'output'}, "
        f"{'features last' if feature_last else 'channels first'}); "
        f"disable with {ENV_ENABLED}=0"
    )
    return tracker


def resolve_module(model):
    """The real ``nn.Module`` behind whatever handle the caller is holding.

    ``wl.watch_or_edit(net, flag="model")`` hands back a ledger ``Proxy``, not
    an ``nn.Module`` -- and that is the handle every training script keeps. On
    it, ``named_modules()`` yields nothing, so a layer lookup against it finds
    no layers and reports that none exist, which is a baffling thing to be told
    about a model you can see. Descend the wrapper's ``.model`` chain to the
    module that actually owns the layers.

    Returns ``None`` when there is no ``nn.Module`` to be found.
    """
    current = model
    for _ in range(6):   # bounded: wrappers nest, but not deeply
        if isinstance(current, nn.Module):
            # A WeightsLab wrapper is itself a Module but delegates to .model;
            # prefer the inner one, whose layer names are the ones the user
            # sees in their own model definition.
            inner = getattr(current, "model", None)
            if isinstance(inner, nn.Module) and inner is not current:
                current = inner
                continue
            return current
        # Not a Module yet (a Proxy, or a plain wrapper object): keep descending.
        nxt = None
        for attr in ("model", "module", "_model", "net"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                nxt = candidate
                break
        if nxt is None:
            return None
        current = nxt
    return current if isinstance(current, nn.Module) else None


def _default_unpack(batch):
    """``(inputs, sample_ids)`` from a WeightsLab batch.

    Every tracked loader yields ``(data, uid, ...)`` -- see the usecases, which
    all destructure as ``for x, ids, y in loader``. Pass ``unpack=`` for a
    loader that does something else.
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(
        "could not read (inputs, sample_ids) from the batch; pass "
        "unpack=lambda batch: (inputs, ids)")


def project_dataset(model, dataloader, layer=None, prefix: str = "umap",
                    epochs: int = 10, fit_batch: int = 512,
                    max_samples: int | None = None, out_dim: int | None = None,
                    n_neighbors: int | None = None, lr: float = 1e-3,
                    min_dist: float = 0.1, spread: float = 1.0,
                    unpack=None, device=None, verbose: bool = True,
                    adopt: bool | None = None) -> dict:
    """Re-project a dataset **offline**, from a model that is not training.

    The live projection commits to one layer the moment training starts. This is
    the answer to "that was the wrong layer": load the experiment as usual (the
    checkpoint restores the weights), then re-project from whichever layer you
    actually wanted, over the whole split, with no training step involved.

    It also fits *better* than the live path can. Online, the UMAP graph is
    whatever one batch happened to contain; here every feature is collected
    first, so each epoch draws its neighbourhoods from shuffled minibatches of
    the **whole** dataset.

    *prefix* is the point of the exercise: writing to ``prefix="umap_layer3"``
    stores ``signals//umap_layer3_{x,y,z}`` alongside any existing projection
    rather than overwriting it, so several candidate layers can coexist and be
    switched between in the Studio board.

    Args:
        model: the model to read features from. Used in ``eval()``, under
            ``no_grad`` -- its weights are never touched.
        dataloader: any iterable of batches; by default ``batch[0]`` is the
            input and ``batch[1]`` the sample ids (see :func:`_default_unpack`).
        layer: module, ``named_modules()`` name, or ``None`` to auto-pick.
        prefix: signal prefix for the written coordinates.
        epochs: passes over the collected features.
        fit_batch: minibatch size for fitting. This is the UMAP graph size, so
            bigger means better-connected neighbourhoods and more memory
            (it is an N x N distance matrix).
        max_samples: stop collecting after this many samples.
        unpack: ``batch -> (inputs, sample_ids)`` override.
        device: where to fit; defaults to the model's device.
        adopt: hand the result to the LIVE projection, so that resuming training
            continues refining *this* fit (same layer, same encoder, same
            prefix) instead of starting a new one from scratch.

            ``None`` (default) means "adopt when it would otherwise corrupt":
            writing to the prefix the live projection owns and NOT adopting
            leaves training to overwrite these coordinates, sample by sample,
            with a freshly-initialised encoder's output -- a cloud that is
            part this layout and part another, which is meaningless. Pass
            ``False`` to keep the live projection untouched anyway (fine when
            *prefix* differs: that snapshot is then never written again).

    Returns:
        dict: ``{"samples", "feature_dim", "prefix", "columns", "epochs",
        "final_loss", "layer"}``.

    Example:
        >>> model = wl.watch_or_edit(MyNet(), flag="model")   # weights restored
        >>> loader = wl.watch_or_edit(train_ds, flag="data", loader_name="train_loader")
        >>> wl.project_dataset(model, loader, layer="backbone.layer3",
        ...                    prefix="umap_layer3", epochs=20)
    """
    unpack = unpack or _default_unpack
    out_dim = out_dim if out_dim is not None else _env_int(ENV_DIM, 3)
    n_neighbors = n_neighbors if n_neighbors is not None else _env_int(ENV_NEIGHBORS, 15)

    # The caller almost certainly holds the ledger Proxy that watch_or_edit
    # returned, not the module itself. Hook and inspect the real module; keep
    # calling the ORIGINAL handle for the forward pass, so whatever the wrapper
    # does on the way through (masking, guards) still happens.
    module = resolve_module(model)
    if module is None:
        raise TypeError(
            "project_dataset: could not find an nn.Module on the object passed "
            "as `model`. Pass the model (or the handle wl.watch_or_edit returned).")

    target, use_input, feature_last = None, True, True
    if isinstance(layer, str):
        found = dict(module.named_modules()).get(layer)
        if found is None:
            names = [n for n, _ in module.named_modules() if n]
            raise ValueError(
                f"project_dataset: layer {layer!r} not found. Available: "
                f"{names[:20]}{'...' if len(names) > 20 else ''}")
        target = found
    elif isinstance(layer, nn.Module):
        target = layer
    if target is not None:
        if isinstance(target, nn.Linear):
            use_input, feature_last = True, True
        elif isinstance(target, nn.modules.conv._ConvNd):
            use_input, feature_last = True, False
        else:
            use_input, feature_last = False, False
    else:
        target, use_input, feature_last = pick_embedding_layer(module)
    if target is None:
        raise ValueError("project_dataset: no hookable layer found")

    tracker = ProjectionTracker(out_dim=out_dim, n_neighbors=n_neighbors, lr=lr,
                                min_dist=min_dist, spread=spread,
                                signal_prefix=prefix)
    tracker.attach(target, use_input=use_input, feature_last=feature_last)

    was_training = module.training
    module.eval()
    feats_chunks, id_chunks = [], []
    collected = 0
    # The collection sweep is a full pass over the split with no output of its
    # own -- on a real dataset that is minutes of silence, which reads as a
    # hang. Same tqdm the ledger's own dataset passes use.
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    sweep = tqdm(dataloader, total=total_batches, desc="[projection] features",
                 unit="batch", disable=not verbose, leave=False)
    try:
        with th.no_grad():
            for batch in sweep:
                inputs, ids = unpack(batch)
                # Inputs follow the MODEL, always. `device` says where to fit
                # the encoder, which is a separate question -- sending a cuda
                # model's inputs to the cpu because the caller wanted to fit
                # there just breaks the forward pass.
                try:
                    inputs = inputs.to(next(module.parameters()).device)
                except StopIteration:
                    pass
                tracker._pending = None
                model(inputs)
                captured = tracker._pending
                if captured is None:
                    continue
                # Keep the collection on CPU: a whole split's features at
                # once is exactly the thing that will not fit in VRAM.
                feats_chunks.append(captured.cpu())
                id_chunks.extend(
                    [str(i) for i in (ids.detach().cpu().tolist()
                                      if hasattr(ids, "detach") else list(ids))])
                collected += captured.shape[0]
                if max_samples is not None and collected >= max_samples:
                    break
    finally:
        sweep.close()
        tracker.detach()
        if was_training:
            module.train()

    if not feats_chunks:
        raise RuntimeError(
            f"project_dataset: {type(target).__name__} produced no features over "
            f"the whole loader -- it is probably not on this model's forward path. "
            f"Pick another with layer=<named_modules() name>.")

    feats = th.cat(feats_chunks, dim=0)
    ids = id_chunks[:feats.shape[0]]
    if max_samples is not None:
        feats, ids = feats[:max_samples], ids[:max_samples]
    if feats.shape[0] < 4:
        raise RuntimeError(
            f"project_dataset: only {feats.shape[0]} samples collected; need at "
            f"least 4 to build a neighbourhood graph.")

    # Follow the model rather than grabbing the GPU because one exists: fitting
    # on cuda for a cpu model leaves the encoder somewhere the live projection
    # cannot use it if it later adopts this fit.
    if device is not None:
        fit_device = device
    else:
        try:
            fit_device = next(module.parameters()).device
        except StopIteration:
            fit_device = feats.device
    tracker._ensure_encoder(feats.shape[1], th.device(fit_device))
    if verbose:
        logger.info(
            f"[projection] offline fit: {feats.shape[0]} samples x {feats.shape[1]} "
            f"features from {type(target).__name__}, {epochs} epochs -> '{prefix}'")

    n = feats.shape[0]
    step = max(4, min(int(fit_batch), n))
    final_loss = None
    epoch_bar = tqdm(range(max(1, int(epochs))), desc="[projection] fit",
                     unit="epoch", disable=not verbose, leave=False)
    for epoch in epoch_bar:
        order = th.randperm(n)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n - 3, step):
            chunk = feats[order[start:start + step]].to(fit_device)
            if chunk.shape[0] < 4:
                continue
            graph = membership_high_dim(chunk, n_neighbors)
            tracker._optimizer.zero_grad(set_to_none=True)
            loss = tracker._umap_loss(chunk, graph)
            loss.backward()
            tracker._optimizer.step()
            epoch_loss += float(loss.detach().item())
            batches += 1
        final_loss = epoch_loss / max(1, batches)
        tracker.steps_trained += batches
        epoch_bar.set_postfix(loss=f"{final_loss:.4f}")
    epoch_bar.close()

    coords = []
    with th.no_grad():
        for start in tqdm(range(0, n, step), desc="[projection] encode",
                          unit="chunk", disable=not verbose, leave=False):
            coords.append(tracker._encoder(feats[start:start + step].to(fit_device)).cpu())
    coords = th.cat(coords, dim=0).numpy()

    tracker._write_coords(ids, coords, step=0)
    tracker.last_loss = final_loss
    save_projection(tracker)

    # --- hand over to the live projection, or say why we did not -----------
    live = get_tracker()
    collides = live is not None and live.signal_prefix == prefix
    should_adopt = collides if adopt is None else bool(adopt)
    if should_adopt and live is not None:
        # Re-point the live hook at the layer we just fitted on: adopting the
        # encoder without the layer would feed it activations of a different
        # width, which only triggers a rebuild and throws the fit away.
        live.detach()
        live.attach(target, use_input=use_input, feature_last=feature_last)
        live._encoder = tracker._encoder
        live._optimizer = tracker._optimizer
        live._encoder_dim = tracker._encoder_dim
        live._previous_dim = tracker._encoder_dim
        live.signal_prefix = prefix
        live.out_dim = tracker.out_dim
        if verbose:
            logger.info(
                f"[projection] live projection adopted this fit: training will "
                f"continue refining '{prefix}' from {type(target).__name__}")
    elif collides:
        logger.warning(
            f"[projection] wrote prefix '{prefix}', which the LIVE projection also "
            f"owns. Resuming training will overwrite these coordinates sample by "
            f"sample with a freshly-initialised encoder, mixing two different "
            f"layouts in one cloud. Use a distinct prefix= for a snapshot, or "
            f"adopt=True to have training continue from this fit."
        )

    stats = {
        "samples": int(n),
        "feature_dim": int(feats.shape[1]),
        "prefix": prefix,
        "columns": [f"signals//{prefix}_{a}" for a in ["x", "y", "z"][:out_dim]],
        "epochs": int(epochs),
        "final_loss": final_loss,
        "layer": type(target).__name__,
        "adopted_by_live": bool(should_adopt and live is not None),
    }
    if verbose:
        logger.info(f"[projection] offline projection written: {stats['columns']}")
    return stats


def checkpoint_path(root_log_dir=None, prefix: str = "umap"):
    """Where a projection encoder is saved.

    Deliberately its OWN file, next to the run's logs, not inside the model
    checkpoint: the encoder is not part of your model and must never ride along
    into a weights file someone might deploy. ``None`` when no log directory
    can be resolved.
    """
    import os
    # Falsy, not just None, at every step: an unset hyperparameter comes back
    # as "" rather than None, and testing `is None` there skipped the env
    # fallback entirely and left the encoder unsaveable.
    root = root_log_dir
    if not root:
        try:
            from weightslab.backend.ledgers import get_hyperparams
            root = get_hyperparams().get("root_log_dir")
        except Exception:
            root = None
    if not root:
        root = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR")
    if not root:
        return None
    return os.path.join(str(root), "projection", f"{prefix}_encoder.pt")


def save_projection(tracker=None, root_log_dir=None) -> str | None:
    """Persist the encoder so a later run continues the SAME layout.

    Without this, restarting refits from a random initialisation and every
    sample lands somewhere new -- the coordinates already in the dataframe then
    describe a different embedding than the one still being written, which is
    two layouts mixed in one cloud.
    """
    import os
    tracker = tracker or get_tracker()
    if tracker is None or tracker._encoder is None:
        return None
    path = checkpoint_path(root_log_dir, tracker.signal_prefix)
    if path is None:
        return None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        th.save({
            "state_dict": tracker._encoder.state_dict(),
            "in_dim": tracker._encoder_dim,
            "out_dim": tracker.out_dim,
            "prefix": tracker.signal_prefix,
            "fits": tracker.steps_trained,
            "a": tracker.a,
            "b": tracker.b,
        }, path)
        logger.info(f"[projection] encoder saved to {path}")
        return path
    except Exception as exc:
        logger.debug(f"[projection] could not save encoder: {exc}")
        return None


def load_projection(tracker=None, root_log_dir=None) -> bool:
    """Restore a saved encoder onto *tracker*. True when one was loaded.

    A width mismatch (the hooked layer changed since the save) is a miss, not
    an error: the tracker simply starts fresh rather than loading an encoder
    that cannot accept the features it will be given.
    """
    import os
    tracker = tracker or get_tracker()
    if tracker is None:
        return False
    path = checkpoint_path(root_log_dir, tracker.signal_prefix)
    if not path or not os.path.exists(path):
        return False
    try:
        blob = th.load(path, map_location="cpu", weights_only=False)
        in_dim = int(blob.get("in_dim") or 0)
        if in_dim <= 0 or int(blob.get("out_dim") or 0) != tracker.out_dim:
            return False
        encoder = ProjectionEncoder(in_dim, tracker.out_dim)
        encoder.load_state_dict(blob["state_dict"])
        tracker._encoder = encoder
        tracker._encoder_dim = in_dim
        tracker._previous_dim = in_dim
        tracker._optimizer = th.optim.Adam(encoder.parameters(), lr=tracker.lr)
        tracker.a, tracker.b = float(blob.get("a", tracker.a)), float(blob.get("b", tracker.b))
        tracker.steps_trained = int(blob.get("fits", 0))
        logger.info(
            f"[projection] encoder restored from {path} "
            f"({in_dim} -> {tracker.out_dim}, {tracker.steps_trained} prior fits); "
            f"the layout continues rather than restarting")
        return True
    except Exception as exc:
        logger.debug(f"[projection] could not load encoder: {exc}")
        return False


def detach_projection() -> None:
    """Remove the hook and drop the tracker."""
    global _TRACKER
    if _TRACKER is not None:
        _TRACKER.detach()
    _TRACKER = None


def observe_batch(batch_ids, step) -> bool:
    """Called from the loss wrapper for every logged batch. No-op when the
    feature is off, which is the common case for the hot path."""
    tracker = _TRACKER
    if tracker is None:
        return False
    try:
        return tracker.observe_batch(batch_ids, step)
    except Exception as exc:
        logger.debug(f"[projection] observe_batch failed: {exc}")
        return False
