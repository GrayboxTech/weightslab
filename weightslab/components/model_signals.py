"""Per-step model signals: global gradient norm, per-layer weight/gradient/activation stats.

Everything WeightsLab's public signal API records is keyed by something in the
dataset -- ``save_signals`` by sample, ``save_instance_signals`` by annotation,
``save_group_signals`` by group. Model health is not: a gradient norm belongs to
the optimization STEP that produced it, and the batch behind it is incidental.
``src.save_model_signals`` is the step-keyed write path; this module is what
produces the values so a training script never writes the collection loop by
hand.

Signals emitted (all opt-in, see ``METRICS``):

    metrics/global/grad_norm                whole-model gradient L2 norm
    metrics/global/weights_norm             whole-model parameter L2 norm
    metrics/layer/<layer_id>/grad_norm      that layer's parameter gradients
    metrics/layer/<layer_id>/weights_norm   that layer's parameters
    metrics/layer/<layer_id>/activation_{mean,std,max,min}

``<layer_id>`` is the module id WeightsLab already assigns for architecture ops
(``NetworkWithOps.get_layer_by_id`` / ``module.get_module_id()``), so a layer's
curve here and the same layer in the model panel / a freeze request name the
same thing. Layers without an id fall back to their position in
``model.layers``.

Collection points, and why each one is where it is:

    weights       read off ``p.data`` at flush time. Weights are always there;
                  no hook needed.
    gradients     ``Tensor.register_post_accumulate_grad_hook`` (torch >= 2.1),
                  which fires the instant a parameter's ``.grad`` is final
                  during backward. Deliberately NOT read at flush time: a
                  training loop is free to call ``optimizer.zero_grad()``
                  before anything we control runs again, and by then the
                  gradients are gone. Reading them from the next forward hook
                  instead would report every step's gradients one step late.
    activations   forward hooks on the tracked layers, reduced on-device into
                  0-d tensors and held there. Nothing is copied to the host
                  until the step flushes, which keeps the per-step cost to one
                  device sync no matter how many layers are tracked.

The flush piggybacks on ``optimizer.step()``: the one point in a step where
gradients are guaranteed present AND the step is guaranteed finished. The
optimizer is resolved from the ledger LAZILY, on the first forward rather than
at construction, because a script watches its model BEFORE it builds the
optimizer from ``model.parameters()`` -- see ``_ensure_flush_hooked``. A script
that never registers an optimizer (or a custom loop that steps by hand) can
call ``tracker.flush()`` itself.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

from weightslab.components.tracking import TrackingMode

_LOGGER = logging.getLogger(__name__)

# Every metric this module can emit. Also the default set: a first run should
# show the whole picture, and dropping metrics you don't want is easier than
# discovering ones you didn't know existed. Pass `metrics=[...]` to narrow.
METRICS = (
    "grad_norm",
    "weights_norm",
    "activation_mean",
    "activation_std",
    "activation_max",
    "activation_min",
)

_ACTIVATION_METRICS = frozenset(m for m in METRICS if m.startswith("activation_"))
_PARAM_METRICS = frozenset({"grad_norm", "weights_norm"})

# Module types whose output is not worth an activation curve: containers (their
# output is just their last child's) and shape-only ops (identical statistics to
# their input, so the curve duplicates the layer before it).
_UNINTERESTING_ACTIVATIONS = (
    torch.nn.Sequential,
    torch.nn.ModuleList,
    torch.nn.ModuleDict,
    torch.nn.Flatten,
    torch.nn.Identity,
    torch.nn.Dropout,
)


def _layer_id(module: torch.nn.Module, position: int) -> str:
    """The id this layer is known by elsewhere in WeightsLab, or its position.

    ``get_module_id`` is what the dependency manager assigns and what every
    architecture op (freeze/reset/operate) and the model panel address a layer
    by, so using it here is what makes a layer's signal curve and a layer's
    controls refer to the same layer. A model whose dependencies were never
    computed (``compute_dependencies=False``) has no ids at all, hence the
    positional fallback.
    """
    getter = getattr(module, "get_module_id", None)
    if callable(getter):
        try:
            mid = getter()
            if mid is not None:
                return str(mid)
        except Exception:
            pass
    return str(position)


def _iter_layers(model) -> list:
    """``(layer_id, module)`` for each tracked layer, most specific source first.

    ``model.layers`` is WeightsLab's own linearized view (the same list the
    model panel and architecture ops walk), so it is preferred -- it excludes
    containers and is ordered the way the UI shows them. A plain ``nn.Module``
    that was never wrapped has no such attribute, so fall back to
    ``named_modules()`` minus containers.
    """
    layers = None
    try:
        candidate = getattr(model, "layers", None)
        if candidate:
            layers = list(candidate)
    except Exception:
        layers = None

    if layers:
        return [(_layer_id(m, i), m) for i, m in enumerate(layers)]

    inner = getattr(model, "model", model)
    out = []
    for i, (_, module) in enumerate(inner.named_modules()):
        if module is inner or isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue
        out.append((_layer_id(module, i), module))
    return out


class ModelSignalTracker:
    """Hooks on one model; emits its signals once per training step.

    One instance per model. Constructing it installs the hooks; ``remove()``
    takes them all off again (and is idempotent, so it is safe in a ``finally``).
    """

    def __init__(
        self,
        model,
        metrics: Iterable[str] = METRICS,
        every_n_steps: int = 1,
        layer_ids: Optional[Iterable] = None,
        include_global: bool = True,
    ):
        unknown = sorted(set(metrics) - set(METRICS))
        if unknown:
            raise ValueError(
                f"Unknown model signal(s) {unknown}. Available: {list(METRICS)}"
            )

        self.model = model
        self.metrics = frozenset(metrics)
        self.every_n_steps = max(1, int(every_n_steps))
        self.include_global = include_global
        self._wanted = {str(l) for l in layer_ids} if layer_ids is not None else None

        # {layer_id: {metric: 0-d tensor}} for the step being collected. Values
        # stay on-device until flush() -- see the module docstring.
        self._activations: dict = {}
        # {layer_id: [0-d tensor of grad**2, ...]}, one entry per parameter of
        # that layer. Squared, so a layer norm is sqrt(sum(...)) and the global
        # norm is sqrt(sum over every layer) -- summing the norms themselves
        # would be wrong (that is an L1 over L2s, not an L2).
        self._grad_sq: dict = {}
        self._handles: list = []
        self._flush_hooked = False
        self._removed = False
        self._layers = _iter_layers(model)
        if self._wanted is not None:
            self._layers = [(lid, m) for lid, m in self._layers if lid in self._wanted]

        self._install()

    # -- setup ------------------------------------------------------------ #

    def _install(self) -> None:
        want_activations = bool(self.metrics & _ACTIVATION_METRICS)
        want_grads = "grad_norm" in self.metrics

        for layer_id, module in self._layers:
            if want_activations and not isinstance(module, _UNINTERESTING_ACTIVATIONS):
                self._handles.append(
                    module.register_forward_hook(self._make_activation_hook(layer_id))
                )
            if want_grads:
                for param in module.parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    try:
                        self._handles.append(
                            param.register_post_accumulate_grad_hook(
                                self._make_grad_hook(layer_id)
                            )
                        )
                    except AttributeError:
                        # torch < 2.1. Gradients then simply aren't collected;
                        # weights and activations still are, and the run is not
                        # worth failing over a missing curve.
                        _LOGGER.warning(
                            "torch %s has no register_post_accumulate_grad_hook; "
                            "gradient-norm signals are unavailable",
                            torch.__version__,
                        )
                        want_grads = False
                        break

        if not self._layers:
            _LOGGER.warning("track_model_signals: no layers matched; nothing will be logged")
            return

        # Driving _ensure_flush_hooked() from a pre-hook on the first tracked
        # layer, rather than from the metric hooks themselves, covers the case
        # where the requested metric set installs no per-step hooks at all
        # (`metrics=["weights_norm"]` reads weights at flush time and hooks
        # nothing) -- without this, that configuration would collect correctly
        # and then never flush.
        first_module = self._layers[0][1]
        self._handles.append(
            first_module.register_forward_pre_hook(lambda *_: self._ensure_flush_hooked())
        )

    def _ensure_flush_hooked(self) -> None:
        """Wrap the watched optimizer's ``step`` so flush() runs once per step.

        Lazy on purpose: a script calls ``watch_or_edit(model, flag="model")``
        before it builds the optimizer out of ``model.parameters()``, so at
        construction time there is nothing to wrap yet. By the first forward
        pass there always is.
        """
        if self._flush_hooked:
            return
        self._flush_hooked = True  # set first: one attempt, success or not

        from weightslab.backend.ledgers import get_optimizer

        try:
            optimizer = get_optimizer()
        except Exception:
            optimizer = None
        step_fn = getattr(optimizer, "step", None)
        if optimizer is None or not callable(step_fn):
            _LOGGER.info(
                "track_model_signals: no watched optimizer found; call "
                "tracker.flush() yourself after backward()"
            )
            return

        tracker = self

        def step_and_flush(*args, **kwargs):
            # Flush BEFORE stepping: these signals describe the gradients and
            # the weights that this step is about to consume, which is the
            # pairing that makes them readable together ("this gradient norm
            # was applied to those weights"). Flushing after would report the
            # post-update weights against the pre-update gradients.
            try:
                tracker.flush()
            except Exception:
                _LOGGER.debug("model signal flush failed", exc_info=True)
            return step_fn(*args, **kwargs)

        try:
            optimizer.step = step_and_flush
        except Exception:
            _LOGGER.info("track_model_signals: could not wrap optimizer.step; call flush() yourself")

    # -- hooks ------------------------------------------------------------ #

    def _should_collect(self) -> bool:
        """Only inside a training step, and only on a sampled step.

        ``tracking_mode`` is set by ``guard_training_context`` /
        ``guard_testing_context``, which is a stronger gate than
        ``model.training`` or ``torch.is_grad_enabled()``: several shipped
        examples run their eval pass without calling ``model.eval()`` and
        without ``torch.no_grad()``, so both of those would happily report
        eval-pass activations as training signals.
        """
        if self._removed:
            return False
        if getattr(self.model, "tracking_mode", None) is not TrackingMode.TRAIN:
            return False
        if self.every_n_steps == 1:
            return True
        age = self._age()
        return age is not None and age % self.every_n_steps == 0

    def _age(self) -> Optional[int]:
        getter = getattr(self.model, "get_age", None)
        if callable(getter):
            try:
                value = getter()
                return None if value is None else int(value)
            except Exception:
                return None
        return None

    def _make_activation_hook(self, layer_id: str):
        def hook(_module, _inputs, output):
            if not self._should_collect():
                return
            tensor = output if isinstance(output, torch.Tensor) else None
            if tensor is None and isinstance(output, (tuple, list)) and output:
                tensor = output[0] if isinstance(output[0], torch.Tensor) else None
            if tensor is None or tensor.numel() == 0:
                return
            # Reduce on-device and keep it there; float() casts half/bf16 so a
            # mixed-precision run doesn't overflow the sum inside std/mean.
            values = tensor.detach().float()
            stats = {}
            if "activation_mean" in self.metrics:
                stats["activation_mean"] = values.mean()
            if "activation_std" in self.metrics:
                # std() of a single element is NaN by definition (zero degrees
                # of freedom); report 0 spread instead of poisoning the curve.
                stats["activation_std"] = (
                    values.std() if values.numel() > 1 else torch.zeros((), device=values.device)
                )
            if "activation_max" in self.metrics:
                stats["activation_max"] = values.max()
            if "activation_min" in self.metrics:
                stats["activation_min"] = values.min()
            # A layer called twice in one forward (weight sharing, a recurrent
            # block) keeps its LAST call, matching how the rest of WeightsLab
            # treats a repeated write within a step.
            self._activations[layer_id] = stats

        return hook

    def _make_grad_hook(self, layer_id: str):
        def hook(param):
            if not self._should_collect():
                return
            grad = param.grad
            if grad is None:
                return
            self._grad_sq.setdefault(layer_id, []).append(
                grad.detach().float().pow(2).sum()
            )

        return hook

    # -- emit -------------------------------------------------------------- #

    def flush(self, step: Optional[int] = None) -> dict:
        """Emit everything collected for the current step and reset.

        Called automatically from the wrapped ``optimizer.step()``; public
        because a loop that steps by hand (or has no watched optimizer) needs
        to drive it itself. Returns the ``{name: value}`` map it emitted, which
        is also what makes it straightforward to assert on in a test.
        """
        activations, grad_sq = self._activations, self._grad_sq
        self._activations, self._grad_sq = {}, {}

        if self._removed:
            return {}

        want_weights = "weights_norm" in self.metrics
        if not activations and not grad_sq and not want_weights:
            return {}
        # Nothing was collected this step (an unsampled step, or an eval pass):
        # don't compute weight norms either, or they would be the one metric
        # logged at a different cadence than everything else.
        if want_weights and not activations and not grad_sq and self.metrics & (_ACTIVATION_METRICS | {"grad_norm"}):
            if not self._should_collect():
                return {}

        # Build the whole batch as on-device 0-d tensors, then convert ONCE.
        # A per-metric .item() here would be one host<->device sync per layer
        # per step, which is exactly the kind of cost that makes people turn
        # instrumentation off.
        names: list = []
        tensors: list = []

        def add(name: str, value: torch.Tensor) -> None:
            names.append(name)
            tensors.append(value.reshape(()))

        for layer_id, stats in activations.items():
            for metric, value in stats.items():
                add(f"metrics/layer/{layer_id}/{metric}", value)

        global_grad_sq = []
        for layer_id, squares in grad_sq.items():
            total = squares[0] if len(squares) == 1 else torch.stack(squares).sum()
            global_grad_sq.append(total)
            add(f"metrics/layer/{layer_id}/grad_norm", total.sqrt())

        global_weight_sq = []
        if want_weights:
            for layer_id, module in self._layers:
                squares = [
                    p.detach().float().pow(2).sum()
                    for p in module.parameters(recurse=False)
                ]
                if not squares:
                    continue
                total = squares[0] if len(squares) == 1 else torch.stack(squares).sum()
                global_weight_sq.append(total)
                add(f"metrics/layer/{layer_id}/weights_norm", total.sqrt())

        if self.include_global:
            if global_grad_sq:
                add("metrics/global/grad_norm", torch.stack(global_grad_sq).sum().sqrt())
            if global_weight_sq:
                add("metrics/global/weights_norm", torch.stack(global_weight_sq).sum().sqrt())

        if not names:
            return {}

        # The single sync for the whole step.
        values = torch.stack(tensors).cpu().tolist()
        signals = dict(zip(names, values))

        from weightslab.src import save_model_signals

        save_model_signals(signals, step=step)
        return signals

    # -- teardown ---------------------------------------------------------- #

    def remove(self) -> None:
        """Take every hook back off. Idempotent."""
        self._removed = True
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        self._activations.clear()
        self._grad_sq.clear()


def track_model_signals(
    model=None,
    metrics: Iterable[str] = METRICS,
    every_n_steps: int = 1,
    layer_ids: Optional[Iterable] = None,
    include_global: bool = True,
) -> ModelSignalTracker:
    """Instrument a watched model so its training dynamics log themselves.

    Args:
        model: The watched model (what ``watch_or_edit(..., flag="model")``
            returned). Resolved from the ledger when omitted.
        metrics: Which of :data:`METRICS` to emit. Defaults to all of them.
        every_n_steps: Sample every Nth step instead of every step. Activation
            hooks are the only per-step cost worth thinking about on a large
            model; raising this is how you make that cost negligible.
        layer_ids: Restrict to these layer ids (as reported by the model panel
            / ``get_module_id()``). ``None`` tracks every layer.
        include_global: Also emit ``metrics/global/{grad_norm,weights_norm}``,
            the whole-model L2 norms.

    Returns:
        ModelSignalTracker: keep it if you want ``.flush()`` or ``.remove()``;
        ignoring it is fine, the hooks are already installed.

    Example:
        ``model = wl.watch_or_edit(net, flag="model", track_model_signals=True)``
        does this for you. Called directly::

            model = wl.watch_or_edit(net, flag="model", device=device)
            wl.track_model_signals(model, every_n_steps=10)
    """
    if model is None:
        from weightslab.backend.ledgers import get_model

        model = get_model()
    if model is None:
        raise ValueError(
            "No model to track. Call wl.watch_or_edit(model, flag='model') first, "
            "or pass the model explicitly."
        )
    return ModelSignalTracker(
        model,
        metrics=metrics,
        every_n_steps=every_n_steps,
        layer_ids=layer_ids,
        include_global=include_global,
    )
