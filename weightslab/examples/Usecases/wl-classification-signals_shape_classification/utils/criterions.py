# =============================================================================
# Custom signal-shape classifier (@wl.signal_classifier)
# =============================================================================
# WeightsLab ships a 6-way default loss-shape classifier (``wl.classify_loss_shape``
# -> monotonic / plateaued / Flat_high / high_variance / U_Shape / Spiked). This
# example OVERRIDES it for our loss signal with a simple BINARY
# ``monotonic`` / ``not_monotonic`` classifier, registered with the
# ``@wl.signal_classifier`` decorator.
#
# Once registered, everything that classifies shapes uses our function for this
# signal: the background auto-tagger, ``wl.write_loss_shapes`` /
# ``wl.write_dataframe(loss_shape_signal=...)``, and ``wl.enable_loss_shape_signal``.
# The result is a categorical ``tag:loss_shape`` column filled with our two
# labels — no extra wiring, and the six-way default is left untouched for every
# other signal.
import weightslab as wl

# A trajectory needs at least this many points before we commit to a label;
# below it we return None so the sample simply stays untagged for now.
MIN_POINTS = 5


@wl.signal_classifier(signal="loss_sample")
def monotonic_or_not(values):
    """Binary loss-shape classifier: ``"monotonic"`` when the loss learned
    (dropped substantially from start to end), else ``"not_monotonic"``.

    Reuses ``wl.trajectory_stats`` — the scale-invariant feature layer the
    built-in classifier is built on — so we read the trend without re-deriving
    it. Returns ``None`` with fewer than ``MIN_POINTS`` points.

    Note the decorator binds this to the ``"loss_sample"`` signal by name. If you
    rename the loss signal, either rename here to match or call
    :func:`register_shape_classifier` with the new name (below).
    """
    s = wl.trajectory_stats(values)
    if s is None or s["n"] < MIN_POINTS:
        return None
    return "monotonic" if s["drop"] > 0.4 else "not_monotonic"


def register_shape_classifier(loss_name):
    """Bind :func:`monotonic_or_not` to *loss_name*, overriding the built-in
    default for just that signal. Handy when the loss signal name comes from
    config and isn't known at import time — call it before training starts."""
    wl.signal_classifier(signal=loss_name)(monotonic_or_not)
    return monotonic_or_not
