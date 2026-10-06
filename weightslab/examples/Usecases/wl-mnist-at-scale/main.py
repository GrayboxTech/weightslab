"""MNIST at scale, with no training: a simulated run to browse in Weights Studio.

Builds a dataset of ``--samples`` samples (1M by default, split 80/10/10 into
train/val/test) out of MNIST's 70,000 images -- each sample is one of them with
its own small, deterministic shift and contrast, so neighbouring samples do not
look identical -- and fills in what a real training run would have left behind,
without training anything:

  * the run itself: ``--steps`` steps (250k) at batch size ``--batch-size`` (4),
    i.e. 1M sample visits, 1.25 epochs of the train split. Curves for
    ``train-loss`` / ``train-acc`` at every step (each step a real batch of 4
    train samples, so the batch accuracy moves in quarters, as it would), and
    ``val-loss`` / ``val-acc`` / ``test-loss`` / ``test-acc`` every
    ``--eval-every`` steps over the whole split -- with the val/test loss
    turning back up late in the run, a little overfitting to look at;
  * per sample: the loss of every training visit (its trajectory), when it was
    last seen and how often; the final model's prediction, loss and confidence
    (~97% accurate on val/test, making MNIST's real mistakes: 4<->9, 3<->5,
    7<->1, ...);
  * a 3-D projection (``signals//umap_{x,y,z}``): one cluster per class, a few
    sub-clusters per class (writing styles), misclassified samples pulled toward
    the class they were mistaken for;
  * two tags: ``hard`` (the highest final losses) and ``label_noise``
    (confident and wrong).

Every number is derived from one per-sample difficulty, so the curves end
exactly on what the grid and the projection show: the last val-acc point is
the share of val samples the grid shows as correctly predicted.

Then it serves the Studio and waits. Sample statistics are not written to H5,
so the run starts the same every time.

Run::

    python main.py --mnist-root <dir holding MNIST/raw>      # 1M samples
    weightslab start                    # another terminal: open the Studio

Memory, measured at 1M samples with the Studio connected (grid, projection,
lasso, selection applied): 2.4 GB committed, about 2.3 GB per million samples
on top of ~0.5 GB of libraries. The ledger keeps several Python objects per
sample and the data board builds its own copy of the table. So 10M samples
needs ~24 GB of free RAM -- a 32 GB machine with little else open -- while a
16 GB laptop holds 2-4M depending on what else is running. Startup warns when
the request will not fit, and prints the resident memory after each stage.
Registration takes ~40 s per million samples.
"""

import argparse
import os
import ssl
import sys
import tempfile
import time

try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# MNIST's classic confusions: for each class, the classes it is mistaken for
# and how often, relative to each other.
CONFUSIONS = {
    0: {6: 3, 2: 1, 8: 1}, 1: {7: 4, 2: 1, 8: 1}, 2: {7: 3, 3: 2, 8: 1},
    3: {5: 4, 8: 2, 2: 1}, 4: {9: 6, 6: 1, 1: 1}, 5: {3: 4, 6: 2, 8: 1},
    6: {0: 3, 5: 2, 4: 1}, 7: {1: 3, 9: 3, 2: 2}, 8: {3: 3, 5: 2, 9: 2},
    9: {4: 6, 7: 2, 8: 1},
}
STYLES = 3                    # sub-clusters per class in the projection
CHANCE_LOSS = float(np.log(10))
GB_PER_MILLION = 2.3          # measured with the Studio connected (see above)
BASE_GB = 0.5


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return float("nan")


def stage(name: str, started: float) -> float:
    now = time.perf_counter()
    print(f"[mnist-at-scale] {name:<46} {now - started:7.1f} s   RSS {rss_gb():5.1f} GB",
          flush=True)
    return now


def warn_if_too_big(samples: int) -> None:
    """Say so up front when the run will not fit in free memory: past that
    point Windows pages the ledger out, and everything slows to a crawl."""
    try:
        import psutil
        available = psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return
    needed = BASE_GB + GB_PER_MILLION * samples / 1e6
    if needed > available:
        fits = max(0.0, (available - BASE_GB) / GB_PER_MILLION)
        print(f"[mnist-at-scale] WARNING: {samples:,} samples need ~{needed:.1f} GB, "
              f"{available:.1f} GB is free -- about {fits:.1f}M samples fit right now. "
              f"Expect paging (slow) until other applications free memory.", flush=True)


def load_mnist(root: str):
    """All 70,000 MNIST images (uint8, [70000, 28, 28]) and labels."""
    from torchvision import datasets
    images, labels = [], []
    for train in (True, False):
        split = datasets.MNIST(root, train=train, download=True)
        images.append(split.data.numpy())
        labels.append(split.targets.numpy())
    return np.concatenate(images), np.concatenate(labels).astype(np.int64)


def mix(x: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic per-sample pseudo-random bits (splitmix64)."""
    x = (np.asarray(x).astype(np.uint64) + np.uint64(salt)) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    return x


def unit(bits: np.ndarray) -> np.ndarray:
    """Pseudo-random bits -> uniform floats in [0, 1)."""
    return (bits >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class ScaledMNIST(Dataset):
    """``count`` samples, sample ``i`` being MNIST image ``(start + i) % 70000``
    shifted by up to 2 px and rescaled in contrast -- decided by its index, so
    the same sample always looks the same.

    ``fast_get_label`` is what lets the ledger register millions of samples
    without decoding a single image.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, start: int, count: int):
        self.images, self.labels = images, labels
        self.start, self.count = start, count

    def __len__(self):
        return self.count

    def _base(self, idx: int) -> int:
        return (self.start + idx) % len(self.images)

    def __getitem__(self, idx):
        idx = int(idx)
        bits = int(mix(np.array([self.start + idx]), 1)[0])
        dx, dy = bits % 5 - 2, (bits >> 8) % 5 - 2
        contrast = 0.75 + ((bits >> 16) % 64) / 256.0
        image = np.roll(self.images[self._base(idx)], (dy, dx), axis=(0, 1))
        tensor = torch.from_numpy(image.astype(np.float32) * (contrast / 255.0)).unsqueeze(0)
        return tensor, int(self.labels[self._base(idx)])

    def fast_get_label(self, idx):
        return None, None, int(self.labels[self._base(int(idx))])


# =============================================================================
# The model, simulated: one difficulty per sample drives everything
# =============================================================================
def final_state(global_index: np.ndarray, base: np.ndarray, target: np.ndarray,
                seed: int, training: bool) -> dict:
    """The trained model's view of these samples: prediction, loss,
    confidence, place in the projection -- and the difficulty behind them,
    which the training curves are built from too."""
    rng = np.random.default_rng(seed)

    # Difficulty: mostly easy, a long tail. Partly the image itself (some MNIST
    # digits are genuinely ambiguous), partly this sample's distortion. The
    # model fits its training samples better than it generalises.
    image_hardness = unit(mix(base, 7)) ** 6
    distortion = unit(mix(global_index, 11)) ** 3
    difficulty = np.clip(0.75 * image_hardness + 0.35 * distortion, 0, 1)
    p_wrong = (0.0006 + 0.08 * difficulty ** 2) if training else (0.002 + 0.22 * difficulty ** 2)
    wrong = unit(mix(global_index, 13)) < p_wrong

    prediction = target.copy()
    pick = unit(mix(global_index, 17))
    for cls, confused in CONFUSIONS.items():
        rows = np.flatnonzero(wrong & (target == cls))
        if not len(rows):
            continue
        choices = np.array(list(confused.keys()))
        weights = np.cumsum(np.array(list(confused.values()), dtype=float))
        prediction[rows] = choices[np.searchsorted(weights / weights[-1], pick[rows])]

    jitter = unit(mix(global_index, 19))
    confidence = np.where(wrong, 0.45 + 0.5 * jitter, 0.9995 - 0.5 * difficulty ** 2 * jitter)
    p_true = np.where(wrong, (1 - confidence) * (0.3 + 0.6 * jitter), confidence)
    loss = -np.log(np.clip(p_true, 1e-6, 1.0))

    # Projection: classes on a sphere, writing styles around each class,
    # distortion spreading a sample out, mistakes pulled toward the class they
    # were mistaken for. Same seed for every split: one embedding space.
    angle = rng.uniform(0, 2 * np.pi, (10, 2))
    centres = 12.0 * np.stack([np.sin(angle[:, 0]) * np.cos(angle[:, 1]),
                               np.sin(angle[:, 0]) * np.sin(angle[:, 1]),
                               np.cos(angle[:, 0])], axis=1)
    styles = rng.normal(0, 2.2, (10, STYLES, 3))
    style = (mix(base, 23) % np.uint64(STYLES)).astype(np.int64)
    noise = np.stack([unit(mix(global_index, s)) for s in (29, 31, 37)], axis=1)
    spread = 0.9 + 1.8 * difficulty[:, None]
    xyz = centres[target] + styles[target, style] + (noise - 0.5) * 2.0 * spread
    pull = np.where(wrong, 0.35 + 0.5 * jitter, 0.0)[:, None]
    xyz = xyz + pull * (centres[prediction] - centres[target])

    return {"difficulty": difficulty, "wrong": wrong, "prediction": prediction,
            "loss": loss, "confidence": confidence, "xyz": xyz}


def untrained(step: np.ndarray, steps: int) -> np.ndarray:
    """How far from trained the model is at *step*: 1 at the start, exactly 0
    at the last step, so the final evaluation IS the final per-sample state.
    A fast early drop, then a long slow tail."""
    step = np.asarray(step, dtype=np.float64)
    tail = lambda s: 1.0 / (1.0 + s / 15_000.0)
    slow = (tail(step) - tail(steps)) / (1.0 - tail(steps))
    return np.clip(0.7 * np.exp(-step / 4_000.0) * (1 - step / steps) + 0.3 * slow, 0.0, 1.0)


def overfit(step: np.ndarray, steps: int) -> np.ndarray:
    """Held-out loss creeping back up over the last 40% of the run."""
    s = np.asarray(step, dtype=np.float64)
    return np.clip((s - 0.6 * steps) / (0.4 * steps), 0.0, 1.0) ** 2


def loss_at(state: dict, g, late) -> np.ndarray:
    """Expected per-sample loss when the model is *g* from trained."""
    start = CHANCE_LOSS * (0.7 + 0.6 * state["difficulty"])
    held_out = 0.04 * late * (0.3 + 2.0 * state["difficulty"] + 3.0 * state["wrong"])
    return state["loss"] + g * (start - state["loss"]) + held_out


def p_correct_at(state: dict, g) -> np.ndarray:
    """Chance the model gets the sample right when it is *g* from trained."""
    return (~state["wrong"]) * (1 - g) + 0.1 * g


def simulate_run(train: dict, held_out: dict, steps: int, batch_size: int,
                 eval_every: int, seed: int) -> dict:
    """The curves of the run, and every training visit of every sample."""
    rng = np.random.default_rng(seed + 1)
    n_train = len(train["difficulty"])
    visits = steps * batch_size

    # Shuffled epochs over the train split, cut at the last step.
    order = np.concatenate([rng.permutation(n_train)
                            for _ in range(-(-visits // n_train))])[:visits]
    step_of = np.arange(visits) // batch_size + 1
    g = untrained(step_of, steps)
    expected = loss_at({k: v[order] for k, v in train.items()}, g, 0.0)
    visit_loss = expected * rng.lognormal(-0.06, 0.35, visits)
    visit_right = rng.random(visits) < p_correct_at({"wrong": train["wrong"][order]}, g)

    per_step_loss = visit_loss.reshape(steps, batch_size)
    curves = {
        "train-loss": (np.arange(1, steps + 1), per_step_loss.mean(axis=1),
                       per_step_loss.min(axis=1), per_step_loss.max(axis=1)),
        "train-acc": (np.arange(1, steps + 1),
                      visit_right.reshape(steps, batch_size).mean(axis=1), None, None),
    }
    eval_steps = np.arange(eval_every, steps + 1, eval_every)
    if eval_steps[-1] != steps:
        eval_steps = np.append(eval_steps, steps)
    for split, state in held_out.items():
        losses, accs = [], []
        for s in eval_steps:
            g_s, late = float(untrained(s, steps)), float(overfit(s, steps))
            losses.append(float(loss_at(state, g_s, late).mean()))
            accs.append(float(p_correct_at(state, g_s).mean()))
        # Evaluation noise, none at the last point: it must read exactly what
        # the grid shows for the final model.
        jitter = 1 + rng.normal(0, 0.004, len(eval_steps))
        jitter[-1] = 1.0
        curves[f"{split}-loss"] = (eval_steps, np.array(losses) * jitter, None, None)
        curves[f"{split}-acc"] = (eval_steps, np.clip(np.array(accs) * jitter, 0, 1), None, None)
    return {"curves": curves, "order": order, "step_of": step_of, "visit_loss": visit_loss}


def log_curves(curves: dict, sample_rows: tuple, run_seconds: float, batch_size: int) -> str:
    """Write the run into the Studio's signal history, as training would have.

    Rows go through the logger's own staging and flush -- the path every
    logged point takes -- but not one ``add_scalars`` call per point: that also
    queues each point for live streaming, half a million entries no browser
    is waiting for.
    """
    from weightslab.backend.ledgers import get_logger
    logger = get_logger()
    manager = getattr(logger, "chkpt_manager", None)
    exp_hash = manager.get_current_experiment_hash() if manager else None
    now = int(time.time())
    total = max(int(c[0][-1]) for c in curves.values())
    with logger._lock:
        for name, (steps, values, low, high) in curves.items():
            logger.graph_names.add(name)
            stamps = now - (run_seconds * (1 - steps / total)).astype(np.int64)
            for i in range(len(steps)):
                logger._stage_signal_row(
                    name, exp_hash, int(steps[i]), float(values[i]), int(stamps[i]),
                    False, False, "", [], "",
                    sample_count=batch_size if low is not None else 0,
                    value_min=None if low is None else float(low[i]),
                    value_max=None if high is None else float(high[i]))
        logger._flush_stage()
    name, sample_ids, steps, values = sample_rows
    logger.ingest_per_sample(name, exp_hash, zip(sample_ids, steps, values))
    logger._flush_stage()
    return exp_hash


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", type=int, default=1_000_000)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=250_000, help="Simulated training steps.")
    p.add_argument("--batch-size", type=int, default=4, help="Simulated training batch size.")
    p.add_argument("--eval-every", type=int, default=2_500,
                   help="Steps between two (simulated) val/test evaluations.")
    p.add_argument("--mnist-root", default=os.environ.get("WL_DATA_ROOT", "./data"),
                   help="Directory holding MNIST/raw (downloaded there if missing).")
    p.add_argument("--chunk", type=int, default=1_000_000,
                   help="Samples written to the ledger per batch (bounds peak memory).")
    p.add_argument("--hard-share", type=float, default=0.01,
                   help="Share of samples tagged 'hard' (the highest final losses).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--grpc-port", type=int, default=50051)
    p.add_argument("--no-serve", action="store_true",
                   help="Build everything, report, and exit (for measuring).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    warn_if_too_big(args.samples)
    started = time.perf_counter()
    log_dir = args.log_dir or tempfile.mkdtemp(prefix="wl-mnist-at-scale-")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = log_dir

    import weightslab as wl
    from weightslab.backend.ledgers import get_dataframe
    from weightslab.projection.registry import register_prefix

    # The run's settings, shown in the Studio's hyperparameters panel. H5 off:
    # a million rows to disk on every flush would be most of the startup, for
    # a run that never needs reloading.
    hp = {"experiment_name": "mnist-at-scale (simulated)", "root_log_dir": log_dir,
          "ledger_enable_h5_persistence": False, "ledger_flush_max_rows": 1_000_000,
          "batch_size": args.batch_size, "training_steps": args.steps,
          "eval_every": args.eval_every, "optimizer": "adam", "learning_rate": 1e-3,
          "architecture": "small CNN (simulated -- nothing is trained)",
          # Where the Studio's hyperparameters panel reads each loader's batch size.
          "data": {"train_loader": {"batch_size": args.batch_size},
                   "val_loader": {"batch_size": 256}, "test_loader": {"batch_size": 256}}}
    wl.watch_or_edit(hp, flag="hyperparameters", defaults=hp)

    images, labels = load_mnist(args.mnist_root)
    t = stage(f"MNIST loaded ({len(images):,} images)", started)

    n_val = int(round(args.samples * args.val_fraction))
    n_test = int(round(args.samples * args.test_fraction))
    n_train = args.samples - n_val - n_test
    splits = [("train_loader", 0, n_train, True),
              ("val_loader", n_train, n_val, False),
              ("test_loader", n_train + n_val, n_test, False)]
    loaders, states = {}, {}
    for name, start, count, training in splits:
        if count <= 0:
            continue
        loaders[name] = wl.watch_or_edit(
            ScaledMNIST(images, labels, start, count),
            flag="data", loader_name=name,
            batch_size=args.batch_size if training else 256, shuffle=training,
            is_training=training, compute_hash=False,
            preload_labels=True, preload_metadata=False, root_log_dir=log_dir,
        )
        global_index = np.arange(start, start + count)
        base = global_index % len(images)
        states[name] = final_state(global_index, base, labels[base], args.seed, training)
        t = stage(f"registered {name} ({count:,} samples)", t)

    # --- the run --------------------------------------------------------------
    run = simulate_run(states["train_loader"],
                       {"val": states["val_loader"], "test": states["test_loader"]},
                       args.steps, args.batch_size, args.eval_every, args.seed)
    train_ids = np.asarray(loaders["train_loader"].wrapped_dataset.unique_ids)
    exp_hash = log_curves(
        run["curves"],
        ("train-loss", train_ids[run["order"]], run["step_of"], run["visit_loss"]),
        run_seconds=args.steps * 0.05, batch_size=args.batch_size)
    t = stage(f"run logged ({args.steps:,} steps, batch {args.batch_size})", t)

    # Per training sample: its last visit, how many, and that visit's loss.
    order, step_of = run["order"], run["step_of"]
    last_seen = np.full(n_train, -1, dtype=np.int64)
    last_seen[order] = step_of                     # later visits overwrite earlier
    nb_seen = np.bincount(order, minlength=n_train)
    last_loss = np.full(n_train, np.nan, dtype=np.float32)
    last_loss[order] = run["visit_loss"]

    # --- per-sample state -------------------------------------------------------
    ledger = get_dataframe()
    ledger.register_boolean_tag("label_noise")
    ledger.register_boolean_tag("hard")
    all_loss = np.concatenate([states[name]["loss"] for name in states])
    hard_cut = float(np.quantile(all_loss, 1 - args.hard_share))
    for name, start, count, training in splits:
        if name not in loaders:
            continue
        state, ids = states[name], np.asarray(loaders[name].wrapped_dataset.unique_ids)
        late = overfit(args.steps, args.steps) if not training else 0.0
        final_loss = loss_at(state, 0.0, late)
        for lo in range(0, count, args.chunk):
            hi = min(count, lo + args.chunk)
            part = slice(lo, hi)
            columns = {
                "prediction": state["prediction"][part],
                "signals//loss": final_loss[part].astype(np.float32),
                "signals//confidence": state["confidence"][part].astype(np.float32),
                "signals//umap_x": state["xyz"][part, 0].astype(np.float32),
                "signals//umap_y": state["xyz"][part, 1].astype(np.float32),
                "signals//umap_z": state["xyz"][part, 2].astype(np.float32),
                "tag:label_noise": state["wrong"][part] & (state["confidence"][part] > 0.9),
                "tag:hard": final_loss[part] >= hard_cut,
            }
            if training:
                columns["signals//train-loss"] = last_loss[part]
                columns["last_seen"] = last_seen[part]
                columns["nb_seen"] = nb_seen[part]
            frame = pd.DataFrame(columns, index=pd.MultiIndex.from_arrays(
                [ids[lo:hi], np.zeros(hi - lo, dtype=np.int64)],
                names=["sample_id", "annotation_id"]))
            ledger.upsert_df(frame, origin=name)
            del frame, columns
        t = stage(f"predictions + projection written ({name})", t)

    # Tell the Projection board "umap" is a projection, not just three columns.
    register_prefix("umap", root_log_dir=log_dir)

    curves = run["curves"]
    summary = ", ".join(f"{k} {curves[k][1][-1]:.3f}" for k in
                        ("train-loss", "val-loss", "val-acc", "test-loss", "test-acc"))
    total = sum(len(l.wrapped_dataset) for l in loaders.values())
    print(f"[mnist-at-scale] {total:,} samples ready in {time.perf_counter() - started:.0f} s "
          f"(RSS {rss_gb():.1f} GB). Run {exp_hash}: {summary}.", flush=True)
    print(f"[mnist-at-scale] log dir: {log_dir}", flush=True)
    if args.no_serve:
        return 0

    wl.serve(serving_grpc=True, serving_cli=True, grpc_port=args.grpc_port)
    print("[mnist-at-scale] serving -- open the Studio with `weightslab start` "
          f"(backend port {args.grpc_port}). Ctrl+C to stop.", flush=True)
    wl.keep_serving()
    return 0


if __name__ == "__main__":
    sys.exit(main())
