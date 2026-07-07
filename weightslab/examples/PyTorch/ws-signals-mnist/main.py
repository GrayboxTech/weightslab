"""
WeightsLab per-sample signals — a minimal, idiomatic MNIST example
==================================================================

Instrument an ordinary PyTorch training loop with WeightsLab so that, for
*every training sample*, you capture a small graph of signals — and measure
what that costs.

    10 base signals    loss, confidence, margin, entropy, correctness, ...
       (from logits)   pushed once per step with ``wl.save_signals``

     1 live derived    loss normalized by a running mean — a dynamic @wl.signal
       (@wl.signal)    the framework computes for you on every loss update

     3 end-of-run      from each sample's *full* loss trajectory, computed once
       derived signals after training:
                         - loss_shape : one of six curve shapes, stored as a
                                        text categorical tag (100% coverage)
                         - loss_cv    : coefficient of variation (instability)
                         - loss_drop  : net improvement first->last

The output is a per-sample table mapping every ``sample_id`` to the latest value
of every signal — the raw material for curation: surface the hard samples, the
likely-mislabeled ones (``Flat_high``), and the ones the model forgets
(``U_Shape``).

The idiom
---------
Three ways to declare a signal, and this example shows all three:

  * **Push it yourself** — value comes from something WeightsLab can't see (raw
    logits): compute it in the loop and call ``wl.save_signals(...)``.

  * **Declare a live @wl.signal** — value is a function of a metric WeightsLab
    already logs (the per-sample loss). Write a function of ``ctx``; the
    framework fires it per sample on every loss update and persists the result.

  * **Derive at the end** — value needs a sample's *whole* trajectory. After
    training, read each sample's loss history from the ledger, reduce it, and
    write the result back (as a signal or a categorical tag) for every sample.

Overhead
--------
The run reports the per-step cost of the signal machinery (watched loss +
``save_signals``) measured against a plain-PyTorch baseline on the same loader,
plus the wall-time of the end-of-run derived sweep. All timings are CUDA-synced.

Usage
-----
    python mnist_signals_example.py --outdir ./out
"""
import argparse
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import weightslab as wl
from weightslab.components.global_monitoring import guard_training_context


# ----------------------------------------------------------------------------
# A small MNIST CNN that returns raw logits (no softmax) — so CrossEntropyLoss
# is correct and the logit-derived signals below are meaningful. Kept inline so
# this example is a single self-contained file.
# ----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 1, 28, 28)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Config — small on purpose. A subset + a handful of epochs trains a real model
# and yields a nice spread of per-sample loss-trajectory shapes in ~a minute.
# ----------------------------------------------------------------------------
SEED = 0
SUBSET = 2000      # samples of MNIST train to use (None -> full 60k)
EPOCHS = 20
BATCH = 64
LR = 0.01
MEASURE_STEPS = 50  # steps used for the per-step overhead breakdown

# The ledger flushes to storage every this-many buffered rows. This is the
# single biggest overhead knob: the default (100) flushes every ~1.5 steps at
# batch 64, which dominates the per-step cost. Set it well above the batch size
# so the ledger flushes only a few times per epoch. A value <= batch size is a
# landmine — it flushes mid-batch, every step. (Here: ~batch*128, a handful of
# flushes over the run.)
LEDGER_FLUSH_MAX_ROWS = 8192

# The one metric everything subscribes to. It is the ``signal_name`` of the
# watched loss below, and the ``subscribe_to`` of every live @wl.signal.
LOSS_SIGNAL = "train/loss_sample"


# ----------------------------------------------------------------------------
# 1. Dataset — every sample carries a stable id so its signals can be tracked
#    across epochs. The id is just the dataset index here; in a real pipeline
#    it would be your annotation / image id.
# ----------------------------------------------------------------------------
class MNISTIdx(Dataset):
    """MNIST that returns ``(image, uid, label)`` instead of ``(image, label)``."""

    def __init__(self, root, train, transform, max_samples=None):
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.mnist), self.max_samples)
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        return self.transform(image), idx, label


# ----------------------------------------------------------------------------
# 2. Base signals — pure functions of the batch logits. Nothing
#    WeightsLab-specific here; these just turn logits + labels into per-sample
#    scalars. They need raw logits, so we push them ourselves (below) with
#    ``wl.save_signals`` rather than declaring them as @wl.signal.
# ----------------------------------------------------------------------------
def base_signals(logits, labels, loss_per_sample):
    """10 base per-sample signals derived from raw logits ``[B, C]``."""
    probs = torch.softmax(logits, dim=1)
    top2 = probs.topk(2, dim=1).values
    confidence = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
    correct = (logits.argmax(dim=1) == labels).float()
    true_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    return {
        "sig/loss": loss_per_sample,
        "sig/confidence": confidence,
        "sig/margin": margin,
        "sig/entropy": entropy,
        "sig/correct": correct,
        "sig/true_class_prob": true_prob,
        "sig/nll": -(true_prob + 1e-12).log(),
        "sig/logit_true": logits.gather(1, labels.view(-1, 1)).squeeze(1),
        "sig/logit_max": logits.max(dim=1).values,
        "sig/logit_std": logits.std(dim=1),
    }


# ----------------------------------------------------------------------------
# 3. Live derived signal — a dynamic @wl.signal. It subscribes to the per-sample
#    loss and fires once per sample every time the loss is logged, receiving a
#    SignalContext ``ctx`` whose ``subscribed_value`` is that sample's current
#    loss. It returns one scalar, stored per sample_id. No save_signals, no
#    plumbing — the framework runs it. Here: loss normalized by a running mean
#    ("how hard is this sample right now, relative to the average?").
# ----------------------------------------------------------------------------
_running_mean_loss = {"v": 1.0}


@wl.signal(name="sig/loss_norm", subscribe_to=LOSS_SIGNAL)
def loss_norm(ctx):
    loss = ctx.subscribed_value
    if loss is None:
        return 0.0
    m = _running_mean_loss
    m["v"] = 0.98 * m["v"] + 0.02 * float(loss)
    return float(loss) / (m["v"] + 1e-9)


# ----------------------------------------------------------------------------
# 4. Trajectory shape — the six curve archetypes, and the classifier that
#    reduces a per-sample loss curve to one of them.
# ----------------------------------------------------------------------------
LOSS_SHAPE_LABELS = [
    "monotonic", "plateaued", "Flat_high",
    "high_variance", "U_Shape", "Spiked",
]


def classify_loss_shape(values):
    """Reduce a per-sample loss trajectory (ordered by step) to a shape label.

    Thresholds are scale-invariant (fractions of the trajectory's own range)
    and illustrative — tune them for your own task. Returns None if there is
    not enough history yet (< 5 points).
    """
    y = np.asarray(values, dtype=float)
    if y.size < 5:
        return None

    n = y.size
    first, last = float(y[0]), float(y[-1])
    ymin, ymax = float(y.min()), float(y.max())
    rng = max(ymax - ymin, 1e-8)
    mean = float(y.mean())

    cv = float(y.std()) / (abs(mean) + 1e-8)         # noisiness
    drop = (first - last) / (abs(first) + 1e-8)      # net improvement
    argmin = int(np.argmin(y))
    rebound = (last - ymin) / rng                     # climb-back from trough
    max_up_jump = float(np.diff(y).max()) / rng       # largest single-step rise

    tail = y[int(0.6 * n):]
    tail_flat = (float(tail.std()) / (abs(float(tail.mean())) + 1e-8)) < 0.1

    # Learning shapes first (a big net drop is "being learned", even though a
    # 2.0 -> 0.0 curve has a high coefficient of variation); then the
    # not-learning / noisy shapes.
    if 0.2 * n < argmin < 0.8 * n and rebound > 0.3:
        return "U_Shape"        # learned then forgotten (min in the middle)
    if drop > 0.4:
        return "monotonic"      # steadily learned (large net drop)
    if drop > 0.15 and tail_flat:
        return "plateaued"      # improved, then stuck still-high
    if max_up_jump > 0.5:
        return "Spiked"         # sudden jump, no net learning
    if cv > 0.5:
        return "high_variance"  # noisy oscillation, not converging
    return "Flat_high"          # never moved, stayed high (likely mislabel)


# ----------------------------------------------------------------------------
# 5. End-of-run derived signals — computed once, from each sample's *full* loss
#    trajectory in the ledger. This is the "derive at the end" idiom: one query,
#    then per sample reduce the curve and write the results back by sample_id.
#    Guarantees 100% coverage, unlike the throttled live signal.
# ----------------------------------------------------------------------------
def finalize_derived_signals():
    """Read every sample's loss trajectory and write the definitive derived
    signals: shape (text categorical tag) + loss_cv + loss_drop (scalars)."""
    series = defaultdict(list)
    for sid, step, val, _ in wl.query_signal_history(LOSS_SIGNAL):
        series[sid].append((step, val))

    ids, cv, drop = [], [], []
    by_label = defaultdict(list)
    dist = defaultdict(int)
    for sid, pts in series.items():
        y = np.asarray([v for _, v in sorted(pts)], dtype=float)
        ids.append(sid)
        mean = float(y.mean())
        cv.append(float(y.std()) / (abs(mean) + 1e-8))
        drop.append((float(y[0]) - float(y[-1])) / (abs(float(y[0])) + 1e-8))
        label = classify_loss_shape(y.tolist())
        if label is not None:
            by_label[label].append(sid)
            dist[label] += 1

    # Scalar derived signals: written by stored sample_id (save_signals writes
    # to rows by the literal id — the same id space the ledger already uses).
    wl.save_signals(
        signals={"sig/loss_cv": torch.tensor(cv), "sig/loss_drop": torch.tensor(drop)},
        batch_ids=ids, log=False,
    )
    # Definitive shape as a text categorical tag, for every sample.
    for label, sids in by_label.items():
        wl.set_categorical_tag(sids, "loss_shape", label)
    return dict(dist)


# ----------------------------------------------------------------------------
# Overhead timing helpers
# ----------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_overhead(model, loader, optimizer, criterion, device, n_steps):
    """Per-step cost breakdown (ms/step, CUDA-synced), on the SAME loader.

    For each measured step we time, separately:
      - baseline : a plain forward+backward+step with an unwatched loss (the
                   pure-PyTorch cost — no signals at all)
      - watched  : forward+backward+step with the *watched* loss (adds the
                   per-sample loss logging + any live @wl.signal subscribers)
      - compute  : the base_signals() math (softmax/entropy/... on logits)
      - persist  : the wl.save_signals() call (WeightsLab persistence to the
                   ledger / H5)

    'watched - baseline' isolates the logging cost; 'compute' is the signal math
    (cheap); 'persist' is the storage cost.
    """
    plain = nn.CrossEntropyLoss()
    acc = defaultdict(float)
    it = iter(loader)
    done = 0
    while done < n_steps:
        batch = next(it, None)
        if batch is None:
            it = iter(loader)
            continue
        inputs, ids, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # baseline: plain step (no WeightsLab)
        _sync(device); t0 = time.perf_counter()
        optimizer.zero_grad()
        plain(model(inputs), labels).backward()
        optimizer.step()
        _sync(device); t1 = time.perf_counter()

        # instrumented step, timing each stage
        with guard_training_context:
            optimizer.zero_grad()
            logits = model(inputs)
            preds = logits.argmax(dim=1, keepdim=True)
            loss = criterion(logits, labels, batch_ids=ids, preds=preds)
            loss.mean().backward()
            optimizer.step()
            _sync(device); t2 = time.perf_counter()
            sig = base_signals(logits.detach(), labels, loss.detach())
            _sync(device); t3 = time.perf_counter()
            wl.save_signals(signals=sig, batch_ids=ids,
                            preds_raw=logits.detach(), preds=preds, targets=labels)
            _sync(device); t4 = time.perf_counter()

        acc["baseline"] += t1 - t0
        acc["watched"] += t2 - t1
        acc["compute"] += t3 - t2
        acc["persist"] += t4 - t3
        done += 1
    return {k: 1000.0 * v / max(done, 1) for k, v in acc.items()}


# ----------------------------------------------------------------------------
# Training — wrap the four moving parts, call the watched loss (which logs
# LOSS_SIGNAL and fires the live @wl.signal subscribers), and push the
# logit-derived base signals. Then derive from the trajectories at the end.
# ----------------------------------------------------------------------------
def train(outdir, data_root, device):
    torch.manual_seed(SEED)
    log_dir = os.path.join(outdir, "wl_logs")
    # Start from a clean ledger so the trajectories reflect only this run
    # (a stale ledger would reload old per-sample losses and freeze the curves).
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)

    hyperparams = {
        "experiment_name": "mnist_signals_example",
        "device": str(device),
        "root_log_dir": log_dir,
        "serving_grpc": False,   # headless: no gRPC / CLI control plane
        "serving_cli": False,
        "ledger_flush_max_rows": LEDGER_FLUSH_MAX_ROWS,  # see note above
        # Declare the loader here too, otherwise WeightsLab's config sync
        # resets batch_size to its default. Keys map to loader kwargs.
        "data": {"train_loader": {"batch_size": BATCH, "shuffle": True}},
    }
    wl.watch_or_edit(hyperparams, flag="hyperparameters", defaults=hyperparams)

    dataset = MNISTIdx(data_root, train=True, transform=transforms.ToTensor(),
                       max_samples=SUBSET)

    # Wrap the four moving parts. After this, model/opt/loader behave exactly
    # like their plain PyTorch counterparts — WeightsLab just observes.
    model = wl.watch_or_edit(SmallCNN().to(device), flag="model", device=device)
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=LR), flag="optimizer")
    loader = wl.watch_or_edit(
        dataset, flag="data", loader_name="train_loader",
        batch_size=BATCH, shuffle=True, is_training=True, preload_labels=True,
    )
    # The watched loss: reduction="none" -> one value per sample. Passing
    # batch_ids= on each call logs LOSS_SIGNAL per sample AND drives the live
    # @wl.signal subscribers above. This is the only "signal" we hand-wire.
    criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name=LOSS_SIGNAL, per_sample=True, log=True,
    )

    # Declare the categorical tag up front so the studio shows all six choices;
    # the live signal + the end sweep populate it.
    wl.register_categorical_tag("loss_shape", LOSS_SHAPE_LABELS)

    wl.serve(serving_grpc=False, serving_cli=False)
    wl.start_training(timeout=3)

    # ---- train: watched loss (logs the per-sample loss trajectory + fires the
    #      live @wl.signal) and push the logit-derived base signals -----------
    for epoch in range(1, EPOCHS + 1):
        for inputs, ids, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with guard_training_context:
                optimizer.zero_grad()
                logits = model(inputs)
                preds = logits.argmax(dim=1, keepdim=True)

                # Watched loss: logs LOSS_SIGNAL + fires the live @wl.signal.
                loss_per_sample = criterion(logits, labels, batch_ids=ids, preds=preds)
                loss_per_sample.mean().backward()
                optimizer.step()

                # Logit-derived base signals: push them ourselves.
                wl.save_signals(
                    signals=base_signals(logits.detach(), labels, loss_per_sample.detach()),
                    batch_ids=ids,
                    preds_raw=logits.detach(), preds=preds, targets=labels,
                )
        print(f"epoch {epoch:2d}/{EPOCHS}  loss={loss_per_sample.mean().item():.4f}")

    # ---- overhead: per-step cost breakdown vs plain PyTorch ------------------
    ovh = measure_overhead(model, loader, optimizer, criterion, device, MEASURE_STEPS)

    # ---- derive from full trajectories, once, at the end --------------------
    _sync(device)
    t_fin = time.perf_counter()
    dist = finalize_derived_signals()
    fin_ms = 1000.0 * (time.perf_counter() - t_fin)

    # ---- export: base + derived signals AND the text shape tag --------------
    report_path = os.path.join(outdir, "per_sample_signals.csv")
    report_path = wl.write_dataframe(report_path, format="csv", columns=["signals", "tags"])

    # ---- summary ------------------------------------------------------------
    instrumented = ovh["watched"] + ovh["compute"] + ovh["persist"]
    logging_ms = ovh["watched"] - ovh["baseline"]
    total_over = instrumented - ovh["baseline"]
    print(f"\nDone. Per-sample report: {report_path}")
    print(f"\n── per-step overhead (ms/step, CUDA-synced, {MEASURE_STEPS} steps) ──")
    print(f"  baseline  (plain PyTorch)     : {ovh['baseline']:7.3f}")
    print(f"  + loss logging + subscribers  : {logging_ms:7.3f}")
    print(f"  + signal compute (the math)   : {ovh['compute']:7.3f}   <- signal computation")
    print(f"  + persist (save_signals/H5)   : {ovh['persist']:7.3f}")
    print(f"  = instrumented total          : {instrumented:7.3f}   "
          f"(+{total_over:.3f} ms, {100.0 * total_over / max(ovh['baseline'], 1e-9):+.0f}%)")
    print(f"  end-of-run derived sweep      : {fin_ms:7.1f} ms once ({sum(dist.values())} samples)")
    print(f"\ntrajectory-shape distribution : {dist}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", default="./out", help="where to write logs + report")
    ap.add_argument("--data-root", default="./data", help="MNIST download location")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  subset={SUBSET or 'FULL(60k)'}  epochs={EPOCHS}  batch={BATCH}")
    train(args.outdir, args.data_root, device)


if __name__ == "__main__":
    main()
