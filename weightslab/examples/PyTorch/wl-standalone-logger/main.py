"""Standalone LOGGER-AND-SIGNALS integration (level 4 of the four-way SDK).

MNIST again, but the only things wrapped are the **loss and the metric**:

  * no ``flag="model"`` / ``flag="optimizer"`` -> plain ``nn.Module`` + ``Adam``
  * no ``flag="data"``                        -> plain ``DataLoader``
  * no ``flag="hyperparameters"``             -> plain argparse

Each wrapped criterion logs one point per call into the experiment history, so
you get real train/eval curves, a persisted history export, and a CLI/UI report
without touching the other three levels.

Because no model is registered there is no model age to derive the x-axis from,
so this file passes ``step=`` explicitly — that is the only extra argument a
logger-only integration needs.

Scope note: per-sample and per-instance signals (``per_sample=True``,
``wl.save_signals(...)``) route values to sample ids in the sample dataframe,
which only exists once a dataset is tracked. Add the data level
(``flag="data"``) when you want those; step-level curves — what this file shows
— need nothing else.

Run it
------
    python main.py                        # real MNIST (downloads on first run)
    weightslab start example --logger     # same thing, from the installed CLI

Drive it
--------
    weightslab cli                        # attach a terminal to this process
      status                              registered signals/loggers
      report --no-agent                   write the experiment HTML report
      pause / resume                      freeze and unfreeze the loop

    weightslab start                      # open Weights Studio (another terminal)
"""

import argparse
import os
import ssl
import sys
import tempfile
from pathlib import Path

# Windows SSL fix (malformed certs in some Windows stores break the download).
try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    """Plain, unwrapped model — the model level is deliberately absent here."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_loaders(data_root: str, batch_size: int, max_samples: int):
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    eval_ds = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    if max_samples:
        train_ds = Subset(train_ds, range(min(max_samples, len(train_ds))))
        eval_ds = Subset(eval_ds, range(min(max_samples, len(eval_ds))))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(eval_ds, batch_size=batch_size, shuffle=False),
    )


def resolve_log_dir(explicit: str | None, slug: str) -> Path:
    """The experiment directory for this run, created if needed.

    Order: an explicit ``--log-dir``, then ``WEIGHTSLAB_ROOT_LOG_DIR`` (what
    ``weightslab start [DIR]`` exports), then a fresh temporary directory — so the
    bundled example never writes inside the installed package.
    """
    if explicit:
        path = Path(explicit)
    else:
        env_dir = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR")
        path = Path(env_dir) if env_dir else Path(tempfile.mkdtemp(prefix=f"wl-{slug}-"))
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--max-samples", type=int, default=2048,
                   help="Cap on MNIST samples per split (0 = full dataset).")
    p.add_argument("--data-root", default=os.environ.get("WL_DATA_ROOT", "./data"))
    p.add_argument("--log-dir", default=None,
                   help="Experiment directory. Default: $WEIGHTSLAB_ROOT_LOG_DIR, "
                        "else a fresh temporary directory. Exported as "
                        "WEIGHTSLAB_ROOT_LOG_DIR for this process.")
    p.add_argument("--history-format", default="csv", choices=("csv", "json", "parquet"))
    p.add_argument("--no-cli", action="store_true", help="Do not start the CLI server.")
    p.add_argument("--no-grpc", action="store_true", help="Do not start the gRPC backend.")
    p.add_argument("--grpc-port", type=int, default=50051,
                   help="gRPC port for Weights Studio (must match `weightslab start "
                        "--backend-port`).")
    p.add_argument("--serve-timeout", type=int, default=None,
                   help="Seconds to keep serving after training (default: forever).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # The logger level needs an experiment directory for history/report output but
    # NOT the config level: WEIGHTSLAB_ROOT_LOG_DIR is the documented way to give
    # an otherwise unconfigured run one.
    log_dir = resolve_log_dir(args.log_dir, "standalone_logger")
    os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = str(log_dir)

    import weightslab as wl

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, eval_loader = build_loaders(
        args.data_root, args.batch_size, args.max_samples)

    model = SmallCNN().to(device)                       # not wrapped
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # not wrapped

    # --- the only WeightsLab registrations in this file ------------------------
    train_loss = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        signal_name="train/loss",
        log=True,
    )
    eval_loss = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        signal_name="eval/loss",
        log=True,
    )
    eval_acc = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric",
        signal_name="eval/accuracy",
        log=True,
    )
    # --------------------------------------------------------------------------

    wl.serve(serving_grpc=not args.no_grpc, serving_cli=not args.no_cli,
             grpc_port=args.grpc_port)
    print("=" * 70)
    print(" LOGGER-ONLY standalone — attach with `weightslab cli`, UI with `weightslab start`")
    print(f" signals: train/loss, eval/loss, eval/accuracy   log_dir={log_dir}")
    print("=" * 70)
    wl.start_training(timeout=3)

    batches = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            inputs, targets = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            inputs, targets = next(batches)

        with wl.guard_training_context:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # step= is what places this point on the x-axis without a wrapped model.
            per_sample = train_loss(model(inputs), targets, step=step)
            per_sample.mean().backward()
            optimizer.step()

        if step % args.eval_every == 0:
            with wl.guard_testing_context, torch.no_grad():
                eval_inputs, eval_targets = next(iter(eval_loader))
                eval_inputs = eval_inputs.to(device)
                eval_targets = eval_targets.to(device)
                logits = model(eval_inputs)
                eval_loss(logits, eval_targets, step=step)
                eval_acc.update(logits, eval_targets)
                accuracy = eval_acc.compute(step=step)
                eval_acc.reset()
            print(f"[step {step:>5}] train/loss={per_sample.mean().item():.4f} "
                  f"eval/accuracy={float(accuracy) * 100:.1f}%")

    # Persist the signal history; this is also what the report and the UI plots read.
    history = wl.write_history(
        path=str(log_dir / f"history.{args.history_format}"),
        format=args.history_format,
    )
    print(f"[logger-level] history written to {history}")
    print("[logger-level] `report --no-agent` in `weightslab cli` renders it as HTML")

    wl.keep_serving(timeout=args.serve_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
