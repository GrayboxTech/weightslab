"""Standalone MODEL-INTERACTION integration (level 1 of the four-way SDK).

MNIST + a small CNN. Only the **model** and its **optimizer** are wrapped:

  * no ``flag="data"``           -> plain ``torch.utils.data.DataLoader``
  * no ``flag="hyperparameters"``-> plain argparse + ``WEIGHTSLAB_ROOT_LOG_DIR``
  * no ``flag="loss"``           -> plain ``nn.CrossEntropyLoss``

so this file is a self-contained proof that the model level runs on its own,
with both integration surfaces live (gRPC for Weights Studio, socket for the
``weightslab cli`` terminal).

Run it
------
    python main.py                       # real MNIST (downloads on first run)
    weightslab start example --model     # same thing, from the installed CLI

Drive it
--------
    weightslab cli                       # attach a terminal to this process
      status            registered model/optimizer + model age
      list_models       names in the ledger
      plot_model        architecture of the wrapped model
      pause / resume    freeze and unfreeze the training loop

    weightslab start                     # open Weights Studio (another terminal)

Architecture operations
-----------------------
``--op add|prune|freeze|reset`` applies one runtime architecture operation to
layer ``--op-layer`` after ``--op-at-step`` steps and prints the parameter count
before/after, which is the whole point of the model level.
"""

import argparse
import os
import ssl
import sys
import tempfile
import time
from pathlib import Path

# Windows SSL fix: some Windows cert stores contain malformed ASN1 certs that
# crash ssl.create_default_context() during the torchvision download.
try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Two conv blocks + two linear layers — small enough to prune live."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc2(self.relu3(self.fc1(x)))


# -----------------------------------------------------------------------------
# Plain (un-tracked) MNIST loaders — the data level is deliberately absent here
# -----------------------------------------------------------------------------
def build_loaders(data_root: str, batch_size: int, max_samples: int):
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    if max_samples:
        train_ds = Subset(train_ds, range(min(max_samples, len(train_ds))))
        test_ds = Subset(test_ds, range(min(max_samples, len(test_ds))))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def apply_architecture_op(wl, model, op: str, layer_id: int, neurons: set):
    """Apply one ADD/PRUNE/FREEZE/RESET operation and report the size change.

    FREEZE and RESET keep the layer shapes, so the loop trains straight through
    them — that is why ``--op freeze`` is the default. ADD and PRUNE do resize the
    layer (the printed parameter count proves it), but the autograd graph of this
    already-running loop still refers to the pre-op tensors, so the backward
    passes after them are dropped by the guard. Apply shape-changing ops from
    Weights Studio / the agent, or restart the loop after applying one.
    """
    from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType

    op_type = {
        "add": ArchitectureNeuronsOpType.ADD,
        "prune": ArchitectureNeuronsOpType.PRUNE,
        "freeze": ArchitectureNeuronsOpType.FREEZE,
        "reset": ArchitectureNeuronsOpType.RESET,
    }[op]

    before = parameter_count(model)
    # ADD grows a layer; the negative indices are the "new neuron" convention.
    indices = {-1} if op == "add" else set(neurons)
    model.apply_architecture_op(op_type, layer_id, indices)
    after = parameter_count(model)
    print(f"[model-level] {op.upper()} layer={layer_id} neurons={sorted(indices)}: "
          f"parameters {before} -> {after}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
    p.add_argument("--steps", type=int, default=200,
                   help="Training steps to run (0 = only serve).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=2048,
                   help="Cap on MNIST samples per split (0 = full dataset).")
    p.add_argument("--data-root", default=os.environ.get("WL_DATA_ROOT", "./data"))
    p.add_argument("--log-dir", default=None,
                   help="Experiment directory. Default: $WEIGHTSLAB_ROOT_LOG_DIR, "
                        "else a fresh temporary directory. Exported as "
                        "WEIGHTSLAB_ROOT_LOG_DIR for this process.")
    p.add_argument("--op", choices=("add", "prune", "freeze", "reset"), default="freeze",
                   help="Architecture operation to demonstrate (see --no-op to skip). "
                        "freeze/reset keep training going; add/prune change layer "
                        "shapes — see the note in apply_architecture_op().")
    p.add_argument("--op-layer", type=int, default=0, help="Layer id to operate on.")
    p.add_argument("--op-neurons", type=int, nargs="*", default=[1, 3],
                   help="Neuron indices for prune/freeze/reset.")
    p.add_argument("--op-at-step", type=int, default=100,
                   help="Step at which the architecture operation is applied.")
    p.add_argument("--no-op", action="store_true", help="Skip architecture ops.")
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

    # The model level needs an experiment directory but NOT the config level:
    # WEIGHTSLAB_ROOT_LOG_DIR is the documented way to point an otherwise
    # unconfigured run at one (see wl.serve / root_log_dir resolution order).
    log_dir = resolve_log_dir(args.log_dir, "standalone_model")
    os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = str(log_dir)

    import weightslab as wl

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = build_loaders(
        args.data_root, args.batch_size, args.max_samples)
    dummy_input = torch.zeros(1, 1, 28, 28, device=device)

    # --- the only two WeightsLab registrations in this file --------------------
    model = wl.watch_or_edit(
        SmallCNN().to(device),
        flag="model",
        device=device,
        dummy_input=dummy_input,        # traces the graph
        compute_dependencies=True,      # required by architecture operations
        use_onnx=False,                 # TorchFX path
    )
    optimizer = wl.watch_or_edit(
        optim.Adam(model.parameters(), lr=args.lr),
        flag="optimizer",
    )
    # --------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()   # plain: the logger level is not used here

    wl.serve(serving_grpc=not args.no_grpc, serving_cli=not args.no_cli,
             grpc_port=args.grpc_port)
    print("=" * 70)
    print(" MODEL-ONLY standalone — attach with `weightslab cli`, UI with `weightslab start`")
    print(f" device={device}  parameters={parameter_count(model)}  log_dir={log_dir}")
    print("=" * 70)
    wl.start_training(timeout=3)

    batches = iter(train_loader)
    started = time.time()
    for step in range(args.steps):
        try:
            inputs, targets = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            inputs, targets = next(batches)

        # guard_training_context is what advances the model age and marks these
        # forward/backward passes as *training* for everything watching.
        with wl.guard_training_context:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        if step and step % 50 == 0:
            with wl.guard_testing_context, torch.no_grad():
                eval_inputs, eval_targets = next(iter(test_loader))
                eval_inputs = eval_inputs.to(device)
                eval_targets = eval_targets.to(device)
                preds = model(eval_inputs).argmax(dim=1)
                acc = (preds == eval_targets).float().mean().item() * 100
            print(f"[step {step:>5}] train_loss={loss.item():.4f} eval_acc={acc:.1f}% "
                  f"age={getattr(model, 'get_age', lambda: step)()}")

        if not args.no_op and step == args.op_at_step:
            apply_architecture_op(
                wl, model, args.op, args.op_layer, set(args.op_neurons))
            if args.op in ("add", "prune"):
                # A shape-changing op replaces layer tensors, so the optimizer
                # must be rebuilt over the new parameters.
                optimizer = wl.watch_or_edit(
                    optim.Adam(model.parameters(), lr=args.lr), flag="optimizer")

    print(f"[model-level] {args.steps} steps in {time.time() - started:.1f}s; "
          f"final parameters={parameter_count(model)}")

    # The model level logs its own signals (model/grad_norm, model/parameters) once
    # per guarded training step, so there is real history to export and plot even
    # though no loss or metric is wrapped here.
    history = wl.write_history(path=str(log_dir / "history.csv"), format="csv")
    print(f"[model-level] history written to {history}")

    # Services stay up so the CLI/UI can still inspect the model after training.
    wl.keep_serving(timeout=args.serve_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
