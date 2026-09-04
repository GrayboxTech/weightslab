"""Standalone DATA-EXPLORATION integration (level 2 of the four-way SDK).

MNIST curation with **no model at all**. Only the datasets are wrapped:

  * no ``flag="model"`` / ``flag="optimizer"`` -> nothing is trained
  * no ``flag="hyperparameters"``              -> plain argparse
  * no ``flag="loss"``                         -> no signals are logged

What it does is a pure data pass: read every batch through the tracked loaders,
tag samples by digit class, discard a deliberately "corrupted" slice, query the
results back through the SDK, and export the subset to CSV. Everything it does
from Python you can also do live from the CLI or Weights Studio.

Run it
------
    python main.py                      # real MNIST (downloads on first run)
    weightslab start example --data     # same thing, from the installed CLI

Drive it
--------
    weightslab cli                      # attach a terminal to this process
      list_loaders                      registered splits
      list_uids train_loader --limit 5  sample ids + tags + discard state
      add_tag 7 hard_examples           tag sample 7
      discard 7                         exclude sample 7 from training batches
      list_uids train_loader --discarded
      dump                              sanitized ledger snapshot

    weightslab start                    # open Weights Studio (another terminal)
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
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MnistSlice(Dataset):
    """``(image, label)`` MNIST slice.

    A plain 2-tuple dataset is all the data wrapper needs: it injects the stable
    sample id itself, so the tracked loader yields ``(images, ids, labels)``.
    """

    def __init__(self, root: str, train: bool, max_samples: int = 0):
        self.mnist = datasets.MNIST(root, train=train, download=True,
                                    transform=transforms.ToTensor())
        self.length = min(max_samples, len(self.mnist)) if max_samples else len(self.mnist)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, label = self.mnist[index]
        return image, label


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
    p.add_argument("--epochs", type=int, default=1,
                   help="Curation passes over the tracked train loader.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=512,
                   help="Cap on MNIST samples per split (0 = full dataset).")
    p.add_argument("--tag-digit", type=int, default=8,
                   help="Digit class to tag as 'hard_examples'.")
    p.add_argument("--tag-name", default="hard_examples")
    p.add_argument("--discard-first", type=int, default=5,
                   help="How many leading train samples to discard.")
    p.add_argument("--data-root", default=os.environ.get("WL_DATA_ROOT", "./data"))
    p.add_argument("--log-dir", default=None,
                   help="Experiment directory. Default: $WEIGHTSLAB_ROOT_LOG_DIR, "
                        "else a fresh temporary directory. Exported as "
                        "WEIGHTSLAB_ROOT_LOG_DIR for this process.")
    p.add_argument("--no-cli", action="store_true", help="Do not start the CLI server.")
    p.add_argument("--no-grpc", action="store_true", help="Do not start the gRPC backend.")
    p.add_argument("--grpc-port", type=int, default=50051,
                   help="gRPC port for Weights Studio (must match `weightslab start "
                        "--backend-port`).")
    p.add_argument("--serve-timeout", type=int, default=None,
                   help="Seconds to keep serving after the pass (default: forever).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # The data level needs an experiment directory (h5 sample store, exports) but
    # NOT the config level: WEIGHTSLAB_ROOT_LOG_DIR is the documented way to give
    # an otherwise unconfigured run one.
    log_dir = resolve_log_dir(args.log_dir, "standalone_data")
    os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = str(log_dir)

    import weightslab as wl

    # --- the only WeightsLab registrations in this file -----------------------
    train_loader = wl.watch_or_edit(
        MnistSlice(args.data_root, train=True, max_samples=args.max_samples),
        flag="data",
        loader_name="train_loader",
        batch_size=args.batch_size,
        shuffle=True,
        is_training=True,        # deny-aware sampling: discarded ids leave the batches
        compute_hash=False,      # True = content hashes, stable across runs but slower
        preload_labels=True,
        preload_metadata=False,
        root_log_dir=str(log_dir),
    )
    val_loader = wl.watch_or_edit(
        MnistSlice(args.data_root, train=False, max_samples=args.max_samples),
        flag="data",
        loader_name="val_loader",
        batch_size=args.batch_size,
        shuffle=False,
        is_training=False,
        compute_hash=False,
        preload_labels=True,
        preload_metadata=False,
        root_log_dir=str(log_dir),
    )
    # --------------------------------------------------------------------------

    wl.serve(serving_grpc=not args.no_grpc, serving_cli=not args.no_cli,
             grpc_port=args.grpc_port)
    print("=" * 70)
    print(" DATA-ONLY standalone — attach with `weightslab cli`, UI with `weightslab start`")
    print(f" train={len(train_loader.dataset)} val={len(val_loader.dataset)} log_dir={log_dir}")
    print("=" * 70)
    wl.start_training(timeout=3)

    # 1) Curation pass: the tracked loader yields (inputs, ids, targets).
    to_tag: list = []
    seen = 0
    for _ in range(args.epochs):
        for inputs, ids, targets in train_loader:
            seen += len(ids)
            hits = (targets.view(-1) == args.tag_digit).nonzero().view(-1).tolist()
            to_tag += [int(ids[i]) for i in hits]

    # 2) Tag / discard through the SDK (identical to what the UI grid does).
    wl.tag_samples(to_tag, args.tag_name, mode="add")

    # `wrapped_dataset` is the tracking wrapper (loader.dataset is the raw one);
    # `unique_ids` are the stable ids WeightsLab assigned to every sample.
    tracked = train_loader.wrapped_dataset
    first_ids = [int(i) for i in tracked.unique_ids[: args.discard_first]]
    wl.discard_samples(first_ids, discarded=True)

    # 3) Query the curation state back.
    tagged = wl.get_samples_by_tag(args.tag_name, origin="train_loader")
    discarded = wl.get_discarded_samples(origin="train_loader")
    print(f"[data-level] visited {seen} samples")
    print(f"[data-level] tagged '{args.tag_name}': {len(tagged)} -> {tagged[:10]}")
    print(f"[data-level] discarded: {len(discarded)} -> {discarded[:10]}")

    # 4) A discarded id no longer shows up in training batches (is_training=True).
    remaining = set()
    for _, ids, _ in train_loader:
        remaining.update(int(i) for i in ids)
    leaked = remaining & set(discarded)
    print(f"[data-level] discarded ids still sampled: {len(leaked)} (expected 0)")

    # 5) Export the curated subset.
    export = log_dir / "curated_samples.csv"
    written = wl.write_dataframe(
        path=str(export),
        format="csv",
        columns=["discarded", f"tag:{args.tag_name}"],
    )
    print(f"[data-level] exported {written}")

    wl.keep_serving(timeout=args.serve_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
