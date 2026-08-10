"""Standalone CONFIG-MANAGEMENT integration (level 3 of the four-way SDK).

The experiment configuration alone, with **nothing else wrapped**:

  * no ``flag="model"`` / ``flag="optimizer"``
  * no ``flag="data"``
  * no ``flag="loss"``

``config.yaml`` next to this file is registered as live hyperparameters and
polled, so the same values can be changed three ways while the loop runs — by
editing the YAML, from the ``weightslab cli`` terminal (``set_hp``), or from the
Weights Studio hyperparameters panel — and the loop below sees every change.

The "training loop" here is a stand-in that only reads the config each step:
it prints a line whenever a value changes, which is exactly what a real loop
would consume ``hp["optimizer"]["lr"]`` for.

Run it
------
    python main.py                        # writes config.yaml on first run
    weightslab start example --config     # same thing, from the installed CLI

Drive it
--------
    weightslab cli                        # attach a terminal to this process
      hp                                  list registered config sets
      hp main                             show the whole config
      set_hp optimizer.lr 0.0005          live edit (loop picks it up)
      set_hp data.train_loader.batch_size 64
      status                              current registrations

    weightslab start                      # open Weights Studio (another terminal)

    ...or just edit config.yaml in your editor and save.
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

# Fallback for a copied script whose config.yaml is missing; the shipped
# config.yaml next to this file is otherwise the source of truth.
# No root_log_dir on purpose: it then resolves from WEIGHTSLAB_ROOT_LOG_DIR (what
# `weightslab start [DIR]` exports) and falls back to a temporary directory, so
# running the bundled example never writes inside the installed package.
DEFAULT_CONFIG = {
    "experiment_name": "standalone_config",
    "is_training": True,
    "optimizer": {"lr": 1e-3},
    "data": {"train_loader": {"batch_size": 16, "shuffle": True}},
    "training_steps_to_do": 300,
}

# Config paths the loop below watches for changes.
WATCHED = (
    "optimizer.lr",
    "data.train_loader.batch_size",
    "is_training",
    "training_steps_to_do",
)


def read_path(config, dotted: str, default=None):
    """``read_path(hp, "optimizer.lr")`` -> value, or *default* if absent."""
    node = config
    for key in dotted.split("."):
        try:
            node = node[key]
        except (KeyError, TypeError):
            return default
    return node


def experiment_dir():
    """The directory WeightsLab resolved for this experiment, or ``None``.

    ``root_log_dir`` in the config is the input; the checkpoint manager holds the
    resolved value (which may come from ``WEIGHTSLAB_ROOT_LOG_DIR`` or a temporary
    directory when the config leaves it out).
    """
    from weightslab.backend.ledgers import get_checkpoint_manager

    try:
        return getattr(get_checkpoint_manager(), "root_log_dir", None)
    except Exception:
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(HERE / "config.yaml"),
                   help="YAML config to register and poll.")
    p.add_argument("--steps", type=int, default=0,
                   help="Loop iterations (0 = take training_steps_to_do from the config).")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="Seconds between config reloads.")
    p.add_argument("--step-delay", type=float, default=0.2,
                   help="Seconds per loop iteration (leaves time to edit the config).")
    p.add_argument("--no-cli", action="store_true", help="Do not start the CLI server.")
    p.add_argument("--no-grpc", action="store_true", help="Do not start the gRPC backend.")
    p.add_argument("--grpc-port", type=int, default=50051,
                   help="gRPC port for Weights Studio (must match `weightslab start "
                        "--backend-port`).")
    p.add_argument("--serve-timeout", type=int, default=None,
                   help="Seconds to keep serving after the loop (default: forever).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        # `defaults=` seeds the in-memory config; the watcher only *reads* the
        # file, so write it once here to make it editable from the start.
        config_path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
                               encoding="utf-8")
        print(f"[config-level] wrote {config_path}")

    import weightslab as wl

    # --- the only WeightsLab registration in this file ------------------------
    # str(...) is deliberate: watch_or_edit rebinds the caller's variable to the
    # returned proxy, and passing a fresh string keeps `config_path` intact.
    hp = wl.watch_or_edit(
        str(config_path),
        flag="hyperparameters",
        defaults=copy.deepcopy(DEFAULT_CONFIG),
        poll_interval=args.poll_interval,
    )
    # --------------------------------------------------------------------------

    wl.serve(serving_grpc=not args.no_grpc, serving_cli=not args.no_cli,
             grpc_port=args.grpc_port)
    print("=" * 70)
    print(" CONFIG-ONLY standalone — attach with `weightslab cli`, UI with `weightslab start`")
    print(f" config={config_path}")
    # Each reload makes the FILE authoritative, so a root_log_dir that was only
    # injected at registration does not survive it; ask for the directory in use.
    print(f" experiment dir={experiment_dir()}")
    print(" try: set_hp optimizer.lr 0.0005   (or edit the YAML)")
    print("=" * 70)
    wl.start_training(timeout=3)

    total = args.steps or int(read_path(hp, "training_steps_to_do", 300) or 300)
    last = {key: read_path(hp, key) for key in WATCHED}
    print(f"[config-level] step 0: " + ", ".join(f"{k}={v}" for k, v in last.items()))

    for step in range(1, total + 1):
        # A real loop would use these values (lr on the optimizer, batch size on
        # the loader). Here we only observe them, so the config level stands alone.
        current = {key: read_path(hp, key) for key in WATCHED}
        changed = {k: v for k, v in current.items() if v != last[k]}
        if changed:
            print(f"[config-level] step {step}: changed -> "
                  + ", ".join(f"{k}: {last[k]} -> {v}" for k, v in changed.items()))
            last = current
        if not current.get("is_training", True):
            print(f"[config-level] step {step}: is_training=False — idling")
        time.sleep(args.step_delay)

    print(f"[config-level] loop finished after {total} steps; final config:")
    print(yaml.safe_dump({k: last[k] for k in WATCHED}, sort_keys=False).strip())

    wl.keep_serving(timeout=args.serve_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
