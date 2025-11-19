"""Example demonstrating `weightslab.ledgers` usage and threaded updates.

Run this example from the repo root (with editable package installed or
`sys.path` patched) to see a main thread and a background thread share an
optimizer wrapper object through the ledger.
"""
import threading
import time

from weightslab.ledgers import register_optimizer, get_optimizer
from weightslab.backend.optimizer_interface import OptimizerInterface


class DummyOpt:
    def __init__(self, lr=0.1):
        self.lr = lr

    def get_lr(self):
        return [self.lr]

    def set_lr(self, lr):
        self.lr = lr


def updater(name: str):
    opt = get_optimizer(name)
    print("[updater] initial lr:", opt.get_lr())
    time.sleep(1.0)
    opt.set_lr(1e-4)
    print("[updater] set lr to", opt.get_lr())


def main():
    # Build a small wrapper around a dummy optimizer
    dummy = DummyOpt(lr=0.01)
    wrapper = OptimizerInterface(dummy)  # accepts instances

    # Register under a conventional name
    register_optimizer("_optimizer", wrapper)

    t = threading.Thread(target=updater, args=("_optimizer",), daemon=True)
    t.start()

    # Main thread inspects the optimizer while background thread modifies it
    main_opt = get_optimizer("_optimizer")
    print("[main] before sleep lr:", main_opt.get_lr())
    t.join()
    print("[main] after join lr:", main_opt.get_lr())


if __name__ == "__main__":
    main()
