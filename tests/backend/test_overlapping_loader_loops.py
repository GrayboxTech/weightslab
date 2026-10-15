"""Two `for batch in loader:` loops over the same loader at once.

The training script's own test pass and an evaluation the Studio triggers can
iterate the same loader together, the first one paused mid-epoch. The loop
that finished first used to clear the loader's shared `is_a_loop` flag under
the other, whose epoch then never ended: the script's `for batch in
test_loader:` ran forever and training froze (the classification e2e's
"train for N steps" hang, after a test-split evaluation).
"""
import unittest

import torch
from torch.utils.data import TensorDataset

import weightslab.data.data_samples_with_ops as _dso
from weightslab.backend.dataloader_interface import DataLoaderInterface
from weightslab.backend.ledgers import Proxy

N, BATCH = 10, 2          # five batches an epoch
CAP = 50                  # "endless", for a loop that should stop after five


def drain(iterator):
    count = 0
    for _ in iterator:
        count += 1
        if count >= CAP:
            break
    return count


class TestOverlappingForLoops(unittest.TestCase):
    def setUp(self):
        with _dso._REGISTRY_LOCK:
            _dso._GLOBAL_UID_REGISTRY.clear()
        data = TensorDataset(torch.arange(N, dtype=torch.float32).unsqueeze(1), torch.zeros(N))
        self.loader = Proxy(DataLoaderInterface(data, batch_size=BATCH, shuffle=False))

    def test_a_paused_loop_still_ends_after_another_one_finishes(self):
        outer = iter(self.loader)       # the script's test pass ...
        next(outer)                     # ... paused mid-epoch
        self.assertLess(drain(iter(self.loader)), CAP)   # the evaluation, to its end
        self.assertLess(drain(outer), CAP, "the paused loop never saw its epoch end")

    def test_without_a_for_loop_plain_next_rolls_over_epochs(self):
        drain(iter(self.loader))
        outer = iter(self.loader)
        next(outer)
        drain(iter(self.loader))
        drain(outer)
        self.assertFalse(self.loader.is_a_loop)
        # The training loop's own `next(loader)` keeps going across epochs.
        for _ in range(3 * N // BATCH):
            next(self.loader)

    def test_an_abandoned_loop_releases_its_hold(self):
        it = iter(self.loader)
        next(it)
        it.close()                      # what a managed evaluation does when it stops early
        it.close()                      # once only
        self.assertFalse(self.loader.is_a_loop)


if __name__ == "__main__":
    unittest.main()
