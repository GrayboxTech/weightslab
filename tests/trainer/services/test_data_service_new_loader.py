"""A loader registered after the studio's view was built reaches the studio.

The Studio notebook runs cells inside the training process, so a cell can wrap
new data -- production samples, say -- with
``wl.watch_or_edit(ds, flag="data", loader_name="prod_loader")``. Its rows
reached the dataframe manager, but GetDataSplits / GetDataSamples kept serving
the old splits for good: the differential refresh read "none of these dirty
samples is in the view" as "nothing to do" and dropped their ids, so every
later refresh found nothing dirty and never rebuilt.
"""

import shutil
import tempfile
import time
import unittest

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset

import weightslab as wl
from weightslab.backend import ledgers
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.data_service import DataService


class _Samples(Dataset):
    def __init__(self, n, seed):
        g = th.Generator().manual_seed(seed)
        self.x = th.randn(n, 1, 8, 8, generator=g)
        self.y = th.randint(0, 3, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], i, int(self.y[i])


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 1, 8, 8)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _Context:
    def peer(self):
        return "test"

    def is_active(self):
        return True

    def invocation_metadata(self):
        return ()


class TestLoaderRegisteredAfterTheView(unittest.TestCase):
    def setUp(self):
        ledgers.clear_all()
        self.root = tempfile.mkdtemp(prefix="wl_new_loader_")
        wl.watch_or_edit({"experiment_name": "new_loader", "root_log_dir": self.root},
                         flag="hyperparameters")
        wl.watch_or_edit(_Net(), flag="model", device="cpu",
                         skip_previous_auto_load=True, projection=False)
        wl.watch_or_edit(_Samples(12, 0), flag="data", loader_name="train_loader",
                         batch_size=4, is_training=True)
        self.service = DataService(ExperimentContext("new_loader"))

    def tearDown(self):
        ledgers.clear_all()
        shutil.rmtree(self.root, ignore_errors=True)

    def _splits(self):
        return list(self.service.GetDataSplits(pb2.Empty(), _Context()).split_names)

    def _samples_per_split(self):
        response = self.service.GetDataSamples(
            pb2.DataSamplesRequest(start_index=0, records_cnt=100), _Context())
        self.assertTrue(response.success)
        counts = {}
        for record in response.data_records:
            origin = next(s.value_string for s in record.data_stats if s.name == "origin")
            counts[origin] = counts.get(origin, 0) + 1
        return counts

    def _wait_for_background_refresh(self):
        # A stale read kicks WL-ViewRefresh and holds this guard until it is done.
        guard = self.service._refresh_in_flight
        self.assertTrue(guard.acquire(timeout=60))
        guard.release()

    def test_the_new_split_and_its_samples_reach_the_studio(self):
        self.assertEqual(self._splits(), ["train_loader"])

        wl.watch_or_edit(_Samples(5, 1), flag="data", loader_name="prod_loader", batch_size=4)

        # Inside the 10 s rebuild throttle the view stays as it was, but the
        # new rows must not be forgotten.
        self.service._last_internals_update_time = time.time()
        self.assertEqual(self._splits(), ["train_loader"])

        # Throttle over: the next read starts the background rebuild ...
        self.service._last_internals_update_time = 0
        self._splits()
        self._wait_for_background_refresh()

        # ... and the reads after it serve the new split and its samples.
        self.assertEqual(self._splits(), ["prod_loader", "train_loader"])
        self.assertEqual(self._samples_per_split(), {"train_loader": 12, "prod_loader": 5})


if __name__ == "__main__":
    unittest.main()
