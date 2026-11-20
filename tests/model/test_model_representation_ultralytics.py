"""The Modeling tab on Ultralytics YOLO models (detection and segmentation).

Built from the bundled yaml (no weights download, no dataset): the graph must
be a connected DAG with backbone / neck / head roles, each shared-activation
call must belong to its own block, a representation must run on images, and
the whole-layer edits must work. Skipped when ultralytics is not installed.
"""
import os
import unittest
from collections import defaultdict, deque
from unittest import mock

import torch as th
from torch.utils.data import Dataset

from weightslab.backend import ledgers
from weightslab.components import model_representation as mr

os.environ.setdefault("YOLO_OFFLINE", "1")
try:
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel
except Exception:  # pragma: no cover - optional dependency
    DetectionModel = SegmentationModel = None


class _NoLock:
    def acquire(self, timeout=None):
        return True

    def release(self):
        pass


class _Images(Dataset):
    def __init__(self, n=4, size=96):
        g = th.Generator().manual_seed(0)
        self.x = th.rand(n, 3, size, size, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], 0


class _Tracked:
    def __init__(self, base):
        self.base = base
        self._dataset_split = "train"
        self.unique_ids = [str(100 + i) for i in range(len(base))]

    def get_index_from_sample_id(self, sid):
        return self.unique_ids.index(str(sid))

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, self.unique_ids[i], y


class _Loader:
    def __init__(self, base):
        self.tracked_dataset = _Tracked(base)

    def get_dataloader(self):
        return None


@unittest.skipIf(DetectionModel is None, "ultralytics is not installed")
class TestUltralyticsModels(unittest.TestCase):
    def setUp(self):
        self._lock = mock.patch("weightslab.components.global_monitoring.weightslab_rlock", _NoLock())
        self._lock.start()
        self.loaders = {"train_loader": _Loader(_Images())}

    def tearDown(self):
        self._lock.stop()
        ledgers.clear_all()

    def _check(self, model):
        trace = mr.trace_architecture(model, th.rand(1, 3, 96, 96))
        self.assertTrue(trace.traced, trace.message)
        arch = mr.build_architecture(model, trace)
        layers = [n for n in arch["nodes"] if n["kind"] == "layer" and n["executed"]]
        groups = {n["id"]: n for n in arch["nodes"] if n["kind"] == "group"}
        self.assertGreater(len(layers), 100)

        # Connected: every layer is reached from the input, the output too.
        graph = defaultdict(list)
        for e in arch["edges"]:
            graph[e["source"]].append(e["target"])
        seen, queue = {mr.INPUT_NODE}, deque([mr.INPUT_NODE])
        while queue:
            for nxt in graph[queue.popleft()]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        self.assertEqual([n["id"] for n in layers if n["id"] not in seen], [])
        self.assertIn(mr.OUTPUT_NODE, seen)

        # The shared SiLU: one node per block, inside that block.
        self.assertEqual([n["id"] for n in layers if "#" in n["id"]], [])
        acts = [n for n in layers if n["id"].endswith(".act")]
        self.assertGreater(len(acts), 20)
        for n in acts:
            self.assertEqual(n["parent_id"], n["id"].rsplit(".", 1)[0])

        # Roles from model.yaml.
        last = max(int(k.split(".")[1]) for k in groups if k.count(".") == 1 and k.startswith("model."))
        self.assertEqual(groups["model.0"]["role"], "backbone")
        self.assertEqual(groups[f"model.{last}"]["role"], "head")
        # (Upsample / Concat blocks have no children: they are layers, roleless.)
        self.assertIn("neck", {g["role"] for k, g in groups.items() if k.count(".") == 1})

        # A representation on three images.
        manager = mr.ModelRepresentationManager()
        status = manager.start(model, self.loaders, ["100", "101", "102"], ["train"] * 3)
        session = manager.wait(status["session_id"], timeout=180)
        self.assertEqual(session.state, "done", session.message)
        self.assertEqual(session.done, 3)
        summaries = {s["node_id"]: s for s in manager.summaries(session)}
        self.assertIn("model.0.conv", summaries)
        self.assertIn("model.0.act", summaries)
        details = mr.layer_details(model, self.loaders, "model.0.conv", manager=manager)
        self.assertTrue(details["success"], details.get("message"))
        self.assertEqual(len(details["neurons"]), 16)
        self.assertTrue(details["previews"])
        head = mr.layer_details(model, self.loaders, f"model.{last}", manager=manager, skip_previews=True)
        self.assertTrue(head["weight_histogram"])

        # Whole-layer edits; prune needs the dependency graph and says so.
        for op in ("freeze", "unfreeze", "reset", "perturb"):
            result = mr.edit_layer(model, "model.0.conv", op, [], ratio=0.1, manager=manager)
            self.assertTrue(result.get("message"))
        self.assertTrue(all(p.requires_grad for p in model.model[0].parameters()))
        with self.assertRaises(ValueError):
            mr.edit_layer(model, "model.0.conv", "prune", [0], manager=manager)
        model(th.rand(1, 3, 96, 96))   # still runs

    def test_yolo11_detection(self):
        self._check(DetectionModel("yolo11n.yaml", nc=3, verbose=False))

    def test_yolo11_segmentation(self):
        self._check(SegmentationModel("yolo11n-seg.yaml", nc=3, verbose=False))


if __name__ == "__main__":
    unittest.main()
