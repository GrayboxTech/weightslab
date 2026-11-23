"""Every Modeling feature on every model of the baseline zoo.

`test_model_representation.py` checks that each zoo model *traces*; this
sweep runs the whole Modeling tab on each of them, the way the Studio does,
on the model wrapped with ``compute_dependencies=True``: architecture, a
representation session, every layer's details (neurons, weights, previews,
the one-sample zoom), whole-layer edits followed by a forward, neuron prune /
add on a layer of every type the model has (the neuron graph keeps every
coupled tensor in step), and compare A vs B (pair and group). A model family
the Modeling tab cannot handle fails here, by name.

Inputs are shrunk to 64 px where a model accepts it, so the sweep stays in
minutes; token models (``vocab_size``) get token ids.
"""
import inspect
import unittest
from unittest import mock

import torch as th
import torch.nn as nn

from weightslab.backend.model_interface import ModelInterface
from weightslab.components import model_comparison as mc
from weightslab.components import model_representation as mr

try:
    from weightslab.examples.utils.baseline_models.pytorch import models as zoo
except Exception:  # pragma: no cover - example data, not a hard dependency
    zoo = None

# Constructor arguments needed / a trainer that loads COCO on forward.
SKIP = {"FlexibleCNNBlock", "Yolov11"}
SMALL = 64


class _NoLock:
    def acquire(self, timeout=None):
        return True

    def release(self):
        pass


def setUpModule():
    global _lock_patch
    _lock_patch = mock.patch("weightslab.components.global_monitoring.weightslab_rlock", _NoLock())
    _lock_patch.start()


def tearDownModule():
    _lock_patch.stop()


class _Tracked:
    def __init__(self, items):
        self.items = items
        self._dataset_split = "train"
        self.unique_ids = [f"s{i}" for i in range(len(items))]

    def get_index_from_sample_id(self, sid):
        return self.unique_ids.index(str(sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.unique_ids[i], 0


class _Loader:
    def __init__(self, items):
        self.tracked_dataset = _Tracked(items)

    def get_dataloader(self):
        return None


def _samples(model: nn.Module, n: int = 6):
    """*n* inputs the model accepts: token ids for a language model, else
    random tensors -- spatially shrunk to SMALL px when the model allows."""
    g = th.Generator().manual_seed(0)
    shape = tuple(model.input_shape)[1:]
    vocab = getattr(model, "vocab_size", None)
    if vocab:
        return [th.randint(0, int(vocab), shape, generator=g) for _ in range(n)]
    if len(shape) >= 3 and max(shape[1:]) > SMALL:
        small = (shape[0],) + tuple(min(s, SMALL) for s in shape[1:])
        try:
            model.eval()
            with th.no_grad():
                model(th.randn((1,) + small))
            shape = small
        except Exception:
            pass
    return [th.randn(shape, generator=g) for _ in range(n)]


def _shapes(out):
    if isinstance(out, th.Tensor):
        return [tuple(out.shape)]
    if isinstance(out, dict):
        return [s for v in out.values() for s in _shapes(v)]
    if isinstance(out, (list, tuple)):
        return [s for v in out for s in _shapes(v)]
    return []


def discover():
    found = []
    for name, obj in vars(zoo).items():
        if not (inspect.isclass(obj) and issubclass(obj, nn.Module)):
            continue
        if obj.__module__ != zoo.__name__ or name in SKIP:
            continue
        try:
            model = obj()
        except Exception:
            continue
        if getattr(model, "input_shape", None):
            found.append((name, model))
    return sorted(found, key=lambda row: row[0])


@unittest.skipIf(zoo is None, "baseline model zoo unavailable")
class TestModelingOnTheZoo(unittest.TestCase):
    def test_every_feature_on_every_zoo_model(self):
        models = discover()
        self.assertGreater(len(models), 20)
        for name, model in models:
            with self.subTest(model=name):
                self._sweep(name, model)

    def _sweep(self, name, model):
        items = _samples(model)
        model = ModelInterface(model, dummy_input=items[0].unsqueeze(0), compute_dependencies=True,
                               register=False, skip_previous_auto_load=True, log_model_signals=False)
        self.assertIsNotNone(model._neuron_editor, f"{name}: no neuron graph")
        loaders = {"train_loader": _Loader(items)}
        rep = mr.ModelRepresentationManager()

        arch = rep.architecture(model, loaders)
        self.assertTrue(arch["success"] and arch["traced"], f"{name}: {arch.get('trace_message')}")
        layers = [n for n in arch["nodes"] if n["kind"] == "layer" and n["executed"]]
        weighted = [n for n in layers if "perturb" in n["supported_ops"]]
        self.assertTrue(weighted, f"{name}: no layer with weights")

        # Representation: every layer that ran has statistics.
        status = rep.start(model, loaders, ["s0", "s1", "s2"], [])
        session = rep.wait(status["session_id"], timeout=600)
        self.assertEqual(session.state, "done", f"{name}: {session.message}")
        stats = {s["node_id"] for s in rep.summaries(session) if len(s) > 1}
        missing = [n["id"] for n in layers if n["id"] not in stats]
        self.assertFalse(missing, f"{name}: no statistics for {missing[:5]}")

        # Details of every layer; neurons for every weighted one.
        for node in layers:
            details = mr.layer_details(model, loaders, node["id"], manager=rep, skip_previews=True)
            self.assertTrue(details["success"], f"{name}/{node['id']}: {details.get('message')}")
            if node in weighted:
                self.assertTrue(details["neurons"], f"{name}/{node['id']}: no neurons")
        first = weighted[0]["id"]
        previews = mr.layer_details(model, loaders, first, manager=rep)["previews"]
        self.assertTrue(previews, f"{name}/{first}: no previews")
        zoom = mr.layer_details(model, loaders, first, manager=rep, sample_id="s1", tile_size=48)["previews"]
        self.assertTrue(zoom and all(p["sample_id"] == "s1" for p in zoom), f"{name}/{first}: zoom")

        # Whole-layer edits, then the model still runs.
        for op in ("freeze", "unfreeze", "perturb", "reset"):
            if op in weighted[0]["supported_ops"]:
                mr.edit_layer(model, first, op, manager=rep, ratio=0.1)
        model.eval()
        with th.no_grad():
            reference = _shapes(model(items[0].unsqueeze(0)))

        # Neuron prune / add on one layer of each type: every coupled tensor
        # follows, the outputs keep their shape. The output layer cannot change.
        nodes = {n["id"]: n for n in rep.architecture(model, loaders)["nodes"]}
        editable = [n for n in weighted if "prune" in nodes[n["id"]]["supported_ops"]]
        self.assertTrue(editable, f"{name}: no layer can be pruned")
        by_type = {}
        for n in editable:
            by_type.setdefault(n["type"], n["id"])
        for layer_type, node_id in by_type.items():
            for op in ("prune", "add"):
                try:
                    mr.edit_layer(model, node_id, op, neuron_indices=[0], count=1, manager=rep)
                except ValueError as exc:
                    if op == "prune" and "every neuron" in str(exc):
                        continue
                    self.fail(f"{name}/{node_id} ({layer_type}) {op}: {exc}")
                model.eval()
                with th.no_grad():
                    self.assertEqual(_shapes(model(items[0].unsqueeze(0))), reference,
                                     f"{name}/{node_id} {op}")
        details = mr.layer_details(model, loaders, editable[0]["id"], manager=rep, skip_previews=True)
        self.assertTrue(details["success"], f"{name}: details after the edits")

        # Compare A vs B: pair and group.
        cm = mc.ComparisonManager(rep)
        pair = cm.wait(cm.start(model, loaders, {"sample_ids": ["s0"]}, {"sample_ids": ["s1"]})["comparison_id"], 600)
        self.assertEqual(pair.state, "done", f"{name} pair: {pair.message}")
        group = cm.wait(cm.start(model, loaders, {"sample_ids": ["s0", "s1", "s2"]},
                                 {"sample_ids": ["s3", "s4", "s5"]})["comparison_id"], 600)
        self.assertEqual(group.state, "done", f"{name} group: {group.message}")
        self.assertGreaterEqual(len(cm.diffs(group)), len(layers) // 2, f"{name}: few diffs")
        self.assertTrue(cm.layer(group.id, first)["success"], f"{name}: layer comparison")


if __name__ == "__main__":
    unittest.main()
