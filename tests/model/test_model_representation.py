"""The Modeling tab's backend: architecture tracing, representation sessions,
layer details and layer edits.

The tracing sweep runs over every zero-arg model of the baseline zoo (CNNs,
residual nets, U-Nets, a 3-D U-Net, VAE/GAN, TinyYOLO...), so a model family
the graph cannot represent fails here, by name, rather than in the Studio.
"""
import inspect
import threading
import unittest
from collections import defaultdict, deque
from unittest import mock

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset

from weightslab.backend import ledgers
from weightslab.backend.model_interface import ModelInterface
from weightslab.components import model_representation as mr

try:
    from weightslab.examples.utils.baseline_models.pytorch import models as zoo
except Exception:  # pragma: no cover - example data, not a hard dependency
    zoo = None

# Same exclusions as the projection sweep: constructor args / a trainer that
# loads COCO from disk on forward.
SKIP = {"FlexibleCNNBlock", "Yolov11"}


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
        shape = getattr(model, "input_shape", None)
        if shape:
            found.append((name, model, tuple(shape)))
    return sorted(found, key=lambda row: row[0])


def reachable_from(edges, start):
    graph = defaultdict(list)
    for edge in edges:
        graph[edge["source"]].append(edge["target"])
    seen, queue = {start}, deque([start])
    while queue:
        for nxt in graph[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


class _NoLock:
    """Stand-in for the training lock in tests that run no training."""

    def acquire(self, timeout=None):
        return True

    def release(self):
        pass


def setUpModule():
    global _lock_patch
    _lock_patch = mock.patch(
        "weightslab.components.global_monitoring.weightslab_rlock", _NoLock())
    _lock_patch.start()


def tearDownModule():
    _lock_patch.stop()


@unittest.skipIf(zoo is None, "baseline model zoo unavailable")
class TestZooArchitectures(unittest.TestCase):
    def test_every_zoo_model_traces_to_a_connected_dag(self):
        models = discover()
        self.assertGreater(len(models), 10)
        for name, model, shape in models:
            with self.subTest(model=name):
                trace = mr.trace_architecture(model, th.randn((1,) + shape[1:]))
                self.assertTrue(trace.traced, f"{name}: {trace.message}")
                arch = mr.build_architecture(model, trace)
                nodes = {n["id"]: n for n in arch["nodes"]}
                layers = [n for n in arch["nodes"] if n["kind"] == "layer" and n["executed"]]
                self.assertTrue(layers, f"{name}: no executed layer")

                # Every learnable leaf that ran is a node.
                for qual, module in model.named_modules():
                    if qual and not list(module.children()) and list(module.parameters(recurse=False)):
                        if any(r["module_name"] == qual for r in trace.calls.values()):
                            self.assertIn(qual, nodes, f"{name}: {qual} missing")

                # Edges only join known nodes, and follow execution order (a DAG).
                order = {n["id"]: n["order"] for n in arch["nodes"]}
                for edge in arch["edges"]:
                    self.assertIn(edge["source"], nodes)
                    self.assertIn(edge["target"], nodes)
                    self.assertLess(order[edge["source"]], order[edge["target"]],
                                    f"{name}: back edge {edge}")

                # The output is fed, and every executed layer is reachable
                # from the input (no island the drawing would leave floating).
                self.assertTrue(any(e["target"] == mr.OUTPUT_NODE for e in arch["edges"]), name)
                seen = reachable_from(arch["edges"], mr.INPUT_NODE)
                unreachable = [n["id"] for n in layers if n["id"] not in seen]
                # A layer fed only by parameters/constants (no input tensor) is legit.
                self.assertLessEqual(len(unreachable), max(1, len(layers) // 20),
                                     f"{name}: unreachable {unreachable[:5]}")

                # Parents exist; groups know where they start.
                for node in arch["nodes"]:
                    if node.get("parent_id"):
                        self.assertIn(node["parent_id"], nodes)

    def test_residual_add_is_an_edge(self):
        model = zoo.SingleBlockResNetTruncated()
        trace = mr.trace_architecture(model, th.randn((1,) + model.input_shape[1:]))
        targets = defaultdict(set)
        for (src, dst) in trace.edges:
            targets[dst].add(src)
        self.assertTrue(any(len(srcs) >= 2 for srcs in targets.values()),
                        "a residual add should give some layer two producers")

    def test_unet_roles(self):
        model = zoo.UNet()
        trace = mr.trace_architecture(model, th.randn((1,) + model.input_shape[1:]))
        arch = mr.build_architecture(model, trace)
        roles = {n["id"]: n["role"] for n in arch["nodes"] if n["kind"] == "group" and n["depth"] == 0}
        self.assertIn("encoder", roles.values())
        self.assertIn("decoder", roles.values())


class TestGraphDetails(unittest.TestCase):
    def test_shared_module_gets_one_node_per_call(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 4)
                self.b = nn.Linear(4, 4)
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(self.b(self.act(self.a(x))))

        arch = mr.build_architecture(Shared(), mr.trace_architecture(Shared(), th.randn(1, 4)))
        ids = {n["id"] for n in arch["nodes"]}
        self.assertIn("act", ids)
        self.assertIn("act#2", ids)

    def test_multihead_attention_is_one_layer(self):
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(20, 8)
                self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
                self.head = nn.Linear(8, 3)

            def forward(self, tokens):
                x = self.emb(tokens)
                x, _ = self.attn(x, x, x)
                return self.head(x.mean(dim=1))

        model = Attn()
        arch = mr.build_architecture(model, mr.trace_architecture(model, th.randint(0, 20, (1, 5))))
        node = {n["id"]: n for n in arch["nodes"]}
        self.assertEqual(node["attn"]["kind"], "layer")
        self.assertNotIn("attn.out_proj", node)   # never ran as a module
        self.assertIn(("emb", "attn"), {(e["source"], e["target"]) for e in arch["edges"]})

    def test_dict_inputs_are_passed_as_kwargs(self):
        class KW(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 4)
                self.out = nn.Linear(4, 2)

            def forward(self, input_ids, attention_mask=None):
                x = self.emb(input_ids)
                if attention_mask is not None:
                    x = x * attention_mask.unsqueeze(-1)
                return self.out(x.mean(1))

        model = KW()
        inputs = {"input_ids": th.randint(0, 10, (1, 6)), "attention_mask": th.ones(1, 6)}
        trace = mr.trace_architecture(model, inputs)
        self.assertTrue(trace.traced, trace.message)
        self.assertIn(("emb", "out"), set(trace.edges))

    def test_untraceable_model_falls_back_to_a_chain(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        arch = mr.build_architecture(model, None)
        self.assertFalse(arch["traced"])
        self.assertEqual(arch["edges"][0]["source"], mr.INPUT_NODE)
        self.assertEqual(arch["edges"][-1]["target"], mr.OUTPUT_NODE)

    def test_preview_layouts(self):
        conv = mr.make_preview(th.randn(6, 28, 28), 0, 4)
        self.assertEqual((conv["layout"], conv["tiles"], conv["height"]), ("channels", 4, 28))
        self.assertEqual(len(conv["data"]), 4 * 28 * 28)
        vec = mr.make_preview(th.randn(64), None, 16)
        self.assertEqual((vec["layout"], vec["tiles"], vec["width"]), ("vector", 1, 64))
        seq = mr.make_preview(th.randn(10, 300), -1, 16)
        self.assertEqual(seq["layout"], "sequence")
        self.assertLessEqual(seq["width"], mr.PREVIEW_SEQUENCE)

    def test_model_input_previews_as_a_picture(self):
        # RGB in [0, 1]: one planar RGB picture, its own colours (no stretch).
        rgb = mr.make_preview(th.rand(3, 28, 28) * 0.5, 0, 16, as_image=True)
        self.assertEqual((rgb["layout"], rgb["tiles"], rgb["channel_indices"]), ("image", 1, [0, 1, 2]))
        self.assertEqual((rgb["tile_min"], rgb["tile_max"]), ([0.0], [1.0]))
        self.assertEqual(len(rgb["data"]), 3 * rgb["height"] * rgb["width"])
        self.assertLessEqual(max(rgb["data"]), 128)              # 0.5 stays mid-grey, not stretched
        # Gray, with or without a channel dim; 0..255 kept as is.
        gray = mr.make_preview(th.rand(28, 28) * 255, None, 16, as_image=True)
        self.assertEqual((gray["layout"], gray["channel_indices"], gray["tile_max"]), ("image", [0], [255.0]))
        # A normalised input: one range for the whole image.
        norm = mr.make_preview(th.randn(3, 20, 20) * 2, 0, 16, as_image=True)
        self.assertLess(norm["tile_min"][0], 0.0)
        # Not images: a point cloud, token ids, a 5-channel tensor.
        self.assertNotEqual(mr.make_preview(th.randn(500, 4), -1, 16, as_image=True)["layout"], "image")
        self.assertNotEqual(mr.make_preview(th.arange(12.0), None, 16, as_image=True)["layout"], "image")
        self.assertEqual(mr.make_preview(th.randn(5, 28, 28), 0, 16, as_image=True)["layout"], "channels")
        # Without as_image (any other layer): one heatmap tile per channel.
        self.assertEqual(mr.make_preview(th.rand(3, 28, 28), 0, 16)["layout"], "channels")

    def test_accumulator_flags_dead_channels(self):
        acc = mr.NodeAccumulator()
        x = th.relu(th.randn(4, 3, 5, 5))
        x[:, 1] = 0.0                       # channel 1 never fires
        acc.update(x, 1, 4)
        summary = acc.summary("n")
        self.assertEqual(summary["channels"], 3)
        self.assertEqual(summary["dead_channels"], 1)
        self.assertEqual(acc.channel_stats()["active_rate"][1], 0.0)

    def test_always_negative_channel_is_not_dead(self):
        # A conv/linear output that is always negative still carries signal;
        # only the ReLU after it (exact zeros) makes it dead.
        acc = mr.NodeAccumulator()
        x = th.randn(4, 3, 5, 5)
        x[:, 2] = -th.rand(4, 5, 5) - 0.1
        acc.update(x, 1, 4)
        self.assertEqual(acc.summary("n")["dead_channels"], 0)
        self.assertEqual(acc.channel_stats()["active_rate"][2], 1.0)

    def test_firing_is_relative_to_the_layer_scale(self):
        # A whole layer at 1e-6 scale (untrained BatchNorm statistics) is
        # alive; one exact-zero channel in it is still dead.
        acc = mr.NodeAccumulator()
        x = th.rand(4, 3, 5, 5) * 2e-6 + 1e-7
        x[:, 0] = 0.0
        acc.update(x, 1, 4)
        self.assertEqual(acc.summary("n")["dead_channels"], 1)
        # A SiLU neuron stuck at -1e-8 next to channels at ~1 is dead.
        acc = mr.NodeAccumulator()
        x = th.rand(4, 3, 5, 5) + 0.5
        x[:, 2] = -1e-8
        acc.update(x, 1, 4)
        self.assertEqual(acc.summary("n")["dead_channels"], 1)
        self.assertEqual(acc.channel_stats()["active_rate"][2], 0.0)

    def test_shared_module_is_named_under_each_caller(self):
        # Ultralytics' Conv blocks all hold ONE SiLU instance as `.act`: each
        # call belongs to the block that made it, not to the first owner.
        shared = nn.SiLU()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)
                self.act = shared

            def forward(self, x):
                return self.act(self.lin(x))

        model = nn.Sequential(Block(), Block(), Block())
        arch = mr.build_architecture(model, mr.trace_architecture(model, th.randn(2, 4)))
        layers = {n["id"]: n for n in arch["nodes"] if n["kind"] == "layer" and n["executed"]}
        for i in range(3):
            self.assertIn(f"{i}.act", layers)
            self.assertEqual(layers[f"{i}.act"]["parent_id"], str(i))
        self.assertFalse([k for k in layers if "#" in k])
        edges = {(e["source"], e["target"]) for e in arch["edges"]}
        self.assertIn(("1.lin", "1.act"), edges)
        self.assertIn(("1.act", "2.lin"), edges)
        # A representation names the calls the same way.
        manager = mr.ModelRepresentationManager()
        loader = _FakeLoader(_Vectors(), "train")
        status = manager.start(model, {"train_loader": loader}, ["1000", "1001"], ["train", "train"])
        session = manager.wait(status["session_id"], timeout=60)
        self.assertEqual(session.state, "done", session.message)
        summaries = {s["node_id"] for s in manager.summaries(session)}
        self.assertTrue({"0.act", "1.act", "2.act"} <= summaries)

    def test_ultralytics_style_yaml_declares_backbone_neck_head(self):
        class Yolo(nn.Module):
            def __init__(self):
                super().__init__()
                self.yaml = {"backbone": [[-1, 1, "Conv", []]] * 2, "head": [[-1, 1, "Conv", []]] * 3}
                self.model = nn.Sequential(*[nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(5)])

            def forward(self, x):
                return self.model(x)

        model = Yolo()
        arch = mr.build_architecture(model, mr.trace_architecture(model, th.randn(1, 4)))
        roles = {n["id"]: n["role"] for n in arch["nodes"] if n["kind"] == "group"}
        self.assertEqual([roles[f"model.{i}"] for i in range(5)],
                         ["backbone", "backbone", "neck", "neck", "head"])


# ---------------------------------------------------------------------------
# Sessions, details and edits on a registered-like setup (no gRPC)
# ---------------------------------------------------------------------------
class _Digits(Dataset):
    def __init__(self, n=24):
        g = th.Generator().manual_seed(0)
        self.x = th.rand(n, 1, 28, 28, generator=g)
        self.y = th.randint(0, 10, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _Vectors(Dataset):
    def __init__(self, n=8):
        g = th.Generator().manual_seed(0)
        self.x = th.randn(n, 4, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], 0


class _FakeTrackedDataset:
    """The parts of DataSampleTrackingWrapper the sample loader touches."""

    def __init__(self, base, split):
        self.base = base
        self._dataset_split = split
        self.unique_ids = [str(1000 + i) for i in range(len(base))] if split == "train" \
            else [str(5000 + i) for i in range(len(base))]

    def get_index_from_sample_id(self, sid):
        return list(self.unique_ids).index(str(sid))

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, self.unique_ids[i], y


class _FakeLoader:
    def __init__(self, base, split):
        self.tracked_dataset = _FakeTrackedDataset(base, split)

    def get_dataloader(self):
        return None


def _fashion_cnn(compute_dependencies):
    from weightslab.examples.utils.baseline_models.pytorch.models import FashionCNN
    wrapped = ModelInterface(
        FashionCNN(), dummy_input=th.randn(1, 1, 28, 28),
        compute_dependencies=compute_dependencies, register=False,
        skip_previous_auto_load=True)
    wrapped._architecture_change_hook_fns = []
    return wrapped


class TestSessionsAndEdits(unittest.TestCase):
    def setUp(self):
        self.loaders = {"train_loader": _FakeLoader(_Digits(), "train"),
                        "test_loader": _FakeLoader(_Digits(10), "test")}

    def tearDown(self):
        ledgers.clear_all()

    def _run(self, manager, model, ids, origins=()):
        status = manager.start(model, self.loaders, ids, list(origins))
        session = manager.wait(status["session_id"], timeout=60)
        self.assertEqual(session.state, "done", session.message)
        return session

    def test_representation_keeps_stats_and_previews(self):
        manager = mr.ModelRepresentationManager()
        model = _fashion_cnn(compute_dependencies=False)
        ids = ["1000", "1001", "1002", "5003"]
        session = self._run(manager, model, ids, ["train", "train", "train", "test"])
        self.assertEqual(session.done, 4)
        self.assertEqual(session.origins, ["train", "train", "train", "test"])
        summaries = {s["node_id"]: s for s in manager.summaries(session)}
        for node in ("c1", "r1", "fc3", "fc4", mr.INPUT_NODE, mr.OUTPUT_NODE):
            self.assertIn(node, summaries)
        self.assertEqual(summaries["c1"]["channels"], 4)
        out = session.previews["c1"]["output"]
        self.assertEqual(len([p for p in out if p]), 4)
        self.assertEqual(out[0]["layout"], "channels")
        self.assertEqual(session.previews["fc3"]["output"][0]["layout"], "vector")
        self.assertEqual(session.previews["c1"]["input"][0]["height"], 28)
        # The model input, and the first conv's input (the same tensor), are
        # pictures; the conv's output stays one heatmap per channel.
        self.assertEqual(session.previews[mr.INPUT_NODE]["output"][0]["layout"], "image")
        self.assertEqual(session.previews["c1"]["input"][0]["layout"], "image")
        self.assertEqual(session.previews["c1"]["output"][0]["layout"], "channels")
        # The model's age is untouched by representation forwards.
        self.assertEqual(model.get_age(), 0)

    def test_trace_uses_a_real_sample_not_the_declared_shape(self):
        import numpy as np

        class AnySize(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_shape = (1, 1, 64, 64)       # what it declares
                self.conv = nn.Conv2d(1, 3, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(3, 2)

            def forward(self, x):
                return self.fc(self.pool(self.conv(x)).flatten(1))

        class Small(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, i):
                return th.rand(1, 16, 16), i            # what the data is

        loader = _FakeLoader(Small(), "train")
        loader.tracked_dataset.unique_ids = np.array(["7", "8", "9", "10"])  # numpy, as in WL
        manager = mr.ModelRepresentationManager()
        arch = manager.architecture(AnySize(), {"train_loader": loader})
        self.assertTrue(arch["traced"], arch["trace_message"])
        node = next(n for n in arch["nodes"] if n["id"] == mr.INPUT_NODE)
        self.assertEqual(node["output_shape"], [1, 1, 16, 16])

    def test_clear_skip_previews_and_zoom(self):
        manager = mr.ModelRepresentationManager()
        model = _fashion_cnn(False)
        self._run(manager, model, ["1000", "1001"])

        # Cards: weights + neurons only, no previews, model age reported.
        light = mr.layer_details(model, self.loaders, "c1", manager=manager, skip_previews=True)
        self.assertTrue(light["success"])
        self.assertEqual(light["previews"], [])
        self.assertEqual(len(light["neurons"]), 4)
        self.assertEqual(light["model_age"], 0)
        self.assertIsNotNone(light["activation"])

        stored = [p for p in manager.get().previews["c1"]["output"] if p]
        # Zoom: one sample, recomputed bigger than the 28 px thumbnails.
        zoom = mr.layer_details(model, self.loaders, "c1", manager=manager,
                                sample_id="1001", origin="train", tile_size=64, max_channels=64)
        self.assertEqual({p["direction"] for p in zoom["previews"]}, {"input", "output"})
        self.assertTrue(all(p["sample_id"] == "1001" for p in zoom["previews"]))
        out = next(p for p in zoom["previews"] if p["direction"] == "output")
        self.assertEqual((out["tiles"], out["height"]), (4, 28))   # 28 px input: never upscaled

        # ...and never written into the session's own thumbnails.
        self.assertEqual([p for p in manager.get().previews["c1"]["output"] if p], stored)

        # Clear: nothing left to show.
        manager.clear()
        self.assertIsNone(manager.get())

    def test_group_details_have_a_weight_histogram(self):
        model = nn.Sequential(nn.Sequential(nn.Linear(4, 8), nn.ReLU()), nn.Linear(8, 2))
        loader = _FakeLoader(type("V", (Dataset,), {
            "__len__": lambda self: 3, "__getitem__": lambda self, i: (th.rand(4), i)})(), "train")
        manager = mr.ModelRepresentationManager()
        details = mr.layer_details(model, {"train_loader": loader}, "0", manager=manager)
        self.assertTrue(details["success"], details.get("message"))
        self.assertEqual(details["node"]["kind"], "group")
        self.assertEqual(len(details["weight_histogram"]), 40)
        self.assertEqual(sum(details["weight_histogram"]), 8 * 4 + 8)   # every parameter

    def test_unknown_samples_error_cleanly(self):
        manager = mr.ModelRepresentationManager()
        status = manager.start(_fashion_cnn(False), self.loaders, ["nope"], [])
        self.assertEqual(status["state"], "error")

    def test_new_session_cancels_the_running_one(self):
        manager = mr.ModelRepresentationManager()
        model = _fashion_cnn(False)
        gate = threading.Event()
        real = manager._forward_chunk

        def slow(*args, **kwargs):
            gate.wait(5)
            return real(*args, **kwargs)

        manager._forward_chunk = slow
        first = manager.start(model, self.loaders, ["1000"], [])
        second = manager.start(model, self.loaders, ["1001"], [])
        gate.set()
        self.assertEqual(manager.wait(first["session_id"]).state, "cancelled")
        self.assertEqual(manager.wait(second["session_id"]).state, "done")

    def test_layer_details_health(self):
        manager = mr.ModelRepresentationManager()
        model = _fashion_cnn(False)
        with th.no_grad():
            # Inputs are in [0, 1]: positive weights + bias always fire (a
            # random init can leave a channel negative everywhere by chance).
            model.model.c1.weight.uniform_(0.01, 0.1)
            model.model.c1.bias.fill_(0.1)
            model.model.c1.weight[2].zero_()
            model.model.c1.bias[2] = -1.0           # channel 2 can never fire
        self._run(manager, model, ["1000", "1001", "1002"])
        details = mr.layer_details(model, self.loaders, "r1", manager=manager)
        self.assertTrue(details["success"], details["message"])
        conv = mr.layer_details(model, self.loaders, "c1", manager=manager)
        self.assertEqual(len(conv["neurons"]), 4)
        # c1's channel 2 is a constant -1: alive at c1, dead at the ReLU it
        # feeds (c1 -> b1 -> r1), which is where the verdict comes from.
        self.assertEqual(conv["health_source"], "r1 (ReLU)")
        self.assertEqual(conv["neurons"][2]["health"], "dead")
        self.assertNotEqual(conv["neurons"][0]["health"], "dead")
        self.assertEqual(conv["neuron_axis"], "channel")
        summaries = {s["node_id"]: s for s in manager.summaries(manager.get())}
        self.assertEqual(summaries["c1"]["dead_channels"], 0)
        self.assertEqual(summaries["r1"]["dead_channels"], 1)
        # fc3 feeds fc4 directly (no ReLU): its own, signed output decides.
        fc3 = mr.layer_details(model, self.loaders, "fc3", manager=manager)
        self.assertEqual(fc3["health_source"], "fc3")
        self.assertFalse(any(n["health"] == "dead" for n in fc3["neurons"]))
        self.assertEqual(len(conv["weight_histogram"]), 40)
        directions = {p["direction"] for p in conv["previews"]}
        self.assertEqual(directions, {"input", "output"})

    def test_whole_layer_edits_without_dependencies(self):
        manager = mr.ModelRepresentationManager()
        model = _fashion_cnn(False)
        arch = manager.architecture(model, self.loaders)
        self.assertTrue(arch["traced"], arch["trace_message"])
        fc3 = next(n for n in arch["nodes"] if n["id"] == "fc3")
        self.assertEqual(set(fc3["supported_ops"]), {"freeze", "unfreeze", "reset", "perturb"})

        mr.edit_layer(model, "fc3", "freeze", manager=manager)
        self.assertFalse(model.model.fc3.weight.requires_grad)
        self.assertTrue(next(n for n in manager.architecture(model, self.loaders)["nodes"]
                             if n["id"] == "fc3")["frozen"])
        mr.edit_layer(model, "fc3", "unfreeze", manager=manager)
        self.assertTrue(model.model.fc3.weight.requires_grad)

        before = model.model.fc3.weight.detach().clone()
        mr.edit_layer(model, "fc3", "reset", [0, 5], manager=manager)
        after = model.model.fc3.weight.detach()
        self.assertFalse(th.equal(before[0], after[0]))
        self.assertTrue(th.equal(before[1], after[1]))

        mr.edit_layer(model, "fc3", "perturb", [], ratio=0.5, manager=manager)
        with self.assertRaisesRegex(ValueError, "compute_dependencies"):
            mr.edit_layer(model, "fc3", "prune", [0], manager=manager)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            mr.edit_layer(model, "fc3", "perturb", ratio=1.5, manager=manager)

    def test_neuron_edits_with_dependencies(self):
        manager = mr.ModelRepresentationManager()
        model = _fashion_cnn(True)
        arch = manager.architecture(model, self.loaders)
        self.assertTrue(arch["dependencies_available"])
        c1 = next(n for n in arch["nodes"] if n["id"] == "c1")
        self.assertIn("prune", c1["supported_ops"])
        self.assertGreaterEqual(c1["layer_id"], 0)
        version = arch["architecture_version"]

        session = self._run(manager, model, ["1000", "1001"])
        mr.edit_layer(model, "c1", "add", count=2, manager=manager)
        arch = manager.architecture(model, self.loaders)
        self.assertEqual(next(n for n in arch["nodes"] if n["id"] == "c1")["output_neurons"], 6)
        self.assertGreater(arch["architecture_version"], version)
        self.assertTrue(manager.is_stale(session))

        mr.edit_layer(model, "c1", "prune", [0], manager=manager)
        arch = manager.architecture(model, self.loaders)
        self.assertEqual(next(n for n in arch["nodes"] if n["id"] == "c1")["output_neurons"], 5)
        self.assertEqual(tuple(model(th.randn(2, 1, 28, 28)).shape), (2, 10))

        mr.edit_layer(model, "fc3", "freeze", [1, 2], manager=manager)
        node = next(n for n in manager.architecture(model, self.loaders)["nodes"] if n["id"] == "fc3")
        self.assertEqual(node["frozen_neuron_count"], 2)

        # The representation still works on the edited model.
        session = self._run(manager, model, ["1000"])
        self.assertEqual({s["node_id"]: s for s in manager.summaries(session)}["c1"]["channels"], 5)


if __name__ == "__main__":
    unittest.main()
