"""The Modeling tab on transformers: microGPT and nanoGPT (scaled down, 2
blocks) from the baseline zoo.

Residual adds as "+" nodes (no all-to-all wiring of the residual stream),
attention / MLP roles, tied weights, attention maps per head (captured from
softmax or scaled_dot_product_attention), q / k / v head groups on the
projections' neurons, token-id previews, per-neuron freeze without layer
dependencies, and compare A vs B on token sequences.
"""
import io
import threading
import unittest
from unittest import mock

import torch as th
import torch.nn.functional as F

from weightslab.components import model_comparison as mc
from weightslab.components import model_representation as mr
from weightslab.examples.utils.baseline_models.pytorch.models import MicroGPT, NanoGPT, ResNet18


def setUpModule():
    global _lock_patch
    _lock_patch = mock.patch("weightslab.components.global_monitoring.weightslab_rlock", threading.RLock())
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


def _setup(model_cls, n=6):
    model = model_cls()
    g = th.Generator().manual_seed(0)
    items = [th.randint(0, model.vocab_size, (model.block_size,), generator=g) for _ in range(n)]
    loaders = {"train_loader": _Loader(items)}
    rep = mr.ModelRepresentationManager()
    return model, loaders, rep


def _session(rep, model, loaders, ids=("s0", "s1")):
    status = rep.start(model, loaders, list(ids), [])
    session = rep.wait(status["session_id"], 60)
    assert session.state == "done", session.message
    return session


class TestGraph(unittest.TestCase):
    def test_the_residual_stream_is_a_chain_of_adds(self):
        for cls in (NanoGPT, MicroGPT):
            with self.subTest(model=cls.__name__):
                model, loaders, rep = _setup(cls)
                arch = rep.architecture(model, loaders)
                nodes = {n["id"]: n for n in arch["nodes"]}
                ops = [n for n in arch["nodes"] if n["kind"] == "op"]
                # token + position embeddings, then two residual adds per block
                self.assertEqual(len(ops), 1 + 2 * 2)
                self.assertTrue(all(n["supported_ops"] == [] and n["type"] == "Add" for n in ops))
                inputs = {}
                for e in arch["edges"]:
                    inputs.setdefault(e["target"], set()).add(e["source"])
                    self.assertIn(e["source"], nodes)
                    self.assertIn(e["target"], nodes)
                    self.assertLess(nodes[e["source"]]["order"], nodes[e["target"]]["order"])
                # Every add joins exactly two branches; no layer is wired to
                # all the earlier blocks any more.
                for op in ops:
                    self.assertEqual(len(inputs[op["id"]]), 2, op["id"])
                block = "transformer.h.0" if cls is NanoGPT else "layers.0"
                self.assertEqual(nodes[f"{block}.add"]["parent_id"], block)
                self.assertEqual(nodes[f"{block}.add#2"]["parent_id"], block)
                last = "transformer.ln_f" if cls is NanoGPT else "lm_head"
                self.assertEqual(inputs[last], {(("transformer.h.1" if cls is NanoGPT else "layers.1") + ".add#2")})

    def test_resnet_shortcuts_become_adds_too(self):
        model = ResNet18()
        arch = mr.build_architecture(model, mr.trace_architecture(model, th.randn(1, 1, 64, 64)))
        adds = [n["id"] for n in arch["nodes"] if n["kind"] == "op"]
        self.assertTrue(any(a.endswith("layer1.0.add") for a in adds), adds[:5])
        self.assertEqual(len(adds), 8)                          # one per BasicBlock

    def test_roles_and_tied_weights(self):
        model, loaders, rep = _setup(NanoGPT)
        nodes = {n["id"]: n for n in rep.architecture(model, loaders)["nodes"]}
        self.assertEqual(nodes["transformer"]["role"], "backbone")
        self.assertEqual(nodes["transformer.h.0.attn"]["role"], "attention")
        self.assertEqual(nodes["transformer.h.1.mlp"]["role"], "mlp")
        self.assertIn("tied with lm_head", nodes["transformer.wte"]["description"])
        self.assertIn("tied with transformer.wte", nodes["lm_head"]["description"])


class TestRepresentation(unittest.TestCase):
    def test_attention_maps_per_head(self):
        for cls, group in ((NanoGPT, "transformer.h.0.attn"), (MicroGPT, "layers.1.attn")):
            with self.subTest(model=cls.__name__):
                model, loaders, rep = _setup(cls)
                _session(rep, model, loaders)
                details = mr.layer_details(model, loaders, group, manager=rep)
                att = [p for p in details["previews"] if p["direction"] == "attention"]
                self.assertEqual(len(att), 2)                    # one per sample
                for p in att:
                    self.assertEqual(p["layout"], "attention")
                    self.assertEqual(p["tiles"], 4)              # heads
                    self.assertEqual(p["shape"], [4, model.block_size, model.block_size])
                    self.assertGreaterEqual(min(p["tile_min"]), 0.0)
                    self.assertLessEqual(max(p["tile_max"]), 1.0 + 1e-6)
                zoom = mr.layer_details(model, loaders, group, manager=rep, sample_id="s1", tile_size=32)
                self.assertEqual([p["direction"] for p in zoom["previews"]].count("attention"), 1)

    def test_sdpa_weights_match_an_explicit_causal_softmax(self):
        g = th.Generator().manual_seed(0)
        q, k, v = (th.randn(2, 3, 5, 4, generator=g) for _ in range(3))
        w = mr._sdpa_weights(q, k, v, is_causal=True)
        att = (q @ k.transpose(-2, -1)) / 2.0
        att = att.masked_fill(th.ones(5, 5).tril() == 0, float("-inf")).softmax(-1)
        self.assertTrue(th.allclose(w, att, atol=1e-6))
        self.assertTrue(th.allclose(w @ v, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5))

    def test_head_groups_on_the_projections(self):
        model, loaders, rep = _setup(NanoGPT)
        groups = mr.layer_details(model, loaders, "transformer.h.0.attn.c_attn", manager=rep,
                                  skip_previews=True)["neuron_groups"]
        self.assertEqual(len(groups), 3 * 4)                     # q, k, v x 4 heads
        self.assertEqual((groups[0]["label"], groups[0]["start"], groups[0]["size"]), ("q · head 0", 0, 8))
        self.assertEqual((groups[5]["label"], groups[5]["start"]), ("k · head 1", 40))
        for out_proj in ("transformer.h.0.attn.c_proj", "transformer.h.0.mlp.c_fc"):
            self.assertEqual(mr.layer_details(model, loaders, out_proj, manager=rep,
                                              skip_previews=True)["neuron_groups"], [], out_proj)
        micro, mloaders, mrep = _setup(MicroGPT)
        groups = mr.layer_details(micro, mloaders, "layers.0.attn.wk", manager=mrep,
                                  skip_previews=True)["neuron_groups"]
        self.assertEqual([g["label"] for g in groups], ["head 0", "head 1", "head 2", "head 3"])

    def test_token_inputs_preview_as_tokens(self):
        model, loaders, rep = _setup(MicroGPT)
        model.itos = [chr(97 + i) for i in range(model.vocab_size)]
        _session(rep, model, loaders)
        p = mr.layer_details(model, loaders, mr.INPUT_NODE, manager=rep)["previews"][0]
        self.assertEqual(p["layout"], "tokens")
        ids = [int(v) for v in loaders["train_loader"].tracked_dataset.items[0]]
        self.assertEqual(p["token_ids"], ids)
        self.assertEqual(p["token_text"], [chr(97 + i) for i in ids])

    def test_compare_a_vs_b_on_token_sequences(self):
        model, loaders, rep = _setup(NanoGPT)
        cm = mc.ComparisonManager(rep)
        pair = cm.wait(cm.start(model, loaders, {"sample_ids": ["s0"]}, {"sample_ids": ["s1"]})["comparison_id"])
        self.assertEqual(pair.state, "done", pair.message)
        diffs = {d["node_id"]: d for d in cm.diffs(pair)}
        self.assertIn("transformer.h.1.mlp.c_fc", diffs)
        self.assertGreater(diffs["lm_head"]["score"], 0.0)
        layer = cm.layer(pair.id, "transformer.h.0.mlp.c_fc")
        self.assertEqual((layer["neuron_axis"], len(layer["neurons"])), ("feature", 128))


class TestNeuronFreezeWithoutDependencies(unittest.TestCase):
    def test_frozen_neurons_hold_under_adamw(self):
        model, loaders, rep = _setup(NanoGPT)
        layer = "transformer.h.0.mlp.c_fc"
        module = model.get_submodule(layer)
        before = module.weight.detach().clone()
        bias_before = module.bias.detach().clone()
        mr.edit_layer(model, layer, "freeze", [1, 3], manager=rep)
        opt = th.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
        for _ in range(3):
            x = th.randint(0, model.vocab_size, (2, model.block_size))
            loss = model(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model(x)
        self.assertTrue(th.equal(module.weight.detach()[[1, 3]], before[[1, 3]]))
        self.assertTrue(th.equal(module.bias.detach()[[1, 3]], bias_before[[1, 3]]))
        self.assertFalse(th.equal(module.weight.detach()[[0, 2]], before[[0, 2]]))
        frozen = [n["index"] for n in mr.layer_details(model, loaders, layer, manager=rep,
                                                       skip_previews=True)["neurons"] if n["frozen"]]
        self.assertEqual(frozen, [1, 3])
        node = next(n for n in rep.architecture(model, loaders, refresh=True)["nodes"] if n["id"] == layer)
        self.assertEqual(node["frozen_neuron_count"], 2)
        # An explicit reset of a frozen neuron sticks.
        mr.edit_layer(model, layer, "reset", [1], manager=rep)
        model(x)
        self.assertFalse(th.equal(module.weight.detach()[1], before[1]))
        # It pickles with the model (checkpoints dump it).
        buf = io.BytesIO()
        th.save(model, buf)
        buf.seek(0)
        restored = th.load(buf, weights_only=False)
        self.assertEqual(mr.FrozenRows.of(restored.get_submodule(layer)).indices(), {1, 3})
        # Unfreezing the neurons, or the whole layer, lifts it.
        mr.edit_layer(model, layer, "unfreeze", [1], manager=rep)
        self.assertEqual(mr.FrozenRows.of(module).indices(), {3})
        mr.edit_layer(model, layer, "unfreeze", [], manager=rep)
        self.assertEqual(mr.FrozenRows.of(module).indices(), set())
        with self.assertRaises(ValueError):
            mr.edit_layer(model, layer, "freeze", [999], manager=rep)

    def test_freezing_a_tied_head_freezes_the_embedding(self):
        model, loaders, rep = _setup(NanoGPT)
        mr.edit_layer(model, "lm_head", "freeze", [], manager=rep)
        nodes = {n["id"]: n for n in rep.architecture(model, loaders, refresh=True)["nodes"]}
        self.assertTrue(nodes["lm_head"]["frozen"])
        self.assertTrue(nodes["transformer.wte"]["frozen"])


if __name__ == "__main__":
    unittest.main()
