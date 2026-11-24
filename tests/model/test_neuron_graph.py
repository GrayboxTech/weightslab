"""Neuron prune / add on any architecture (weightslab.utils.neuron_graph).

Each test builds a small model with one structural pattern, edits it through
the engine and checks the shapes of every coupled tensor -- and, where the
pattern allows it, that the edit is exact: pruning a neuron gives the model
with that neuron silenced, adding one leaves the outputs unchanged.
"""
import copy
import unittest

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from weightslab.utils.neuron_graph import NeuronEditError, NeuronEditor


def _out(model, x):
    model.eval()
    with th.no_grad():
        return model(x)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 5)
        self.fc3 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class ConvFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 5 * 5, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, x):
        return self.head(F.relu(self.fc(F.relu(self.conv(x)).flatten(1))))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.branch = nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        h = F.relu(self.bn(self.stem(x)))
        return self.head(h + self.branch(h))


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 4, 3, padding=1)
        self.b = nn.Conv2d(3, 6, 3, padding=1)
        self.mix = nn.Conv2d(10, 5, 1)
        self.head = nn.Conv2d(5, 2, 1)

    def forward(self, x):
        return self.head(F.relu(self.mix(th.cat([F.relu(self.a(x)), F.relu(self.b(x))], dim=1))))


class Grouped(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.grouped = nn.Conv2d(8, 12, 3, padding=1, groups=2)
        self.dw = nn.Conv2d(12, 12, 3, padding=1, groups=12)
        self.gn = nn.GroupNorm(2, 12)
        self.head = nn.Conv2d(12, 2, 1)

    def forward(self, x):
        return self.head(self.gn(self.dw(self.grouped(F.relu(self.stem(x))))))


class TinyGPT(nn.Module):
    """Fused q/k/v split with chunk(3), heads, SDPA, residual, LayerNorm,
    an MLP and an lm_head tied to the token embedding."""

    def __init__(self, vocab=20, E=16, H=4):
        super().__init__()
        self.H = H
        self.wte = nn.Embedding(vocab, E)
        self.ln1 = nn.LayerNorm(E)
        self.qkv = nn.Linear(E, 3 * E)
        self.proj = nn.Linear(E, E)
        self.ln2 = nn.LayerNorm(E)
        self.fc = nn.Linear(E, 4 * E)
        self.fc_out = nn.Linear(4 * E, E)
        self.lm_head = nn.Linear(E, vocab, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, ids):
        x = self.wte(ids)
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.H, -1).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, -1))
        x = x + self.fc_out(F.gelu(self.fc(self.ln2(x))))
        return self.lm_head(x)


class HardCodedHeads(nn.Module):
    """Head size written as a number: neuron edits must refuse, not break."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        h = self.q(x).view(x.shape[0], 2, 4)
        return self.out(F.relu(h).reshape(x.shape[0], 8))


class Unflatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4 * 3 * 3)
        self.unflat = nn.Unflatten(1, (4, 3, 3))
        self.conv = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.conv(F.relu(self.unflat(self.fc(x))))


class OneChannel(nn.Module):
    """A 1-channel conv flattened into a linear layer (batch of 1)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.end = nn.Conv2d(4, 1, 1)
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc(self.end(F.relu(self.conv(x))).flatten(1))


class Flipped(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 6)
        self.b = nn.Linear(6, 2)

    def forward(self, x):
        return self.b(th.flip(self.a(x), dims=[1]))


class TestExactEdits(unittest.TestCase):
    def test_prune_equals_silencing_the_neuron(self):
        th.manual_seed(0)
        model, x = Mlp(), th.randn(4, 6)
        silenced = copy.deepcopy(model)
        with th.no_grad():
            silenced.fc2.weight[3] = 0
            silenced.fc2.bias[3] = 0
        expected = _out(silenced, x)
        res = NeuronEditor(model, x[:1]).prune(model.fc2, [3])
        self.assertEqual(tuple(model.fc2.weight.shape), (4, 8))
        self.assertEqual(tuple(model.fc3.weight.shape), (3, 4))
        self.assertEqual(model.fc2.out_features, 4)
        self.assertEqual(model.fc3.in_features, 4)
        self.assertEqual(res.neurons, 1)
        self.assertEqual(res.layers, ["fc2", "fc3"])
        self.assertTrue(th.allclose(_out(model, x), expected, atol=1e-6))

    def test_add_keeps_the_function(self):
        th.manual_seed(0)
        for model, x in ((Mlp(), th.randn(3, 6)), (ConvFlatten(), th.randn(2, 3, 5, 5)),
                         (Concat(), th.randn(2, 3, 6, 6))):
            with self.subTest(model=type(model).__name__):
                before = _out(model, x)
                first = next(m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d)))
                n = first.weight.shape[0]
                NeuronEditor(model, x[:1]).add(first, 3)
                self.assertEqual(first.weight.shape[0], n + 3)
                self.assertTrue(th.allclose(_out(model, x), before, atol=1e-6))

    def test_flatten_takes_every_column_of_a_channel(self):
        th.manual_seed(0)
        model, x = ConvFlatten(), th.randn(2, 3, 5, 5)
        silenced = copy.deepcopy(model)
        with th.no_grad():
            silenced.fc.weight[:, 25:50] = 0      # channel 1 = columns 25..49
        expected = _out(silenced, x)
        NeuronEditor(model, x[:1]).prune(model.conv, [1])
        self.assertEqual(model.conv.out_channels, 3)
        self.assertEqual(model.fc.in_features, 75)
        self.assertTrue(th.allclose(_out(model, x), expected, atol=1e-6))

    def test_concat_prunes_the_right_slice(self):
        th.manual_seed(0)
        model, x = Concat(), th.randn(2, 3, 6, 6)
        silenced = copy.deepcopy(model)
        with th.no_grad():
            silenced.mix.weight[:, 4 + 2] = 0     # b's channel 2 sits after a's 4
        expected = _out(silenced, x)
        NeuronEditor(model, x[:1]).prune(model.b, [2])
        self.assertEqual(model.b.out_channels, 5)
        self.assertEqual(model.mix.in_channels, 9)
        self.assertEqual(model.a.out_channels, 4)
        self.assertTrue(th.allclose(_out(model, x), expected, atol=1e-6))


class TestCouplings(unittest.TestCase):
    def test_residual_add_couples_both_branches(self):
        model, x = Residual(), th.randn(1, 3, 6, 6)
        res = NeuronEditor(model, x).prune(model.branch, [5])
        for m in (model.stem, model.branch):
            self.assertEqual(m.out_channels, 7)
        self.assertEqual(model.bn.num_features, 7)
        self.assertEqual(tuple(model.bn.running_mean.shape), (7,))
        self.assertEqual(model.head.in_channels, 7)
        self.assertEqual(set(res.layers), {"stem", "bn", "branch", "head"})
        _out(model, x)

    def test_groups_keep_their_size(self):
        model, x = Grouped(), th.randn(1, 3, 6, 6)
        ed = NeuronEditor(model, x)
        ed.prune(model.stem, [1])              # grouped input: one position per group
        self.assertEqual(model.stem.out_channels, 6)
        self.assertEqual(model.grouped.in_channels, 6)
        ed.prune(model.dw, [0])                # depthwise + 2 groups (conv, norm): one per group
        self.assertEqual(model.dw.out_channels, 10)
        self.assertEqual(model.dw.groups, 10)
        self.assertEqual(model.grouped.out_channels, 10)
        self.assertEqual(model.gn.num_channels, 10)
        _out(model, x)
        ed.add(model.grouped, 1)               # one per group
        self.assertEqual(model.grouped.out_channels, 12)
        self.assertEqual(model.dw.groups, 12)
        _out(model, x)

    def test_incompatible_groups_refuse(self):
        # 6 channels split in 2 groups by a conv and in 3 by a group norm:
        # removing one position per group of both removes them all.
        model = nn.Sequential(nn.Conv2d(3, 6, 1), nn.Conv2d(6, 6, 1, groups=2), nn.GroupNorm(3, 6),
                              nn.Conv2d(6, 2, 1))
        with self.assertRaisesRegex(NeuronEditError, "every neuron"):
            NeuronEditor(model, th.randn(1, 3, 4, 4)).prune(model[1], [0])

    def test_transformer_heads_residual_and_tied_weights(self):
        th.manual_seed(0)
        model, ids = TinyGPT(), th.randint(0, 20, (1, 6))
        ed = NeuronEditor(model, ids)
        # One q/k/v neuron: that position in every head, of q, k and v (chunk(3)).
        res = ed.prune(model.qkv, [5])
        self.assertEqual(res.neurons, 12)
        self.assertEqual(model.qkv.out_features, 36)
        self.assertEqual(model.proj.in_features, 12)
        self.assertEqual(model.ln1.normalized_shape, (16,))
        # One residual channel: every block writing / reading the stream, and
        # the embedding the lm_head shares.
        res = ed.prune(model.ln2, [0])
        self.assertEqual(model.wte.embedding_dim, 15)
        self.assertIs(model.lm_head.weight, model.wte.weight)
        self.assertEqual(model.ln1.normalized_shape, (15,))
        self.assertEqual(model.qkv.in_features, 15)
        self.assertEqual(model.fc_out.out_features, 15)
        self.assertEqual(model.proj.out_features, 15)
        self.assertIn("lm_head", res.layers)
        # MLP hidden neurons only touch the MLP.
        res = ed.add(model.fc, 2)
        self.assertEqual(model.fc.out_features, 66)
        self.assertEqual(model.fc_out.in_features, 66)
        self.assertEqual(set(res.layers), {"fc", "fc_out"})
        self.assertEqual(tuple(_out(model, ids).shape), (1, 6, 20))

    def test_one_channel_is_still_a_neuron(self):
        th.manual_seed(0)
        model, x = OneChannel(), th.randn(1, 3, 4, 4)
        before = _out(model, x)
        ed = NeuronEditor(model, x)
        self.assertTrue(ed.editable(model.end))
        ed.add(model.end, 1)
        self.assertEqual(model.end.out_channels, 2)
        self.assertEqual(model.fc.in_features, 32)
        self.assertTrue(th.allclose(_out(model, x), before, atol=1e-6))

    def test_unflatten_module_follows(self):
        model, x = Unflatten(), th.randn(1, 10)
        NeuronEditor(model, x).prune(model.fc, [0])   # one of 36 = a whole channel (3x3)
        self.assertEqual(model.fc.out_features, 27)
        self.assertEqual(tuple(model.unflat.unflattened_size), (3, 3, 3))
        self.assertEqual(model.conv.in_channels, 3)
        _out(model, x)


class TestRefusals(unittest.TestCase):
    def test_model_outputs_are_fixed(self):
        model, x = Mlp(), th.randn(1, 6)
        ed = NeuronEditor(model, x)
        self.assertIn("model output", ed.pinned_reason(model.fc3))
        with self.assertRaisesRegex(NeuronEditError, "model output"):
            ed.prune(model.fc3, [0])
        self.assertIsNone(ed.pinned_reason(model.fc1))

    def test_vocabulary_is_fixed(self):
        model, ids = TinyGPT(), th.randint(0, 20, (1, 5))
        ed = NeuronEditor(model, ids)
        self.assertFalse(ed.editable(model.lm_head))
        self.assertTrue(ed.editable(model.wte))

    def test_hard_coded_size_is_refused_and_rolled_back(self):
        model, x = HardCodedHeads(), th.randn(2, 8)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(NeuronEditError, "hard-codes a size"):
            NeuronEditor(model, x).prune(model.q, [0])
        after = model.state_dict()
        self.assertEqual(set(before), set(after))
        for k in before:
            self.assertTrue(th.equal(before[k], after[k]), k)
        self.assertEqual(model.q.out_features, 8)
        _out(model, x)

    def test_unknown_ops_pin_what_they_touch(self):
        model, x = Flipped(), th.randn(1, 4)
        ed = NeuronEditor(model, x)
        self.assertIn("flip", ed.pinned_reason(model.a))
        with self.assertRaises(NeuronEditError):
            ed.prune(model.a, [0])

    def test_never_every_neuron(self):
        model, x = Mlp(), th.randn(1, 6)
        with self.assertRaisesRegex(NeuronEditError, "every neuron"):
            NeuronEditor(model, x).prune(model.fc2, list(range(5)))


class TestTrainingAfterEdits(unittest.TestCase):
    def test_gradients_and_a_step(self):
        th.manual_seed(0)
        model, ids = TinyGPT(), th.randint(0, 20, (2, 6))
        ed = NeuronEditor(model, ids[:1])
        ed.prune(model.fc, [1, 2])
        ed.add(model.qkv, 1)
        opt = th.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        loss = F.cross_entropy(model(ids).flatten(0, 1), ids.flatten())
        loss.backward()
        opt.step()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
