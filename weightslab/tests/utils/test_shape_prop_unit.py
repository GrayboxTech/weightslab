import unittest

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from weightslab.utils.shape_prop import ShapeProp, _extract_tensor_metadata


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x).relu()


class TestShapePropUnit(unittest.TestCase):
    def test_extract_tensor_metadata_non_quantized(self):
        x = torch.randn(2, 4, requires_grad=True)
        meta = _extract_tensor_metadata(x)

        self.assertEqual(tuple(meta.shape), (2, 4))
        self.assertEqual(meta.dtype, x.dtype)
        self.assertTrue(meta.requires_grad)
        self.assertFalse(meta.is_quantized)
        self.assertEqual(meta.qparams, {})

    def test_shape_prop_populates_node_meta(self):
        model = _TinyNet()
        gm = symbolic_trace(model)
        inp = torch.randn(5, 4)

        out = ShapeProp(gm).propagate(inp)
        self.assertEqual(tuple(out.shape), (5, 3))

        node_names = {n.name: n for n in gm.graph.nodes}
        self.assertIn("x", node_names)
        self.assertIn("fc", node_names)
        self.assertIn("relu", node_names)

        self.assertIn("tensor_meta", node_names["fc"].meta)
        self.assertEqual(tuple(node_names["fc"].meta["tensor_meta"].shape), (5, 3))
        self.assertIn("type", node_names["relu"].meta)


if __name__ == "__main__":
    unittest.main()
