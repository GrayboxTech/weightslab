import unittest

import torch
import torch.nn as nn

from weightslab.utils.plot_graph import get_module_by_name, get_shape_string, make_safelist


class _NodeLike:
    def __init__(self, meta):
        self.meta = meta


class TestPlotGraphUnit(unittest.TestCase):
    def test_get_shape_string_with_and_without_tensor_meta(self):
        x = torch.randn(2, 3)
        node_with = _NodeLike(meta={"tensor_meta": x})
        node_without = _NodeLike(meta={})

        self.assertEqual(get_shape_string(node_with), str(tuple(x.shape)))
        self.assertEqual(get_shape_string(node_without), "N/A")

    def test_make_safelist_and_get_module_by_name(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.assertIsNotNone(get_module_by_name(model, "0"))
        self.assertIsNone(get_module_by_name(model, "missing"))

        self.assertEqual(make_safelist(1), [1])
        self.assertEqual(make_safelist([1, 2]), [1, 2])


if __name__ == "__main__":
    unittest.main()
