import unittest

import torch
import torch.nn as nn

from weightslab.utils.tools import (
    extract_in_out_params,
    get_children,
    get_module_by_name,
    get_module_device,
    is_module_with_ops,
    make_safelist,
    rename_with_ops,
    what_layer_type,
)


class _AttrModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_custom = 2
        self.out_custom = 3


class TestUtilsToolsUnit(unittest.TestCase):
    def test_extract_in_out_params_linear_batchnorm_relu(self):
        linear = nn.Linear(4, 2)
        in_dim, out_dim, in_name, out_name = extract_in_out_params(linear)
        self.assertEqual((in_dim, out_dim, in_name, out_name), (4, 2, "in_features", "out_features"))

        bn = nn.BatchNorm1d(6)
        in_dim2, out_dim2, _, _ = extract_in_out_params(bn)
        self.assertEqual((in_dim2, out_dim2), (6, 6))
        self.assertTrue(getattr(bn, "wl_same_flag", False))

        relu = nn.ReLU()
        in_dim3, out_dim3, _, _ = extract_in_out_params(relu)
        self.assertEqual((in_dim3, out_dim3), (None, None))
        self.assertTrue(getattr(relu, "wl_same_flag", False))

    def test_get_children_and_rename_with_ops(self):
        seq = nn.Sequential(nn.ReLU(), nn.Linear(3, 2))
        renamed_linear = seq[1]
        rename_with_ops(renamed_linear)

        self.assertTrue(is_module_with_ops(renamed_linear))
        children = get_children(seq)
        self.assertEqual(len(children), 1)
        self.assertTrue(is_module_with_ops(children[0]))

    def test_get_module_device_and_module_by_name(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        dev = get_module_device(model[0])
        self.assertEqual(str(dev), "cpu")

        self.assertIsNotNone(get_module_by_name(model, "0"))
        self.assertIsNone(get_module_by_name(model, "does_not_exist"))

        class _Paramless(nn.Module):
            def forward(self, x):
                return x

        self.assertEqual(str(get_module_device(_Paramless())), "cpu")

    def test_what_layer_type_and_make_safelist(self):
        m = _AttrModule()
        self.assertEqual(what_layer_type(m), 1)

        class _ShapeOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.output_shape = (1, 2)

        self.assertEqual(what_layer_type(_ShapeOnly()), 2)
        self.assertEqual(what_layer_type(nn.ReLU()), 0)

        self.assertEqual(make_safelist(1), [1])
        self.assertEqual(make_safelist([1, 2]), [1, 2])


if __name__ == "__main__":
    unittest.main()
