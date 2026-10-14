import unittest

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



class TestWidenColumnFor(unittest.TestCase):
    """widen_column_for mirrors pandas' implicit upcast, minus the FutureWarning
    ("Setting an item of incompatible dtype is deprecated"; an error in pandas 3)."""

    CASES = [
        ("int64 <- floats", "int64", [1, 2, 3], [107.695, 96.51]),
        ("int64 <- int-valued floats", "int64", [1, 2, 3], [5.0, 6.0]),
        ("float32 <- 0.1", "float32", [1, 2, 3], [0.1, float("nan")]),
        ("float32 <- exact float32", "float32", [1, 2, 3], [107.69508361816406, 2.5]),
        ("int8 <- big int", "int8", [1, 2, 3], [1000, 2]),
        ("bool <- floats", "bool", [True, False, True], [1.0, float("nan")]),
        ("float64 <- bools", "float64", [1.0, 2.0, 3.0], [False, False]),
        ("int64 <- bools", "int64", [1, 2, 3], [True, False]),
        ("float64 <- object floats", "float64", [1.0, 2.0, 3.0], ("object", [1.5, float("nan")])),
        ("bool <- object False/NaN", "bool", [True, True, True], ("object", [False, float("nan")])),
    ]

    def test_matches_pandas_upcast_without_warning(self):
        import warnings
        import numpy as np
        import pandas as pd
        from weightslab.utils.tools import widen_column_for

        for label, dtype, initial, new in self.CASES:
            with self.subTest(label):
                vals = (np.array(new[1], dtype=object) if isinstance(new, tuple)
                        else np.array(new))
                expected = pd.DataFrame({"c": pd.Series(initial, dtype=dtype)})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    expected.iloc[[0, 1], 0] = vals          # pandas' own upcast
                got = pd.DataFrame({"c": pd.Series(initial, dtype=dtype)})
                with warnings.catch_warnings():
                    warnings.simplefilter("error", FutureWarning)
                    got.iloc[[0, 1], 0] = widen_column_for(got, 0, vals)   # must not warn
                self.assertEqual(got["c"].dtype, expected["c"].dtype)
                self.assertTrue(got["c"].equals(expected["c"]))

    def test_object_bools_keep_a_bool_column_bool(self):
        """Better than pandas' own upcast: plain bools in an object array are
        written as bools, so the column doesn't degrade to object."""
        import warnings
        import numpy as np
        import pandas as pd
        from weightslab.utils.tools import widen_column_for

        df = pd.DataFrame({"discarded": [True, True, True]})
        vals = np.array([False, False], dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            df.iloc[[0, 1], 0] = widen_column_for(df, 0, vals)
        self.assertEqual(df["discarded"].dtype, bool)
        self.assertEqual(df["discarded"].tolist(), [False, False, True])

    def test_leaves_non_numeric_columns_alone(self):
        import numpy as np
        import pandas as pd
        from weightslab.utils.tools import widen_column_for

        df = pd.DataFrame({"s": ["a", "b"], "k": pd.Categorical(["x", "y"])})
        widen_column_for(df, 0, np.array([1.5, 2.5]))
        widen_column_for(df, 1, np.array([1.5, 2.5]))
        self.assertEqual(df["s"].dtype, object)
        self.assertIsInstance(df["k"].dtype, pd.CategoricalDtype)


if __name__ == "__main__":
    unittest.main()
