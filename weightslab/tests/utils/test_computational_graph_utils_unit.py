import unittest

import torch
import torch.nn as nn

from weightslab.utils.computational_graph import (
    _alias_from_tensor_name,
    _clean_dependencies,
    _generate_mappings,
    _infer_dependency_type,
    _propagate_constraints_through_dependencies,
)
from weightslab.utils.modules_dependencies import DepType


class TestComputationalGraphUtilsUnit(unittest.TestCase):
    def test_alias_from_tensor_name(self):
        self.assertEqual(_alias_from_tensor_name("/conv1/Conv_output_0"), "conv1")
        self.assertEqual(
            _alias_from_tensor_name("/model/layer1/layer1.0/conv1/Conv_output_0"),
            "model.layer1.0.conv1",
        )
        self.assertEqual(_alias_from_tensor_name("plain_name"), "plain_name")

    def test_generate_mappings_equal_and_many_to_one(self):
        src_to_dst, dst_to_src = _generate_mappings(range(4), range(4))
        self.assertEqual(src_to_dst[0], [0])
        self.assertEqual(dst_to_src[3], [3])

        src_to_dst2, dst_to_src2 = _generate_mappings(range(6), range(3))
        self.assertIn(0, src_to_dst2)
        self.assertIn(0, dst_to_src2)

    def test_clean_dependencies_and_infer_dependency_type(self):
        a, b = nn.Linear(2, 2), nn.ReLU()
        deps = [
            (a, a, DepType.SAME),
            (a, b, DepType.SAME),
            (a, b, DepType.SAME),
        ]
        cleaned = _clean_dependencies(deps)
        self.assertEqual(len(cleaned), 1)
        self.assertIs(cleaned[0][0], a)
        self.assertIs(cleaned[0][1], b)

        self.assertEqual(_infer_dependency_type(nn.Linear(3, 2)), DepType.INCOMING)
        self.assertEqual(_infer_dependency_type(nn.BatchNorm1d(4)), DepType.SAME)

    def test_propagate_constraints_through_dependencies(self):
        src = nn.Conv2d(4, 4, kernel_size=3, groups=2)
        mid = nn.ReLU()
        dst = nn.Linear(4, 2)

        deps = [
            (src, mid, DepType.SAME),
            (mid, dst, DepType.INCOMING),
        ]
        out = _propagate_constraints_through_dependencies(deps)

        src_id, mid_id, dst_id = id(src), id(mid), id(dst)
        self.assertIn("grouped", out[src_id]["outgoing"])
        self.assertIn("grouped", out[mid_id]["outgoing"])
        self.assertIn("grouped", out[mid_id]["incoming"])
        self.assertIn("grouped", out[dst_id]["incoming"])


if __name__ == "__main__":
    unittest.main()
