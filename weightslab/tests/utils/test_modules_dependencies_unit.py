import unittest

import torch.nn as nn

from weightslab.utils.modules_dependencies import DepType, _ModulesDependencyManager


class TestModulesDependencyManagerUnit(unittest.TestCase):
    def test_register_and_query_dependencies(self):
        mgr = _ModulesDependencyManager()
        m1, m2, m3 = nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)

        mgr.register_module(1, m1)
        mgr.register_module(2, m2)
        mgr.register_module(3, m3)

        mgr.register_same_dependency(1, 2)
        mgr.register_incoming_dependency(2, 3)
        mgr.register_rec_dependency(1, 3)

        self.assertIs(mgr.id_2_layer[1], m1)
        self.assertEqual(mgr.get_child_ids(1, DepType.SAME), [2])
        self.assertEqual(mgr.get_child_ids(2, DepType.INCOMING), [3])
        self.assertEqual(mgr.get_child_ids(1, DepType.REC), [3])
        self.assertEqual(mgr.get_child_ids(42, DepType.SAME), [])

        self.assertEqual(mgr.get_parent_ids(3, DepType.INCOMING), [2])
        self.assertEqual(mgr.get_parent_ids(3, DepType.REC), [1])

    def test_str_contains_manager_name(self):
        mgr = _ModulesDependencyManager()
        text = str(mgr)
        self.assertIn("ModulesDependencyManager", text)


if __name__ == "__main__":
    unittest.main()
