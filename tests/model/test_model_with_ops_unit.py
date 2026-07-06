import unittest

import torch.nn as nn

from weightslab.components.tracking import TrackingMode
from weightslab.models.model_with_ops import NetworkWithOps
from weightslab.utils.modules_dependencies import DepType


class _Layer(nn.Module):
    def __init__(self, module_id):
        super().__init__()
        self.module_id = module_id
        self.tracking_mode = TrackingMode.DISABLED

    def get_module_id(self):
        return self.module_id


class _Net(NetworkWithOps):
    def __init__(self):
        super().__init__()
        self.l1 = _Layer(10)
        self.l2 = _Layer(20)


@unittest.skip("Constraint detection and propagation tests are currently skipped due to ongoing refactor and potential changes in the underlying implementation. Will be re-enabled once the new system is in place more modeling.")
class TestModelWithOpsUnit(unittest.TestCase):
    def test_reverse_index_and_tracking_mode(self):
        net = _Net()
        self.assertEqual(net._reverse_indexing(-1, 2), 1)

        net._dep_manager.id_2_layer = {10: net.l1, 20: net.l2}

        self.assertEqual(net._reverse_indexing(-1, 2), 20)
        self.assertEqual(net._reverse_indexing(0, 2), 10)
        self.assertEqual(net._reverse_indexing(999, 2), 999)

        net.linearized_layers = [net.l1, net.l2]
        net.set_tracking_mode(TrackingMode.TRAIN)
        self.assertEqual(net.tracking_mode, TrackingMode.TRAIN)
        self.assertEqual(net.l1.tracking_mode, TrackingMode.TRAIN)

    def test_same_ancestors_returns_chain(self):
        net = _Net()
        net.register_dependencies([(net.l1, net.l2, DepType.SAME)])
        leaves = net._same_ancestors(20)
        self.assertIn(10, leaves)


if __name__ == "__main__":
    unittest.main()
