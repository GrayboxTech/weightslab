import unittest
import torch
from torch import nn

from weightslab.backend.optimizer_interface import OptimizerInterface


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 1)

    def forward(self, x):
        return self.l1(x)


class TestOptimizerInterface(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()

    def test_wrap_existing_optimizer_and_step(self):
        optim = torch.optim.SGD(self.model.parameters(), lr=0.1)
        oi = OptimizerInterface(optim, register=False)

        # initial lr
        self.assertEqual(oi.get_lr(), [0.1])

        # do a tiny forward/backward and step
        x = torch.randn(4, 10)
        y = torch.randn(4, 1)
        pred = self.model(x)
        loss = (pred - y).pow(2).mean()
        oi.zero_grad()
        loss.backward()
        oi.step()

    def test_construct_from_class_and_set_lr(self):
        oi = OptimizerInterface(torch.optim.SGD, params=self.model.parameters(), lr=0.05, register=False)
        # constructed should be True when created from class
        self.assertTrue(getattr(oi, '_constructed', False))
        self.assertEqual(oi.get_lr(), [0.05])

        # set lr for all groups
        oi.set_lr(0.01)
        self.assertEqual(oi.get_lr(), [0.01])

        # set lr for specific group (0)
        oi.set_lr(0.02, group_idx=0)
        self.assertEqual(oi.get_lr(), [0.02])

    def test_state_dict_and_load(self):
        oi = OptimizerInterface(torch.optim.Adam, params=self.model.parameters(), lr=1e-3, register=False)
        original = oi.state_dict()

        # change lr and then reload original
        oi.set_lr(0.5)
        self.assertNotEqual(oi.get_lr(), [g.get('lr') for g in original['param_groups']])
        oi.load_state_dict(original)
        self.assertEqual(oi.get_lr(), [g.get('lr') for g in original['param_groups']])


if __name__ == '__main__':
    unittest.main()
