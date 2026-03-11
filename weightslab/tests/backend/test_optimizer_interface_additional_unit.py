import unittest
from unittest.mock import MagicMock, patch

import torch

from weightslab.backend.optimizer_interface import OptimizerInterface


class TestOptimizerInterfaceAdditionalUnit(unittest.TestCase):
    def test_construct_from_class_without_params_raises(self):
        with self.assertRaises(ValueError):
            OptimizerInterface(torch.optim.SGD, params=None, register=False)

    def test_step_skips_when_audit_mode_enabled(self):
        p = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        optim = torch.optim.SGD([p], lr=0.1)
        oi = OptimizerInterface(optim, register=False)

        with patch("weightslab.backend.ledgers.resolve_hp_name", return_value="main"), \
             patch("weightslab.backend.ledgers.get_hyperparams", return_value={"auditorMode": True}), \
             patch.object(oi.optimizer, "step") as step_mock:
            result = oi.step()

        self.assertIsNone(result)
        step_mock.assert_not_called()

    def test_step_updates_lr_from_hyperparams(self):
        p = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        optim = torch.optim.SGD([p], lr=0.1)
        oi = OptimizerInterface(optim, register=False)

        with patch("weightslab.backend.ledgers.resolve_hp_name", return_value="main"), \
             patch("weightslab.backend.ledgers.get_hyperparams", return_value={"optimizer": {"lr": 0.02}}):
            oi.step()

        self.assertEqual(oi.get_lr(), [0.02])

    def test_repr_contains_optimizer_name(self):
        p = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        optim = torch.optim.Adam([p], lr=0.01)
        oi = OptimizerInterface(optim, register=False)

        text = repr(oi)
        self.assertIn("OptimizerInterface", text)
        self.assertIn("Adam", text)


if __name__ == "__main__":
    unittest.main()
