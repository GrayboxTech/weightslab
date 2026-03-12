import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from weightslab.backend.model_interface import ModelInterface


class _DummyWrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = 123

    def hello(self):
        return "world"


class TestModelInterfaceUnit(unittest.TestCase):
    def test_init_attributes_exposes_model_fields_and_methods(self):
        mi = ModelInterface.__new__(ModelInterface)
        nn.Module.__init__(mi)
        mi.model = _DummyWrapped()

        ModelInterface.init_attributes(mi, mi.model)

        self.assertEqual(mi.foo, 123)
        self.assertEqual(mi.hello(), "world")

    def test_exit_resets_visited_nodes_and_returns_false(self):
        mi = ModelInterface.__new__(ModelInterface)
        mi.visited_nodes = {"a", "b"}

        ret_ok = ModelInterface.__exit__(mi, None, None, None)
        self.assertFalse(ret_ok)
        self.assertEqual(mi.visited_nodes, set())

        ret_err = ModelInterface.__exit__(mi, RuntimeError, RuntimeError("x"), None)
        self.assertFalse(ret_err)

    def test_update_optimizer_recreates_optimizer_with_same_lr(self):
        mi = ModelInterface.__new__(ModelInterface)

        model = nn.Linear(2, 2)

        old_optimizer_wrapper = MagicMock()
        old_optimizer_wrapper.get_lr.return_value = [0.03]
        old_optimizer_wrapper.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with patch("weightslab.backend.model_interface.get_optimizers", return_value=["main"]), \
             patch("weightslab.backend.model_interface.get_optimizer", return_value=old_optimizer_wrapper), \
             patch("weightslab.backend.model_interface.wl.watch_or_edit") as watch_or_edit_mock:
            ModelInterface._update_optimizer(mi, model=model)

        watch_or_edit_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
