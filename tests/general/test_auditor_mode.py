import unittest
import torch
import torch.nn as nn
from unittest.mock import patch

from weightslab.backend.optimizer_interface import OptimizerInterface

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

class TestAuditorMode(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.raw_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        # Register=False avoids polluting the global ledgers in tests
        self.optimizer = OptimizerInterface(self.raw_optimizer, register=False)

    @patch("weightslab.backend.ledgers.resolve_hp_name", return_value="main")
    @patch("weightslab.backend.ledgers.get_hyperparams")
    def test_auditor_mode_skips_optimization(self, mock_get_hp, mock_resolve_hp):
        """Test that optimizer step is skipped when auditor_mode is True."""
        # Enable auditor mode
        mock_get_hp.return_value = {"auditor_mode": True}
        
        # We patch the underlying raw_optimizer.step to see if it's called
        with patch.object(self.raw_optimizer, "step") as mock_raw_step:
            self.optimizer.step()
            
            # Underlying step should NOT be called in auditor mode
            mock_raw_step.assert_not_called()

    @patch("weightslab.backend.ledgers.resolve_hp_name", return_value="main")
    @patch("weightslab.backend.ledgers.get_hyperparams")
    def test_normal_mode_runs_optimization(self, mock_get_hp, mock_resolve_hp):
        """Test that optimizer step runs normally when auditor_mode is False."""
        # Disable auditor mode
        mock_get_hp.return_value = {"auditor_mode": False}
        
        with patch.object(self.raw_optimizer, "step") as mock_raw_step:
            self.optimizer.step()
            
            # Underlying step SHOULD be called in normal mode
            mock_raw_step.assert_called_once()
            
    @patch("weightslab.backend.ledgers.resolve_hp_name", return_value="main")
    @patch("weightslab.backend.ledgers.get_hyperparams")
    def test_auditor_mode_camel_case(self, mock_get_hp, mock_resolve_hp):
        """Test legacy 'auditorMode' camelCase parameter mapping."""
        mock_get_hp.return_value = {"auditorMode": True}
        
        with patch.object(self.raw_optimizer, "step") as mock_raw_step:
            self.optimizer.step()
            mock_raw_step.assert_not_called()

if __name__ == "__main__":
    unittest.main()
