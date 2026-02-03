"""
Unit tests for WeightsLab PyTorch Lightning integration.

Tests ensure compatibility with current PyTorch Lightning version and verify
that WeightsLab tracking works correctly within Lightning's training loop.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchmetrics.classification import Accuracy

try:
    import pytorch_lightning as pl
    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    PYTORCH_LIGHTNING_AVAILABLE = False

import weightslab as wl
from weightslab.backend.ledgers import GLOBAL_LEDGER, Proxy
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
    get_current_context,
    Context, 
    pause_controller
)


# Simple CNN model for testing
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.input_shape = (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Lightning Module with WeightsLab integration
class LitTestModel(pl.LightningModule):
    def __init__(self, model, optimizer, criterion_wl=None, metric_wl=None):
        super().__init__()
        self.model = model
        self.criterion_wl = criterion_wl
        self.metric_wl = metric_wl
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        with guard_training_context:
            x, ids, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            
            if self.criterion_wl is not None:
                loss_batch = self.criterion_wl(
                    logits.float(),
                    y.long(),
                    batch_ids=ids,
                    preds=preds
                )
                loss = loss_batch.mean()
            else:
                loss = nn.functional.cross_entropy(logits, y)
            
            return loss

    def validation_step(self, batch):
        with guard_testing_context:
            x, ids, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            
            if self.criterion_wl is not None:
                self.criterion_wl(
                    logits.float(),
                    y.long(),
                    batch_ids=ids,
                    preds=preds
                )
            
            if self.metric_wl is not None:
                self.metric_wl.update(logits, y)
            
            # Per-sample accuracy
            acc_per_sample = (preds == y).float()
            
            signals = {
                "val_metric/Accuracy_per_sample": acc_per_sample,
            }
            wl.save_signals(
                preds_raw=logits,
                targets=y,
                batch_ids=ids,
                signals=signals,
                preds=preds,
            )

    def configure_optimizers(self):
        return self.optimizer


@unittest.skipIf(not PYTORCH_LIGHTNING_AVAILABLE, "PyTorch Lightning not installed")
class TestPyTorchLightningIntegration(unittest.TestCase):
    """Test WeightsLab integration with PyTorch Lightning."""

    def setUp(self):
        """Set up test fixtures before each test."""
        GLOBAL_LEDGER.clear()
        self.temp_dir = tempfile.mkdtemp()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create small synthetic dataset (100 samples as requested)
        self.n_samples = 100
        self.n_classes = 10
        self.img_size = 28
        
        torch.manual_seed(42)
        x_data = torch.randn(self.n_samples, 1, self.img_size, self.img_size)
        y_data = torch.randint(0, self.n_classes, (self.n_samples,))
        
        self.train_dataset = TensorDataset(x_data[:80], y_data[:80])
        self.val_dataset = TensorDataset(x_data[80:], y_data[80:])

    def tearDown(self):
        """Clean up after each test."""
        GLOBAL_LEDGER.clear()
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_proxy_hashable_in_lightning(self):
        """Test that Proxy objects are hashable and work with Lightning's module system."""
        model = SimpleCNN()
        model_wl = wl.watch_or_edit(model, flag="model", device=self.device)
        
        # Test that proxy can be used in sets (requires __hash__)
        proxy_set = {model_wl}
        self.assertIn(model_wl, proxy_set)
         
        # Test that proxy can be used as dict key
        proxy_dict = {model_wl: "test_value"}
        self.assertEqual(proxy_dict[model_wl], "test_value")

    def test_lightning_module_with_weightslab_tracking(self):
        """Test that Lightning module can be created with WeightsLab tracked objects."""
        pause_controller.resume()  # Ensure not pausedv
        # Create model and wrap with WeightsLab
        _model = SimpleCNN().to(self.device)
        model_wl = wl.watch_or_edit(_model, flag="model", device=self.device)
        
        # Create tracked loss and metrics
        criterion = wl.watch_or_edit(
            nn.CrossEntropyLoss(reduction="none"),
            flag="loss", signal_name="loss-CE", log=True
        )
        
        metric = wl.watch_or_edit(
            Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device),
            flag="metric", signal_name="metric-ACC", log=True
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model_wl.parameters(), lr=0.001)
        optimizer_wl = wl.watch_or_edit(optimizer, flag="optimizer")
        
        # Create Lightning module with tracked objects
        lit_model = LitTestModel(
            model=model_wl,
            optimizer=optimizer_wl,
            criterion_wl=criterion,
            metric_wl=metric
        )
        
        # Verify Lightning module was created successfully
        self.assertIsInstance(lit_model, pl.LightningModule)
        self.assertIsInstance(lit_model.model, Proxy)

    def test_lightning_training_with_weightslab_loaders(self):
        """Test full training loop with WeightsLab tracked data loaders."""
        pause_controller.resume()  # Ensure not paused

        # Create tracked loaders
        train_loader = wl.watch_or_edit(
            self.train_dataset,
            flag="data",
            loader_name="train_loader",
            batch_size=16,
            shuffle=True,
            is_training=True,
            compute_hash=False,
            enable_h5_persistence=False
        )
        
        val_loader = wl.watch_or_edit(
            self.val_dataset,
            flag="data",
            loader_name="val_loader",
            batch_size=16,
            shuffle=False,
            is_training=False,
            compute_hash=False,
            enable_h5_persistence=False
        )
        
        # Create model with tracked components
        _model = SimpleCNN().to(self.device)
        model_wl = wl.watch_or_edit(_model, flag="model", device=self.device)
        
        criterion = wl.watch_or_edit(
            nn.CrossEntropyLoss(reduction="none"),
            flag="loss", signal_name="loss-CE", log=True
        )
        
        metric = wl.watch_or_edit(
            Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device),
            flag="metric", signal_name="metric-ACC", log=True
        )
        
        optimizer = torch.optim.Adam(model_wl.parameters(), lr=0.001)
        optimizer_wl = wl.watch_or_edit(optimizer, flag="optimizer")
        
        lit_model = LitTestModel(
            model=model_wl,
            optimizer=optimizer_wl,
            criterion_wl=criterion,
            metric_wl=metric
        )
        
        # Create Lightning trainer with minimal configuration
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator=self.device if self.device in ["cpu", "cuda"] else "auto",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        
        # Train the model - this should complete without errors
        try:
            trainer.fit(lit_model, train_loader, val_loader)
            training_succeeded = True
        except Exception as e:
            training_succeeded = False
            self.fail(f"Training failed with error: {e}")
        
        self.assertTrue(training_succeeded, "Training should complete successfully")

    def test_weightslab_context_guards_in_lightning(self):
        """Test that WeightsLab context guards work correctly in Lightning steps."""
        pause_controller.resume()  # Ensure not paused
        context_log = []
        
        class ContextTestModule(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = SimpleCNN()
            
            def forward(self, x):
                return self.model(x)
            
            def training_step(self, batch):
                with guard_training_context:
                    context = get_current_context()
                    context_log.append(("train", context))
                    x, y = batch
                    logits = self(x)
                    loss = nn.functional.cross_entropy(logits, y)
                    return loss
            
            def validation_step(self, batch):
                with guard_testing_context:
                    context = get_current_context()
                    context_log.append(("val", context))
                    x, y = batch
                    logits = self(x)
            
            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Create simple loaders without WeightsLab wrapping for this test
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=False
        )
        
        lit_model = ContextTestModule()
        
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        
        trainer.fit(lit_model, train_loader, val_loader)
        
        # Check that contexts were properly set
        train_contexts = [ctx for phase, ctx in context_log if phase == "train"]
        val_contexts = [ctx for phase, ctx in context_log if phase == "val"]
        
        self.assertTrue(len(train_contexts) > 0, "Training context should be captured")
        self.assertTrue(len(val_contexts) > 0, "Validation context should be captured")
        
        # Verify contexts are correct types
        for ctx in train_contexts:
            self.assertEqual(ctx, Context.TRAINING)
        
        for ctx in val_contexts:
            self.assertEqual(ctx, Context.TESTING)

    def test_lightning_version_compatibility(self):
        """Test that current PyTorch Lightning version is compatible."""
        import pytorch_lightning
        
        # Get version
        version = pytorch_lightning.__version__
        self.assertIsNotNone(version, "PyTorch Lightning version should be available")
        
        # Check minimum version (Lightning 2.0+)
        major_version = int(version.split('.')[0])
        self.assertGreaterEqual(major_version, 2, 
                               f"PyTorch Lightning version {version} may not be fully compatible")


if __name__ == "__main__":
    unittest.main()
