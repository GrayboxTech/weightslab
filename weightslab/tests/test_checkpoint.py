import os
import time
import warnings; warnings.filterwarnings("ignore")
import unittest
import tempfile
import torch as th

from unittest import mock


from torchvision import transforms as T
from torchvision import datasets as ds
from torch.utils.data import DataLoader

from weightslab.backend.ledgers import (
    register_model,
    register_optimizer,
    register_dataloader,
    register_logger,
)

from weightslab.tests.torch_models import FashionCNN
from weightslab.components.checkpoint import CheckpointManager


# Set Global Default Settings
th.manual_seed(42)  # Set SEED
TMP_DIR = tempfile.mkdtemp()
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.temporary_directory = TMP_DIR

        # Initialize the checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.temporary_directory)

        # Instanciate the model
        model = FashionCNN().to(DEVICE)

        # Dataset initialization
        data_eval = ds.MNIST(
            os.path.join(self.temporary_directory, "data"),
            download=True,
            train=False,
            transform=T.Compose([T.ToTensor()])
        )
        data_train = ds.MNIST(
            os.path.join(self.temporary_directory, "data"),
            train=True,
            transform=T.Compose([T.ToTensor()]),
            download=True
        )

        # Mock the summary writer
        self.summary_writer_mock = mock.Mock()
        self.summary_writer_mock.add_scalars = mock.MagicMock()

        # Register model, optimizer and dataloaders in the ledger
        register_model('exp_model', model)

        optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
        register_optimizer('exp_optimizer', optimizer)

        # Wrap datasets in DataLoaders so checkpoint manager can access .dataset
        train_loader = DataLoader(data_train, batch_size=32, shuffle=False)
        eval_loader = DataLoader(data_eval, batch_size=32, shuffle=False)
        register_dataloader('exp_train', train_loader)
        register_dataloader('exp_eval', eval_loader)

        # Register a mock logger as well (keeps compatibility with older code)
        register_logger('exp_logger', self.summary_writer_mock)

        # Helpers to mimic Experiment training/eval behavior
        self._model = model
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._eval_loader = eval_loader

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    # --- Helpers to mimic Experiment behaviour used in the original tests ---
    def _reset_data_iterators(self):
        """Reset any iterator state. DataLoader in PyTorch restarts on new
        iteration, so this is a no-op provided here for API compatibility.
        """
        return

    def _eval_n_steps(self, num_batches: int = 1):
        """Run evaluation for `num_batches` batches and return (loss, accuracy).

        This mimics `Experiment.eval_n_steps` used by the tests.
        """
        self._model.eval()
        correct = 0
        total = 0
        loss_accum = 0.0
        crit = th.nn.CrossEntropyLoss()
        with th.no_grad():
            for i, batch in enumerate(self._eval_loader):
                if i >= num_batches:
                    break
                x, y = batch
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                out = self._model(x)
                loss = crit(out, y)
                loss_accum += float(loss.item())
                preds = out.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += y.size(0)
        acc = correct / total if total else 0.0
        return (loss_accum / max(1, num_batches), acc)

    def _train_n_steps(self, n_samples: int = 32):
        """Train for approximately `n_samples` samples (consumes batches).

        This mimics `Experiment.train_n_steps` used by the tests.
        """
        self._model.train()
        eaten = 0
        crit = th.nn.CrossEntropyLoss()
        optimizer = self._optimizer
        for x, y in self._train_loader:
            if eaten >= n_samples:
                break
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            out = self._model(x)
            loss = crit(out, y)
            loss.backward()
            optimizer.step()
            eaten += y.size(0)

    def _set_learning_rate(self, lr: float):
        for g in self._optimizer.param_groups:
            g['lr'] = lr

    def test_three_dumps_one_load(self):
        # Dump a untrained model into checkpoint.
        self.assertFalse(self.checkpoint_manager.id_to_path)
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            train_loader_name='exp_train',
            eval_loader_name='exp_eval',
            experiment_name='x0'
        )
        self.assertTrue(0 in self.checkpoint_manager.id_to_path)
        self.assertEqual(self.checkpoint_manager.next_id, 0)
        self.assertEqual(self.checkpoint_manager.prnt_id, 0)

        # Eval the model pretraining.
        _, _ = self._eval_n_steps(16)
        self._reset_data_iterators()

        # Train for 2k samples. Eval on 8k samples.
        self._train_n_steps(32 * 2)
        _, eval_accuracy_post_2k_samples = self._eval_n_steps(16)
        self._reset_data_iterators()
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            train_loader_name='exp_train',
            eval_loader_name='exp_eval',
            experiment_name='x0'
        )
        self.assertTrue(1 in self.checkpoint_manager.id_to_path)

        # Train for another 2k samples. Eval on 8k samples.
        self._train_n_steps(32 * 2)
        _, _ = self._eval_n_steps(16)
        self._reset_data_iterators()
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            train_loader_name='exp_train',
            eval_loader_name='exp_eval',
            experiment_name='x0'
        )
        self.assertTrue(2 in self.checkpoint_manager.id_to_path)

        # Load the checkpoint afte first 2k samples. Eval.
        # Then change some hyperparameters and retrain.
        self.checkpoint_manager.load(
            1,
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            train_loader_name='exp_train',
            eval_loader_name='exp_eval'
        )
        _, eval_accuracy_post_2k_loaded = self._eval_n_steps(16)
        self._reset_data_iterators()
        self.assertEqual(eval_accuracy_post_2k_loaded,
                         eval_accuracy_post_2k_samples)
        self.assertEqual(self.checkpoint_manager.next_id, 2)
        self.assertEqual(self.checkpoint_manager.prnt_id, 1)
        self._set_learning_rate(1e-2)
        self._train_n_steps(32 * 2)
        _, _ = self._eval_n_steps(16)
        self._reset_data_iterators()
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            train_loader_name='exp_train',
            eval_loader_name='exp_eval',
            experiment_name='x0'
        )
        self.assertTrue(3 in self.checkpoint_manager.id_to_path)
        self.assertEqual(self.checkpoint_manager.id_to_prnt[3], 1)


if __name__ == '__main__':
    unittest.main()
