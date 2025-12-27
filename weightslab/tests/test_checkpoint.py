import os
import time
import tqdm
import tempfile
import warnings; warnings.filterwarnings("ignore")
import unittest
import torch as th
import weightslab as wl

from unittest import mock

from torchvision import transforms as T
from torchvision import datasets as ds

from torch.utils.data import DataLoader
from weightslab.baseline_models.pytorch.models import FashionCNN
from weightslab.components.checkpoint import CheckpointManager


# Set Global Default Settings
th.manual_seed(42)  # Set SEED
DEVICE = "cpu"


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.temporary_directory = '/tmp/utests/'; os.makedirs('/tmp/utests/', exist_ok=True)

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
            train=False,
            transform=T.Compose([T.ToTensor()]),
            download=True
        )

        # Mock the summary writer
        self.summary_writer_mock = mock.Mock()
        self.summary_writer_mock.add_scalars = mock.MagicMock()

        # Register model, optimizer and dataloaders in the ledger
        self._model = wl.watch_or_edit(model, flag='model', name='exp_model')
        optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
        self._optimizer = wl.watch_or_edit(optimizer, flag='optimizer', name='exp_optimizer')

        # Wrap datasets in DataLoaders so checkpoint manager can access .dataset
        self._train_loader = wl.watch_or_edit(DataLoader(data_train, batch_size=128, shuffle=True), flag='dataloader', name='exp_train')
        self._eval_loader = wl.watch_or_edit(DataLoader(data_eval, batch_size=128, shuffle=False), flag='dataloader', name='exp_eval')

        # Register a mock logger as well (keeps compatibility with older code)
        self._logger = wl.watch_or_edit(self.summary_writer_mock, flag='logger', name='exp_logger')

        # Force controller to resume step
        from weightslab.components.global_monitoring import pause_controller
        pause_controller.resume()

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

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
            for i, batch in enumerate(tqdm.tqdm(self._eval_loader)):
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
        for x, y in tqdm.tqdm(self._train_loader):
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
        self.checkpoint_manager = CheckpointManager(tempfile.mkdtemp())
        self.assertFalse(self.checkpoint_manager.id_to_path)
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            experiment_name='x0'
        )
        self.assertTrue(0 in self.checkpoint_manager.id_to_path)
        self.assertEqual(self.checkpoint_manager.next_id, 0)
        self.assertEqual(self.checkpoint_manager.prnt_id, 0)

        # Eval the model pretraining.
        self._eval_n_steps(16)

        # Train for 2k samples. Eval on 8k samples.
        self._train_n_steps(32 * 2)
        _, eval_accuracy_post_2k_samples = self._eval_n_steps(16)
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            experiment_name='x0'
        )
        self.assertTrue(1 in self.checkpoint_manager.id_to_path)

        # Train for another 2k samples. Eval on 8k samples.
        self._train_n_steps(32 * 2)
        self._eval_n_steps(16)
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            experiment_name='x0'
        )
        self.assertTrue(2 in self.checkpoint_manager.id_to_path)

        # Load the checkpoint after first 2k samples. Eval.
        # Then change some hyperparameters and retrain.
        self.checkpoint_manager.load(
            1,
            model_name='exp_model',
            optimizer_name='exp_optimizer',
        )
        _, eval_accuracy_post_2k_loaded = self._eval_n_steps(16)
        self.assertEqual(eval_accuracy_post_2k_loaded,
                         eval_accuracy_post_2k_samples)
        self.assertEqual(self.checkpoint_manager.next_id, 2)
        self.assertEqual(self.checkpoint_manager.prnt_id, 1)
        self._set_learning_rate(1e-2)
        self._train_n_steps(32 * 2)
        self._eval_n_steps(16)
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            experiment_name='x0'
        )
        self.assertTrue(3 in self.checkpoint_manager.id_to_path)
        self.assertEqual(self.checkpoint_manager.id_to_prnt[3], 1)

    def test_operate_and_dump_and_load(self):
        self.checkpoint_manager = CheckpointManager(tempfile.mkdtemp())
        # Dump a untrained model into checkpoint.
        self.assertFalse(self.checkpoint_manager.id_to_path)
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            experiment_name='x0'
        )
        self.assertTrue(0 in self.checkpoint_manager.id_to_path)
        self.assertEqual(self.checkpoint_manager.next_id, 0)
        self.assertEqual(self.checkpoint_manager.prnt_id, 0)

        # Eval the model pretraining.
        _, _ = self._eval_n_steps(16)
        # Train for 2k samples. Eval on 8k samples.
        self._train_n_steps(32 * 2)

        # Operate on the model architecture
        # # Change HP
        self._set_learning_rate(1e-5)
        # # Add two neurons to layer 2
        self._model.apply_architecture_op(
            op_type=1,
            layer_id=0,
            neuron_indices=None
        )
        self._model.apply_architecture_op(
            op_type=1,
            layer_id=0,
            neuron_indices=None
        )
        # # Prune one neuron from layer 3
        self._model.apply_architecture_op(
            op_type=2,
            layer_id=3,
            neuron_indices=[-1]
        )
        # # Reset & Freeze last layer
        self._model.apply_architecture_op(
            op_type=4,
            layer_id=-1,
            neuron_indices=None
        )
        self._model.apply_architecture_op(
            op_type=3,
            layer_id=-1,
            neuron_indices=None
        )
        init_out_neurons_last_layer = self._model.layers[-2].out_features

        # Dump the modified model
        self.checkpoint_manager.dump(
            model_name='exp_model',
            optimizer_name='exp_optimizer',
            experiment_name='x0'
        )
        self.assertTrue(1 in self.checkpoint_manager.id_to_path)

        # Operate again
        # # Change HP
        self._set_learning_rate(1)
        # # Add two neurons to layer 2
        self._model.apply_architecture_op(
            op_type=1,
            layer_id=2,
            neuron_indices=[-1, -1]
        )
        # # Prune one neuron from layer 3
        self._model.apply_architecture_op(
            op_type=2,
            layer_id=4,
            neuron_indices=[-1]
        )

        # Load the checkpoint after first training and operations.
        # Then eval & change some hyperparameters and retrain.
        self.checkpoint_manager.load(
            1,
            model_name='exp_model',
            optimizer_name='exp_optimizer',
        )
        self.assertEqual(self.checkpoint_manager.next_id, 1)
        self.assertEqual(self.checkpoint_manager.prnt_id, 1)
        self.assertEqual(
            self._model.layers[-2].out_neurons,
            init_out_neurons_last_layer
        )
        # Operate
        # # Change HP
        self._set_learning_rate(1e-4)
        # # Prune one neuron from layer 3
        self._model.apply_architecture_op(
            op_type=2,
            layer_id=1,
            neuron_indices=[0]
        )
        # Retrain
        self._train_n_steps(32 * 2)
        # Eval again
        _, _ = self._eval_n_steps(16)


if __name__ == '__main__':
    unittest.main()
