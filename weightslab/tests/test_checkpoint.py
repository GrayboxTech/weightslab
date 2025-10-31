import os
import time
import warnings; warnings.filterwarnings("ignore")
import unittest
import tempfile
import torch as th

from unittest import mock


from torchvision import transforms as T
from torchvision import datasets as ds

from weightslab.tests.torch_models import FashionCNN
from weightslab.experiment.experiment import Experiment
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
        model = FashionCNN()

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

        # Initialize the experiment
        self.experiment = Experiment(
            model=model,
            optimizer_class=th.optim.Adam,
            input_shape=model.input_shape,
            train_dataset=data_train,
            eval_dataset=data_eval,
            device=DEVICE,
            learning_rate=1e-3,
            batch_size=32,
            name="x0",
            root_log_dir=self.temporary_directory,
            logger=self.summary_writer_mock,
            train_shuffle=False
        )

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    def test_three_dumps_one_load(self):
        # Dump a untrained model into checkpoint.
        self.assertFalse(self.checkpoint_manager.id_to_path)
        self.checkpoint_manager.dump(self.experiment)
        self.assertTrue(0 in self.checkpoint_manager.id_to_path)
        self.assertEqual(self.checkpoint_manager.next_id, 0)
        self.assertEqual(self.checkpoint_manager.prnt_id, 0)

        # Eval the model pretraining.
        _, _ = self.experiment.eval_n_steps(16)
        self.experiment.reset_data_iterators()

        # Train for 2k samples. Eval on 8k samples.
        self.experiment.train_n_steps(32 * 2)
        _, eval_accuracy_post_2k_samples = self.experiment.eval_n_steps(16)
        self.experiment.reset_data_iterators()
        self.checkpoint_manager.dump(self.experiment)
        self.assertTrue(1 in self.checkpoint_manager.id_to_path)

        # Train for another 2k samples. Eval on 8k samples.
        self.experiment.train_n_steps(32 * 2)
        _, _ = self.experiment.eval_n_steps(16)
        self.experiment.reset_data_iterators()
        self.checkpoint_manager.dump(self.experiment)
        self.assertTrue(2 in self.checkpoint_manager.id_to_path)

        # Load the checkpoint afte first 2k samples. Eval.
        # Then change some hyperparameters and retrain.
        self.checkpoint_manager.load(1, self.experiment)
        _, eval_accuracy_post_2k_loaded = self.experiment.eval_n_steps(16)
        self.experiment.reset_data_iterators()
        self.assertEqual(eval_accuracy_post_2k_loaded,
                         eval_accuracy_post_2k_samples)
        self.assertEqual(self.checkpoint_manager.next_id, 2)
        self.assertEqual(self.checkpoint_manager.prnt_id, 1)
        self.experiment.set_learning_rate(1e-2)
        self.experiment.train_n_steps(32 * 2)
        _, _ = self.experiment.eval_n_steps(16)
        self.experiment.reset_data_iterators()
        self.checkpoint_manager.dump(self.experiment)
        self.assertTrue(3 in self.checkpoint_manager.id_to_path)
        self.assertEqual(self.checkpoint_manager.id_to_prnt[3], 1)


if __name__ == '__main__':
    unittest.main()
