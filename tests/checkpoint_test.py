import torch as th
import unittest
import tempfile

from torch import optim
from torchvision import transforms as T
from torchvision import datasets as ds

from unittest import mock
from unittest.mock import ANY

from graybox.checkpoint import CheckpointManager
from graybox.experiment import Experiment

from .test_utils import MNISTModel

class CheckpointManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temporary_directory)

        th.manual_seed(1337)
        device = th.device("cuda:0")
        transform = T.Compose([T.ToTensor()])

        self.summary_writer_mock = mock.Mock()
        self.summary_writer_mock.add_scalars = mock.MagicMock()
        self.experiment = Experiment(
            model=MNISTModel(),
            optimizer_class=optim.Adam,
            train_dataset=ds.MNIST(
                "../data", train=True, transform=transform, download=True),
            eval_dataset=ds.MNIST("../data", train=False, transform=transform),
            device=device,
            learning_rate=1e-3,
            batch_size=32,
            name="x0",
            root_log_dir=self.temporary_directory,
            logger=self.summary_writer_mock,
            train_shuffle=False) 

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