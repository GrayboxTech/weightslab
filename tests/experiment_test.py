"""Test for the core and main object of the graybox package."""
import unittest
import tempfile

from unittest.mock import ANY
from unittest import mock

import torch as th
from torch import optim
from torchvision import transforms as T
from torchvision import datasets as ds

from weightslab.experiment import Experiment

from test_utils import MNISTModel


class ExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        th.manual_seed(1337)
        self.temporary_directory = tempfile.mkdtemp()
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        transform = T.Compose([T.ToTensor()])
        data_eval = ds.MNIST("../data", train=False, transform=transform)
        data_train = ds.MNIST(
            "../data", train=True, transform=transform, download=True)

        self.summary_writer_mock = mock.Mock()
        self.summary_writer_mock.add_scalars = mock.MagicMock()
        self.experiment = Experiment(
            model=MNISTModel(),
            optimizer_class=optim.Adam,
            train_dataset=data_train,
            eval_dataset=data_eval,
            device=device,
            learning_rate=1e-3,
            batch_size=32,
            name="x0",
            root_log_dir=self.temporary_directory,
            logger=self.summary_writer_mock,
            train_shuffle=False)        

    def test_set_learning_rate(self):
        self.assertEqual(self.experiment.learning_rate, 1e-3)
        self.assertEqual(
            self.experiment.optimizer.state_dict()['param_groups'][0]['lr'],
            1e-3)
        self.experiment.set_learning_rate(1e-2)
        self.assertEqual(self.experiment.learning_rate, 1e-2)
        self.assertEqual(
            self.experiment.optimizer.state_dict()['param_groups'][0]['lr'],
            1e-2)

    def test_set_batch_size(self):
        self.assertEqual(self.experiment.batch_size, 32)
        self.assertEqual(self.experiment.train_loader.batch_size, 32)
        self.assertEqual(self.experiment.eval_loader.batch_size, 32)
        self.experiment.set_batch_size(64)
        self.assertEqual(self.experiment.batch_size, 64)
        self.assertEqual(self.experiment.train_loader.batch_size, 64)
        self.assertEqual(self.experiment.eval_loader.batch_size, 64)

    def test_eval_step(self):
        self.assertEqual(self.experiment.model.get_age(), 0)
        self.summary_writer_mock.add_scalars.assert_not_called()
        eval_loss, eval_accuracy = self.experiment.eval_n_steps(32)
        # The model is randomly initilized, should be bad
        self.assertGreater(eval_loss, 50.0)
        # Expected correct preds out of a batch of 32 ~= 5
        self.assertLess(eval_accuracy, 5 * 32)
        self.assertEqual(self.experiment.model.get_age(), 0)
        self.summary_writer_mock.add_scalars.assert_not_called()

    def test_eval_full(self):
        _, _ = self.experiment.eval_full()
        self.summary_writer_mock.add_scalars.assert_any_call(
            'eval-loss', ANY, global_step=ANY)
        self.summary_writer_mock.add_scalars.assert_any_call(
            'eval-acc', ANY, global_step=ANY)

    def test_train_and_eval_full(self):
        self.experiment.set_batch_size(256)
        _, pre_train_eval_accuracy = self.experiment.eval_full()
        self.experiment.train_n_steps(len(self.experiment.train_loader) + 8)
        self.summary_writer_mock.add_scalars.assert_any_call(
            'train-loss', ANY, global_step=ANY)
        _, post_train_eval_accuracy = self.experiment.eval_full()
        self.assertGreater(post_train_eval_accuracy,
                           pre_train_eval_accuracy * 1.5)

    def test_train_loop_callbacks(self):
        loop_hook = mock.MagicMock()
        self.experiment.register_train_loop_callback(loop_hook)
        self.experiment.set_train_loop_clbk_freq(8)
        self.experiment.train_n_steps_with_eval_full(23)
        self.assertEqual(loop_hook.call_count, 3)


if __name__ == '__main__':
    unittest.main()
