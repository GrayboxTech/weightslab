"""Test for the core and main object of the graybox package."""
import os
import time
import unittest
import tempfile
import warnings; warnings.filterwarnings("ignore")
import torch as th

from unittest.mock import ANY
from unittest import mock

from torch import optim

from torchvision import transforms as T
from torchvision import datasets as ds

from torchmetrics import Accuracy

from weightslab.experiment.experiment import Experiment
from weightslab.tests.torch_models import FashionCNN


# Set Global Default Settings
th.manual_seed(42)  # Set SEED
TMP_DIR = tempfile.mkdtemp()
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")


class ExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.temporary_directory = TMP_DIR

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

        # Mocking the SummaryWriter
        self.summary_writer_mock = mock.Mock()
        self.summary_writer_mock.add_scalars = mock.MagicMock()

        # Init Experiment
        self.experiment = Experiment(
            model=model,
            optimizer_class=optim.Adam,
            train_dataset=data_train,
            metrics={
                'acc': Accuracy(task="multiclass", num_classes=10)
            },
            input_shape=model.input_shape,
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
        self.assertLess(eval_accuracy['acc'], 5 * 32)
        self.assertEqual(self.experiment.model.get_age(), 0)
        self.summary_writer_mock.add_scalars.assert_not_called()

    def test_eval_full(self):
        self.experiment.eval_full()
        self.summary_writer_mock.add_scalars.assert_any_call(
            'eval-loss',
            ANY,
            global_step=ANY
        )
        self.summary_writer_mock.add_scalars.assert_any_call(
            'eval-acc',
            ANY,
            global_step=ANY
        )

    def test_train_and_eval_full(self):
        self.experiment.set_batch_size(256)
        _, pre_train_eval_accuracy = self.experiment.eval_full()
        self.experiment.set_is_training(True)
        self.experiment.train_n_steps(len(self.experiment.train_loader) + 8)
        self.summary_writer_mock.add_scalars.assert_any_call(
            'train-loss', ANY, global_step=ANY)
        _, post_train_eval_accuracy = self.experiment.eval_full()
        self.assertNotAlmostEqual(
            post_train_eval_accuracy['acc'],
            pre_train_eval_accuracy['acc']
        )

    def test_train_loop_callbacks(self):
        loop_hook = mock.MagicMock()
        self.experiment.register_train_loop_callback(loop_hook)
        self.experiment.set_train_loop_clbk_freq(8)
        self.experiment.set_is_training(True)
        self.experiment.train_n_steps_with_eval_full(23)
        self.assertEqual(loop_hook.call_count, 23)


if __name__ == '__main__':
    unittest.main()
