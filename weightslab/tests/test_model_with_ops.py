""" Tests for model logic."""
import os
import time
import warnings; warnings.filterwarnings("ignore")
import unittest
import tempfile
import torch as th
import torch.optim as opt

from os import path
from tqdm import trange
from torch.nn import functional as F

from torchvision import datasets as ds
from torchvision import transforms as T

from weightslab.components.tracking import TrackingMode
from weightslab.backend.model_interface import ModelInterface
from weightslab.tests.torch_models import FashionCNN as Model
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType


# Set Global Default Settings
th.manual_seed(42)  # Set SEED
TMP_DIR = tempfile.mkdtemp()


class NetworkWithOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        print(f"\n--- Start {self._testMethodName} ---\n")
        self.stamp = time.time()
        transform = T.Compose([T.ToTensor()])
        self.test_dir = TMP_DIR
        os.makedirs(self.test_dir, exist_ok=True)
        self.model = Model()
        self.dummy_network = ModelInterface(
            self.model,
            dummy_input=th.randn(self.model.input_shape),
            print_graph=False
        )

        self.dataset_train = ds.MNIST(
            os.path.join(self.test_dir, "data"),
            train=True,
            transform=transform,
            download=True
        )
        self.dataset_eval = ds.MNIST(
            os.path.join(self.test_dir, "data"),
            train=False,
            transform=transform
        )
        self.train_sample1 = self.dataset_train[0]
        self.train_sample2 = self.dataset_train[1]
        self.tracked_input = th.stack(
            [self.train_sample1[0], self.train_sample2[0]])

        self.train_loader = th.utils.data.DataLoader(
            self.dataset_train, batch_size=8, shuffle=True)

        self.eval_loader = th.utils.data.DataLoader(
            self.dataset_eval, batch_size=8)

        self.optimizer = opt.SGD(
            self.dummy_network.parameters(), lr=1e-3)

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    def _replicated_model(self):
        return ModelInterface(
            self.model,
            dummy_input=th.randn(self.model.input_shape),
            print_graph=False
        )

    def _train_one_epoch(self, cutoff: int | None = None):
        corrects = 0
        for idx, (image, label) in enumerate(self.train_loader):
            if cutoff and cutoff <= idx:
                break
            self.dummy_network.train()
            self.optimizer.zero_grad()
            output = self.dummy_network(image)
            prediction = output.argmax(dim=1, keepdim=True)
            losses_batch = F.cross_entropy(output, label, reduction='none')
            loss = th.mean(losses_batch)
            loss.backward()
            self.optimizer.step()
            corrects += prediction.eq(label.view_as(prediction)).sum().item()
        return corrects

    def _eval_one_epoch(self, cutoff: int | None = None):
        corrects = 0
        for idx, (image, label) in enumerate(self.eval_loader):
            if cutoff and cutoff <= idx:
                break
            self.dummy_network.eval()
            output = self.dummy_network(image)
            prediction = output.argmax(dim=1, keepdim=True)
            corrects += prediction.eq(label.view_as(prediction)).sum().item()
        return corrects

    def test_update_age_and_tracking_mode(self):
        self.dummy_network.maybe_update_age(self.tracked_input)
        self.assertEqual(self.dummy_network.get_age(), 0)
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        self.dummy_network.maybe_update_age(self.tracked_input)

    def test_store_and_load(self):
        # Forward
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        _ = self.dummy_network.forward(self.tracked_input)

        # Replicate model
        replicated_model = self._replicated_model()
        self.assertNotEqual(self.dummy_network, replicated_model)

        # Save
        state_dict_file_path = path.join(self.test_dir, 'mnist_model.txt')
        th.save(self.dummy_network.state_dict(), state_dict_file_path)

        # Load
        state_dict = th.load(state_dict_file_path)
        replicated_model.load_state_dict(state_dict, strict=False)
        self.assertEqual(self.dummy_network, replicated_model)

    def test_store_and_load_different_architectures(self):
        # Create a dummy model
        replicated_model = self._replicated_model()
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        _ = self.dummy_network.forward(self.tracked_input)
        self.assertNotEqual(self.dummy_network, replicated_model)

        # Operate
        self.dummy_network.operate(
            self.dummy_network.layers[0].get_module_id(),
            neuron_indices=2,
            op_type=ArchitectureNeuronsOpType.ADD
        )
        self.dummy_network.operate(
            -1,
            neuron_indices=set([0, 1, 2]),
            op_type=ArchitectureNeuronsOpType.PRUNE
        )

        # Store
        state_dict_file_path = path.join(self.test_dir, 'mnist_model.txt')
        th.save(self.dummy_network.state_dict(), state_dict_file_path)
        state_dict = th.load(state_dict_file_path)

        # Load
        replicated_model.load_state_dict(state_dict, strict=False)
        self.assertEqual(self.dummy_network, replicated_model)

    def test_train_add_neurons_train(self):
        # Set Tracker
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)

        # Train for like 10 epochs
        for _ in trange(1, desc="Training.."):
            corrects_first_epochs = self._train_one_epoch(cutoff=10)

        # Operate on the first layer - ADD
        with self.dummy_network as model:
            model.operate(
                0,
                -1,
                op_type=ArchitectureNeuronsOpType.ADD
            )

        # Train for another 10 epochs
        for _ in trange(1, desc="Training again.."):
            corrects_secnd_epochs = self._train_one_epoch(cutoff=10)

        # Check if the model has been trained correctly and any updated ?
        self.assertNotEqual(
            corrects_first_epochs, corrects_secnd_epochs)

    def test_train_prune(self):
        # Set Tracker
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)

        # Train for like 10 epochs
        for _ in trange(10, desc="Training.."):
            self._train_one_epoch(cutoff=20)

        # Evaluate
        corrects_first_epoch = self._eval_one_epoch(cutoff=50)

        # Operate on the first layer - PRUNE
        to_remove_ids = set()
        tracker = self.dummy_network.layers[0].train_dataset_tracker
        for neuron_id in range(tracker.number_of_neurons):
            frq_curr = tracker.get_neuron_stats(neuron_id)
            if frq_curr < 1.0:
                to_remove_ids.add(neuron_id)
        # If not neuron is low impact, then add the lowest impact one
        if not to_remove_ids:
            to_remove_ids.add(-3)
            to_remove_ids.add(-1)
        with self.dummy_network as model:
            model.operate(
                0,
                to_remove_ids,
                op_type=ArchitectureNeuronsOpType.PRUNE
            )

        # Evaluate
        corrects_after_prunning = self._eval_one_epoch(cutoff=50)

        # Check if the model has been trained correctly and any updated?
        self.assertNotEqual(
            corrects_first_epoch,
            corrects_after_prunning
        )

    def test_train_freeze(self):
        # Set Tracker
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)

        # Train for like 10 epochs
        for _ in trange(10, desc="Training.."):
            self._train_one_epoch(cutoff=20)

        # Operate on the first layer - FREEZE
        to_freeze_ids = {-1, -3}
        with self.dummy_network as model:
            model.operate(
                0,
                to_freeze_ids,
                op_type=ArchitectureNeuronsOpType.FREEZE
            )

        # Get weights sum
        init_learnable_weights_sum_value = th.sum(
            self.dummy_network.layers[0].weight[
                th.tensor([0, 2])
            ]
        )
        init_frozen_weights_sum_value = th.sum(
            self.dummy_network.layers[0].weight[
                th.tensor(list(to_freeze_ids))
            ]
        )

        # Train for like 5 epochs again
        for _ in trange(5, desc="Training again.."):
            self._train_one_epoch(cutoff=20)

        # Get weights sum
        learnable_weights_sum_value = th.sum(
            self.dummy_network.layers[0].weight[
                th.tensor([0, 2])
            ]
        )
        frozen_weights_sum_value = th.sum(
            self.dummy_network.layers[0].weight[
                th.tensor(list(to_freeze_ids))
            ]
        )

        # Check if the model has been trained correctly and any updated?
        self.assertAlmostEqual(
            init_frozen_weights_sum_value.item(),
            frozen_weights_sum_value.item()
        )
        self.assertNotEqual(
            init_learnable_weights_sum_value,
            learnable_weights_sum_value
        )

    def test_train_reset(self):
        # Set Tracker
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)

        # Get weights sum
        init_weights_sum_value = th.sum(
            self.dummy_network.layers[0].weight
        )

        # Train for like 10 epochs
        for _ in trange(10, desc="Training.."):
            self._train_one_epoch(cutoff=20)

        # Get weights sum
        weights_sum_value = th.sum(
            self.dummy_network.layers[0].weight
        )

        # Operate on the first layer - RESET
        to_freeze_ids = {-1, -3}
        with self.dummy_network as model:
            model.operate(
                0,
                to_freeze_ids,
                op_type=ArchitectureNeuronsOpType.RESET
            )

        after_reset_weights_sum_value = th.sum(
            self.dummy_network.layers[0].weight
        )

        # Train for like 5 epochs again
        for _ in trange(5, desc="Training again.."):
            self._train_one_epoch(cutoff=20)

        # Check that weights have been trained
        self.assertNotEqual(
            init_weights_sum_value.item(),
            weights_sum_value.item(),
        )

        # Check that weights have been reset
        self.assertNotEqual(
            weights_sum_value.item(),
            after_reset_weights_sum_value.item()
        )


if __name__ == '__main__':
    unittest.main()
