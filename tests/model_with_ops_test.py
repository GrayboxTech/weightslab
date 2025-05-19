""" Tests for model logic."""
import unittest
import tempfile

import torch as th
import torch.optim as opt

from torch.nn import functional as F
from torchvision import datasets as ds
from torchvision import transforms as T
from tqdm import tqdm
from os import path
from weightslab.tracking import TrackingMode

from .test_utils import MNISTModel


th.manual_seed(0)

class NetworkWithOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        transform = T.Compose([T.ToTensor()])
        self.test_dir = tempfile.mkdtemp()
        self.dummy_network = MNISTModel()
        self.dataset_train = ds.MNIST(
            "../data", train=True, transform=transform, download=True)
        self.dataset_eval = ds.MNIST(
            "../data", train=False, transform=transform)
        self.train_sample1 = self.dataset_train[0]
        self.train_sample2 = self.dataset_train[1]
        self.tracked_input = th.stack(
            [self.train_sample1[0], self.train_sample2[0]])

        self.train_loader = th.utils.data.DataLoader(
            self.dataset_train, batch_size=512, shuffle=True)

        self.eval_loader = th.utils.data.DataLoader(
            self.dataset_eval, batch_size=512)

        self.optimizer = opt.SGD(
            self.dummy_network.parameters(), lr=1e-3)

    def test_update_age_and_tracking_mode(self):
        self.dummy_network.maybe_update_age(self.tracked_input)
        self.assertEqual(self.dummy_network.get_age(), 0)
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        self.dummy_network.maybe_update_age(self.tracked_input)
        self.assertEqual(self.dummy_network.get_age(), 2)
        
    def test_store_and_load(self):
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        _ = self.dummy_network.forward(self.tracked_input)
        replicated_model = MNISTModel()
        self.assertNotEqual(self.dummy_network, replicated_model)
        state_dict_file_path = path.join(self.test_dir, 'mnist_model.txt')
        th.save(self.dummy_network.state_dict(), state_dict_file_path)
        state_dict = th.load(state_dict_file_path)
        replicated_model.load_state_dict(state_dict, strict=False)
        self.assertEqual(self.dummy_network, replicated_model)

    def test_store_and_load_different_architectures(self):
        replicated_model = MNISTModel()
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        _ = self.dummy_network.forward(self.tracked_input)
        self.assertNotEqual(self.dummy_network, replicated_model)

        self.dummy_network.add_neurons(id(self.dummy_network.conv0), 2)
        self.dummy_network.prune(
            id(self.dummy_network.linear0), set([0, 1, 2]))

        state_dict_file_path = path.join(self.test_dir, 'mnist_model.txt')
        th.save(self.dummy_network.state_dict(), state_dict_file_path)
        state_dict = th.load(state_dict_file_path)

        replicated_model.load_state_dict(state_dict, strict=False)
        self.assertEqual(self.dummy_network, replicated_model)

    def train_one_epoch(self, cutoff: int | None = None):
        corrects = 0
        for idx, (image, label) in tqdm(enumerate(self.train_loader)):
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

    def eval_one_epoch(self, cutoff: int | None = None):
        corrects = 0
        for idx, (image, label) in tqdm(enumerate(self.eval_loader)):
            if cutoff and cutoff <= idx:
                break
            self.dummy_network.eval()
            output = self.dummy_network(image)
            prediction = output.argmax(dim=1, keepdim=True)
            corrects += prediction.eq(label.view_as(prediction)).sum().item()
        return corrects

    def test_train_add_neurons_train(self):
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        corrects_first_epochs = self.train_one_epoch(cutoff=200)
        self.dummy_network.add_neurons(id(self.dummy_network.conv0), 10)
        corrects_secnd_epochs = self.train_one_epoch(cutoff=200)
        self.assertNotEqual(
            corrects_first_epochs, corrects_secnd_epochs)

    def test_train_reorder(self):
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        self.train_one_epoch()
        corrects_first_epoch = self.eval_one_epoch()
        self.dummy_network.reorder_neurons_by_trigger_rate(
            id(self.dummy_network.conv0))
        corrects_after_reorder = self.eval_one_epoch()
        self.assertEqual(corrects_first_epoch, corrects_after_reorder)

    def test_train_prune(self):
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        # Train for like 10 epochs
        for _ in range(10):
            self.train_one_epoch()

        corrects_first_epoch = self.eval_one_epoch()
        self.dummy_network.reorder_neurons_by_trigger_rate(
            id(self.dummy_network.conv0))

        to_remove_ids = set()
        tracker = self.dummy_network.conv0.train_dataset_tracker
        for neuron_id in range(tracker.number_of_neurons):
            frq_curr = tracker.get_neuron_stats(neuron_id)
            # print("n: ", neuron_id, "\t: ", frq_curr)
            if frq_curr < 1.0:
                to_remove_ids.add(neuron_id)
        # If not neuron is low impact, then add the lowest impact one
        if not to_remove_ids:
            to_remove_ids.add(15)
        self.dummy_network.prune(
            id(self.dummy_network.conv0), to_remove_ids)
        corrects_after_prunning = self.eval_one_epoch()
        self.assertNotEqual(
            corrects_first_epoch, corrects_after_prunning)

    def test_train_reset(self):
        self.dummy_network.set_tracking_mode(TrackingMode.TRAIN)
        # Train for like 10 epochs
        for _ in range(10):
            self.train_one_epoch()

        corrects_first_epoch = self.eval_one_epoch()
        self.dummy_network.reorder_neurons_by_trigger_rate(
            id(self.dummy_network.conv0))

        to_reinit_ids = set()
        tracker = self.dummy_network.conv0.train_dataset_tracker
        for neuron_id in range(tracker.number_of_neurons):
            frq_curr = tracker.get_neuron_stats(neuron_id)
            # print("n: ", neuron_id, "\t: ", frq_curr)
            if frq_curr < 1.0:
                to_reinit_ids.add(neuron_id)
        # If not neuron is low impact, then add the lowest impact one
        if not to_reinit_ids:
            to_reinit_ids.add(15)
        self.dummy_network.reinit_neurons(
            id(self.dummy_network.conv0), to_reinit_ids)
        for _ in range(5):
            self.train_one_epoch()
        corrects_after_reinit = self.eval_one_epoch()
        self.assertLessEqual(
            corrects_first_epoch, corrects_after_reinit)
        
    def test_zerofy(self):
        print("Initial neuron count:", self.dummy_network.conv0.neuron_count)
        layer_id = self.dummy_network.conv0.get_module_id()
        self.dummy_network.add_neurons(layer_id, 1)
        new_neuron_id = self.dummy_network.conv0.neuron_count - 1

        # self.dummy_network.conv0.train_dataset_tracker.set_neuron_age(new_neuron_id, 0)
        age = self.dummy_network.conv0.train_dataset_tracker.triggrs_by_neuron[new_neuron_id] = 0
        for nid in range(self.dummy_network.linear0.neuron_count):
            # self.dummy_network.linear0.train_dataset_tracker.set_neuron_age(nid, 5)
            age = self.dummy_network.linear0.train_dataset_tracker.triggrs_by_neuron[nid] = 5

        self.dummy_network.zerofy(layer_id)

        for old_id in range(self.dummy_network.linear0.neuron_count):
            weight_val = self.dummy_network.linear0.weight[old_id, new_neuron_id].item()
            self.assertEqual(weight_val, 0.0)


if __name__ == '__main__':
    unittest.main()

#test    
