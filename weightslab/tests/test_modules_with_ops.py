""" Tests for modules with operations. """
import unittest
import tempfile

from copy import deepcopy

from os import path
from torch.nn import functional as F

from weightslab.tracking import TrackingMode
from weightslab.layers.modules_with_ops import (
    LinearWithNeuronOps,
    Conv2dWithNeuronOps,
    BatchNorm2dWithNeuronOps
)

import torch as th

class LinearWithNeuronsOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.module = LinearWithNeuronOps(in_features=4, out_features=4)
        with th.no_grad():
            self.module.weight.data = th.nn.Parameter(
                th.arange(0, 1, 1/16).view(4, 4), requires_grad=False)
            self.module.bias.data = th.nn.Parameter(
                th.arange(0, 1, 1/4).view(4), requires_grad=False)

        self.zeros_module = LinearWithNeuronOps(in_features=4, out_features=4)
        with th.no_grad():
            self.zeros_module.weight.data = th.nn.Parameter(
                th.zeros(4, 4), requires_grad=False)
            self.zeros_module.bias.data = th.nn.Parameter(
                th.zeros(4), requires_grad=False)

    def test_modules_got_ids(self):
        self.assertNotEqual(self.module.id, self.zeros_module.id)

    def reorder_outgoing_neurons(self, device):
        self.module.to(device)
        self.module.reorder([3, 1, 0, 2])
        module_test_input = th.Tensor([1.0000, 1.0000, 1.0000, 1.000])
        module_test_input = module_test_input.to(device)
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        expected_output = th.Tensor([4.1250, 1.6250, 0.3750, 2.8750])
        expected_output = expected_output.to(device)

        self.assertTrue(th.equal(module_test_output, expected_output))

    def test_reorder_outgoing_neurons_cpu(self):
        self.reorder_outgoing_neurons(th.device('cpu'))

    def test_reorder_outgoing_neurons_cuda(self):
        self.reorder_outgoing_neurons(th.device('cuda'))

    def reorder_incoming_neurons(self, device):
        self.module.to(device)
        self.module.reorder_incoming_neurons([3, 1, 0, 2])
        module_test_input = th.Tensor([1.0000, 1.0000, 1.0000, 1.000])
        module_test_input = module_test_input.to(device)
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        expected_output = th.Tensor([0.3750, 1.6250, 2.8750, 4.1250]).to(
            device)
        self.assertTrue(th.equal(module_test_output, expected_output))

    def test_reorder_incoming_neurons_cpu(self):
        self.reorder_incoming_neurons(th.device('cpu'))

    def test_reorder_incoming_neurons_cuda(self):
        self.reorder_incoming_neurons(th.device('cuda'))

    def prune_neurons(self, device):
        self.module.to(device)
        module_test_input = th.Tensor([1.0000, 1.0000, 1.0000, 1.0000])
        module_test_input = module_test_input.to(device)
        module_test_output_before_prune = self.module(module_test_input)
        self.assertTrue(module_test_output_before_prune.shape[0] == 4)
        expected_output_before_prune = th.Tensor(
            [0.3750, 1.6250, 2.8750, 4.1250])
        expected_output_before_prune = expected_output_before_prune.to(device)
        self.assertTrue(
            th.equal(
                expected_output_before_prune,
                module_test_output_before_prune))
        self.module.prune(set([1, 3]))
        # TODO(rotaru): re-evaluate whether this is good practice. Proly' not.
        # after_prune_wght = th.Tensor([
        #     [0.0000, 0.0625, 0.1250, 0.1875],
        #     [0.5000, 0.5625, 0.6250, 0.6875],])
        # after_prune_bias = th.Tensor([0.0000, 0.5000])
        # self.assertTrue(th.equal(self.module.weight, after_prune_wght))
        # self.assertTrue(th.equal(self.module.bias, after_prune_bias))
        module_test_output_after_prune = self.module(module_test_input)
        self.assertTrue(module_test_output_after_prune.shape[0] == 2)
        self.assertTrue(th.equal(module_test_output_after_prune,
                                 th.Tensor([0.3750, 2.8750]).to(device)))

    def test_prune_neurons_cpu(self):
        self.prune_neurons(th.device('cpu'))

    def test_prune_neurons_cuda(self):
        self.prune_neurons(th.device('cuda'))

    def prune_incoming_neurons(self, device):
        self.module.to(device)
        module_test_input_before_prune = th.Tensor(
            [1.0000, 1.0000, 1.0000, 1.000]).to(device)
        module_test_output_before_prune = self.module(
            module_test_input_before_prune)
        self.assertTrue(module_test_output_before_prune.shape[0] == 4)
        self.assertTrue(
            th.equal(
                module_test_output_before_prune,
                th.Tensor([0.3750, 1.6250, 2.8750, 4.1250]).to(device)))

        self.module.prune_incoming_neurons(set([1, 3]))
        module_test_input_after_prune = th.Tensor([1.0000, 1.0000]).to(device)

        module_test_output_after_prune = self.module(
            module_test_input_after_prune)
        self.assertTrue(module_test_output_after_prune.shape[0] == 4)
        self.assertTrue(
            th.equal(
                module_test_output_after_prune,
                th.Tensor([0.1250, 0.8750, 1.6250, 2.3750]).to(device)))

    def test_prune_incoming_neurons_cpu(self):
        self.prune_incoming_neurons(th.device('cpu'))

    def test_prune_incoming_neurons_cuda(self):
        self.prune_incoming_neurons(th.device('cuda'))

    def reset_neurons(self, device):
        self.module.to(device)
        module_test_input = th.Tensor([1.0000, 1.0000, 1.0000, 1.000])
        module_test_input = module_test_input.to(device)
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 2.8750, 4.1250]).to(device)))
        self.module.reset(set([1, 3]), skip_initialization=True)
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 0.0000, 2.8750, 0.0000]).to(device)))

    def test_reset_neurons_cpu(self):
        self.reset_neurons(th.device('cpu'))

    def test_reset_neuron_cuda(self):
        self.reset_neurons(th.device('cuda'))

    @th.no_grad
    def perturb_neurons(self, device):
        self.module.to(device)

        before_perturb_neuron_weights = deepcopy(self.module.weight[1])
        self.module.reset(set([1,]), perturbation_ratio=0.1)
        after_perturb_neuron_weights = self.module.weight[1]

        absolute_diff_neuron_weight = (
            after_perturb_neuron_weights - before_perturb_neuron_weights).abs()
        relative_diff_neuron_weight = (
            absolute_diff_neuron_weight / before_perturb_neuron_weights)
        self.assertTrue(th.all(relative_diff_neuron_weight <= 0.1001))

    def test_perturb_neuron_cpu(self):
        self.perturb_neurons(th.device('cpu'))

    def test_perturb_neuron_cuda(self):
        self.perturb_neurons(th.device('cuda'))

    def reset_incoming_neurons(self, device):
        self.module.to(device)
        self.module.reset_incoming_neurons([1], skip_initialization=True)
        module_test_input = th.Tensor([1.0000, 0.5000, 0.5000, 1.000])
        module_test_input = module_test_input.to(device)
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        expected_output = th.Tensor([0.2500, 1.1250, 2.0000, 2.8750]).to(
            device)
        self.assertTrue(th.equal(module_test_output, expected_output))

    def test_reset_incoming_neurons_cpu(self):
        self.reset_incoming_neurons(th.device('cpu'))

    def test_reset_incoming_neurons_cuda(self):
        self.reset_incoming_neurons(th.device('cuda'))

    @th.no_grad
    def perturb_incoming_neurons(self, device):
        self.module.to(device)
        before_perturb_neuron_weights = deepcopy(self.module.weight[:, 1])
        self.module.reset_incoming_neurons(set([1,]),
                                           perturbation_ratio=0.1)
        after_perturb_neuron_weights = self.module.weight[:, 1]
        absolute_diff_neuron_weight = (
            after_perturb_neuron_weights - before_perturb_neuron_weights).abs()
        relative_diff_neuron_weight = (
            absolute_diff_neuron_weight / before_perturb_neuron_weights)
        self.assertTrue(th.all(relative_diff_neuron_weight <= 0.10001))

    def test_perturb_incoming_neuron_cpu(self):
        self.perturb_incoming_neurons(th.device('cpu'))

    def test_perturb_incoming_neuron_cuda(self):
        self.perturb_incoming_neurons(th.device('cuda'))

    def add_neurons(self, device):
        self.module.to(device)
        module_test_input = th.Tensor([1.0000, 1.0000, 1.0000, 1.000])
        module_test_input = module_test_input.to(device)
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 2.8750, 4.1250]).to(device)))

        self.module.add_neurons(2, skip_initialization=True)

        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 6)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor(
                    [0.3750, 1.6250, 2.8750, 4.1250, 0.0000, 0.0000]
                ).to(device)))

    def test_add_neurons_cpu(self):
        self.add_neurons(th.device('cpu'))

    def test_add_neurons_cuda(self):
        self.add_neurons(th.device('cuda'))

    def is_incoming(self, device):
        self.module.to(device)
        module_test_input = th.Tensor([1.0000, 1.0000, 1.0000, 1.0000]).to(
            device)
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 2.8750, 4.1250]).to(device)))
        self.module.add_incoming_neurons(2, skip_initialization=True)
        module_test_input_after_incoming_neurons_adding = th.Tensor(
            [1.0000, 1.0000, 1.0000, 1.000, 0.5000, 0.5000]).to(device)
        module_test_output = self.module(
                module_test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 4)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 2.8750, 4.1250]).to(device)))

    def test_incoming_neurons_cpu(self):
        self.is_incoming(th.device('cpu'))

    def test_incoming_neurons_cuda(self):
        self.is_incoming(th.device('cuda'))

    def test_all_ops(self):
        # Basic
        module_test_input = th.Tensor([1.0000, 1.0000, 1.0000, 1.000])
        module_test_output = self.module(module_test_input)
        self.assertTrue(module_test_output.shape[0] == 4)
        self.assertTrue(th.equal(module_test_output,
                                 th.Tensor([0.3750, 1.6250, 2.8750, 4.1250])))

        # Add incoming neurons.
        self.module.add_incoming_neurons(2, skip_initialization=True)

        module_test_input_after_incoming_neurons_adding = th.Tensor(
            [1.0000, 1.0000, 1.0000, 1.000, 0.5000, 0.5000])
        module_test_output = self.module(
                module_test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 4)
        self.assertTrue(th.equal(module_test_output,
                                 th.Tensor([0.3750, 1.6250, 2.8750, 4.1250])))

        # Add outgoing neurons.
        self.module.add_neurons(1, skip_initialization=True)

        module_test_input_after_incoming_neurons_adding = th.Tensor(
            [1.0000, 1.0000, 1.0000, 1.000, 0.5000, 0.5000])
        module_test_output = self.module(
            module_test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 5)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 2.8750, 4.1250, 0.0000])))

        # Reset neurons.
        self.module.reset(set([2, ]), skip_initialization=True)

        module_test_output = self.module(
                module_test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 5)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 0.0000, 4.1250, 0.0000])))

        # Reorder incoming neurons.
        self.module.reorder_incoming_neurons([4, 5, 0, 1, 2, 3])

        module_test_output = self.module(
                module_test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 5)
        self.assertTrue(
            not th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 0.0000, 4.1250, 0.0000])))
        reordered_module_test_input_after_incoming_neurons_adding = th.Tensor(
            [0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000])
        module_test_output = self.module(
            reordered_module_test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 5)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 0.0000, 4.1250, 0.0000])))

        # Reorder outgoing neurons.
        self.module.reorder([2, 4, 0, 1, 3])
        reordered_module_test_input_after_incoming_neurons_adding = th.Tensor(
            [0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000])
        module_test_output = self.module(
            reordered_module_test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 5)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.0000, 0.0000, 0.3750, 1.6250, 4.1250])))

        # Prune incoming neurons.
        self.module.prune_incoming_neurons(set([0, 1]))
        test_input_after_incoming_neurons_adding = th.Tensor(
            [1.0000, 1.0000, 1.0000, 1.0000])
        module_test_output = self.module(
            test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 5)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.0000, 0.0000, 0.3750, 1.6250, 4.1250])))

        # Prune neurons.
        self.module.prune(set([0, 1]))
        test_input_after_incoming_neurons_adding = th.Tensor(
            [1.0000, 1.0000, 1.0000, 1.0000])
        module_test_output = self.module(
            test_input_after_incoming_neurons_adding)
        self.assertTrue(module_test_output.shape[0] == 3)
        self.assertTrue(
            th.equal(
                module_test_output,
                th.Tensor([0.3750, 1.6250, 4.1250])))

    def store_and_load(self, device):
        self.module.to(device)
        self.zeros_module.to(device)
        self.module.set_tracking_mode(TrackingMode.EVAL)
        # Pass some tensors through the module to have some stats in the
        # counters.
        # tracked_input = TrackedInput(
        #     input_batch=th.Tensor([[1.0000, 1.0000, 1.000, 1.000]]).to(device),
        #     in_id_batch=th.Tensor([[16]]).to(device),
        #     label_batch=th.Tensor([[3]]).to(device)
        # )
        tracked_input = th.Tensor([[1.0000, 1.0000, 1.000, 1.000]]).to(device)
        _ = self.module(tracked_input)
        self.assertNotEqual(self.module, self.zeros_module)

        state_dict_file_path = path.join(self.test_dir, 'linear_module.txt')
        th.save(self.module.state_dict(), state_dict_file_path)
        state_dict = th.load(state_dict_file_path)
        self.zeros_module.load_state_dict(state_dict, strict=False)
        self.assertEqual(self.module, self.zeros_module)

    def test_store_and_load_cpu(self):
        self.store_and_load(th.device('cpu'))

    def test_store_and_load_cuda(self):
        self.store_and_load(th.device('cuda'))


class Conv2dWithNeuronOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.module = Conv2dWithNeuronOps(
            in_channels=2, out_channels=2, kernel_size=2)
        with th.no_grad():
            self.module.weight.data = th.nn.Parameter(
                th.tensor([
                            # Here the first neuron weights begin
                            [[[-1.0000, -0.8750],
                             [-0.7500, -0.6250]],
                            [[-0.5000, -0.3750],
                             [-0.2500, -0.1250]]],
                            #  Here the second neuron weights begin
                           [[[0.0000, 0.1250],
                             [0.2500, 0.3750]],
                            [[0.5000, 0.6250],
                             [0.7500, 0.8750]]]]), requires_grad=False)
            self.module.bias.data = th.nn.Parameter(
                th.arange(0, 1, 1/2).view(2), requires_grad=False)

        self.zeros_module = Conv2dWithNeuronOps(
            in_channels=2, out_channels=2, kernel_size=2)
        with th.no_grad():
            self.zeros_module.weight.data = th.nn.Parameter(
                th.zeros(2, 2, 2, 2), requires_grad=False)
            self.zeros_module.bias.data = th.nn.Parameter(
                th.zeros(2), requires_grad=False)

        self.test_input = th.tensor([[
            [1.0000, 0.9688, 0.9375, 0.9062],
            [0.8750, 0.8438, 0.8125, 0.7812],
            [0.7500, 0.7188, 0.6875, 0.6562],
            [0.6250, 0.5938, 0.5625, 0.5312]
            ], [
            [0.5000, 0.4688, 0.4375, 0.4062],
            [0.3750, 0.3438, 0.3125, 0.2812],
            [0.2500, 0.2188, 0.1875, 0.1562],
            [0.1250, 0.0938, 0.0625, 0.0312]]])

        self.test_input_first_channel = th.tensor([[
            [1.0000, 0.9688, 0.9375, 0.9062],
            [0.8750, 0.8438, 0.8125, 0.7812],
            [0.7500, 0.7188, 0.6875, 0.6562],
            [0.6250, 0.5938, 0.5625, 0.5312]
        ]])

        self.test_input_reordered_channels = th.tensor([[
            [0.5000, 0.4688, 0.4375, 0.4062],
            [0.3750, 0.3438, 0.3125, 0.2812],
            [0.2500, 0.2188, 0.1875, 0.1562],
            [0.1250, 0.0938, 0.0625, 0.0312]
            ], [
            [1.0000, 0.9688, 0.9375, 0.9062],
            [0.8750, 0.8438, 0.8125, 0.7812],
            [0.7500, 0.7188, 0.6875, 0.6562],
            [0.6250, 0.5938, 0.5625, 0.5312]]]
        )

        self.test_input_with_3_channels = th.tensor([[
            [1.0000, 0.9688, 0.9375, 0.9062],
            [0.8750, 0.8438, 0.8125, 0.7812],
            [0.7500, 0.7188, 0.6875, 0.6562],
            [0.6250, 0.5938, 0.5625, 0.5312]
            ], [
            [0.5000, 0.4688, 0.4375, 0.4062],
            [0.3750, 0.3438, 0.3125, 0.2812],
            [0.2500, 0.2188, 0.1875, 0.1562],
            [0.1250, 0.0938, 0.0625, 0.0312]
            ], [
            [0.0000, -0.0625, -0.1250, -0.1875],
            [-0.2500, -0.3125, -0.3750, -0.4375],
            [-0.5000, -0.5625, -0.6250, -0.6875],
            [-0.7500, -0.8125, -0.8750, -0.9375]
            ]]
        )

    def reorder_outgoing_neurons(self, device):
        test_input = self.test_input.to(device)
        self.module.to(device)
        test_output = self.module(test_input)
        self.assertTrue(th.all(th.isclose(test_output, th.tensor([[
            [-3.5938, -3.4532, -3.3124],
            [-3.0313, -2.8907, -2.7499],
            [-2.4688, -2.3282, -2.1874]
            ], [
            [2.2814,  2.1719,  2.0624],
            [1.8439,  1.7344,  1.6249],
            [1.4064,  1.2969,  1.1874]]]).to(device), rtol=1e-03, atol=1e-03)))

        self.module.reorder([1, 0])
        reordered_test_output = self.module(test_input)

        self.assertTrue(
            th.all(
                th.isclose(
                    reordered_test_output,
                    th.tensor([[
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]
                        ], [
                        [-3.5938, -3.4532, -3.3124],
                        [-3.0313, -2.8907, -2.7499],
                        [-2.4688, -2.3282, -2.1874]]]
                    ).to(device),
                    rtol=1e-03, atol=1e-03)))

    def test_reorder_outgoing_neurons_cpu(self):
        self.reorder_outgoing_neurons(th.device('cpu'))

    def test_reorder_outgoing_neurons_cuda(self):
        self.reorder_outgoing_neurons(th.device('cuda'))

    def reorder_incoming_neurons(self, device):
        test_input = self.test_input.to(device)
        self.module.to(device)
        test_output = self.module(test_input)
        self.assertTrue(
            th.all(
                th.isclose(
                    test_output,
                    th.tensor([[
                        [-3.5938, -3.4532, -3.3124],
                        [-3.0313, -2.8907, -2.7499],
                        [-2.4688, -2.3282, -2.1874]
                        ], [
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

        self.module.reorder_incoming_neurons([1, 0])
        test_input_reordered_channels = self.test_input_reordered_channels.to(
            device)
        reordered_test_output = self.module(test_input_reordered_channels)
        self.assertTrue(
            th.all(
                th.isclose(
                    reordered_test_output,
                    th.tensor([[
                        [-3.5939, -3.4532, -3.3124],
                        [-3.0314, -2.8907, -2.7499],
                        [-2.4689, -2.3282, -2.1874]
                        ], [
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

    def test_reorder_incoming_neurons_cpu(self):
        self.reorder_incoming_neurons(th.device('cpu'))

    def test_reorder_incoming_neurons_cuda(self):
        self.reorder_incoming_neurons(th.device('cuda'))

    def prune_neurons(self, device):
        test_input = self.test_input.to(device)
        self.module.to(device)
        module_test_output_before_prune = self.module(test_input)
        self.assertTrue(module_test_output_before_prune.shape[0] == 2)
        self.assertTrue(
            th.all(
                th.isclose(
                    module_test_output_before_prune,
                    th.tensor([[
                        [-3.5938, -3.4532, -3.3124],
                        [-3.0313, -2.8907, -2.7499],
                        [-2.4688, -2.3282, -2.1874]
                        ], [
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

        self.module.prune(set([0]))
        module_test_output_after_prune = self.module(test_input)
        self.assertTrue(module_test_output_after_prune.shape[0] == 1)
        self.assertTrue(
            th.all(
                th.isclose(
                    module_test_output_after_prune,
                    th.tensor([[
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

    def test_prune_neurons_cpu(self):
        self.prune_neurons(th.device('cpu'))

    def test_prune_neurons_cuda(self):
        self.prune_neurons(th.device('cuda'))

    def prune_incoming_neurons(self, device):
        self.module.to(device)
        test_input = self.test_input.to(device)
        module_test_output_before_prune = self.module(test_input)
        self.assertTrue(module_test_output_before_prune.shape[0] == 2)
        self.assertTrue(
            th.all(
                th.isclose(
                    module_test_output_before_prune,
                    th.tensor([[
                        [-3.5938, -3.4532, -3.3124],
                        [-3.0313, -2.8907, -2.7499],
                        [-2.4688, -2.3282, -2.1874]
                        ], [
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

        self.module.prune_incoming_neurons(set([0]))
        test_input_first_channel = self.test_input_first_channel.to(device)
        module_test_output_after_prune = self.module(test_input_first_channel)
        self.assertTrue(module_test_output_after_prune.shape[0] == 2)
        self.assertTrue(
            th.all(
                th.isclose(
                    module_test_output_after_prune,
                    th.tensor([[
                        [-1.1875, -1.1485, -1.1093],
                        [-1.0313, -0.9922, -0.9531],
                        [-0.8750, -0.8360, -0.7968]
                        ], [
                        [3.0001,  2.9141,  2.8281],
                        [2.6563,  2.5704,  2.4843],
                        [2.3126,  2.2266,  2.1406]]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

    def test_prune_incoming_neurons_cpu(self):
        self.prune_incoming_neurons(th.device('cpu'))

    def test_prune_incoming_neurons_cuda(self):
        self.prune_incoming_neurons(th.device('cuda'))

    def add_neurons(self, device):
        self.module.to(device)
        test_input = self.test_input.to(device)
        self.module.add_neurons(1, skip_initialization=True)
        module_test_output = self.module(test_input)
        self.assertTrue(module_test_output.shape[0] == 3)
        self.assertTrue(
            th.all(
                th.isclose(
                    module_test_output,
                    th.tensor([[
                        [-3.5938, -3.4532, -3.3124],
                        [-3.0313, -2.8907, -2.7499],
                        [-2.4688, -2.3282, -2.1874]
                        ], [
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]
                        ], [
                        [0.0000,  0.0000,  0.0000],
                        [0.0000,  0.0000,  0.0000],
                        [0.0000,  0.0000,  0.0000]
                        ]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

    def test_add_neurons_cpu(self):
        self.add_neurons(th.device('cpu'))

    def test_add_neurons_cuda(self):
        self.add_neurons(th.device('cuda'))

    def add_incoming_neurons(self, device):
        self.module.to(device)
        test_input_with_3_channels = self.test_input_with_3_channels.to(
            device)
        self.module.add_incoming_neurons(1, skip_initialization=True)
        module_test_output = self.module(test_input_with_3_channels)
        self.assertTrue(module_test_output.shape[0] == 2)
        # Nothing should change in the output since the newly added weights
        # are 0.
        self.assertTrue(
            th.all(
                th.isclose(
                    module_test_output,
                    th.tensor([[
                        [-3.5938, -3.4532, -3.3124],
                        [-3.0313, -2.8907, -2.7499],
                        [-2.4688, -2.3282, -2.1874]
                        ], [
                        [2.2814,  2.1719,  2.0624],
                        [1.8439,  1.7344,  1.6249],
                        [1.4064,  1.2969,  1.1874]
                        ]]).to(device),
                    rtol=1e-03,
                    atol=1e-03)))

    def test_add_incoming_neurons_cpu(self):
        self.add_incoming_neurons(th.device('cpu'))

    def test_add_incoming_neurons_cuda(self):
        self.add_incoming_neurons(th.device('cuda'))

    def test_reset(self):
        self.module.reset({1}, skip_initialization=True)
        module_test_output = self.module(self.test_input)
        self.assertTrue(module_test_output.shape[0] == 2)
        self.assertTrue(th.all(th.isclose(
                module_test_output,
                th.tensor([[
                    [-3.5938, -3.4532, -3.3124],
                    [-3.0313, -2.8907, -2.7499],
                    [-2.4688, -2.3282, -2.1874]
                    ], [
                    [0.0000,  0.0000,  0.0000],
                    [0.0000,  0.0000,  0.0000],
                    [0.0000,  0.0000,  0.0000]
                    ]]), rtol=1e-03, atol=1e-03)))

    @th.no_grad
    def perturb_neurons(self, device):
        self.module.to(device)

        before_perturb_neuron_weights = deepcopy(self.module.weight[1])
        self.module.reset(set([1,]), perturbation_ratio=0.1)
        after_perturb_neuron_weights = self.module.weight[1]

        absolute_diff_neuron_weight = (
            after_perturb_neuron_weights - before_perturb_neuron_weights).abs()
        relative_diff_neuron_weight = (
            absolute_diff_neuron_weight /
            (before_perturb_neuron_weights + 0.001))
        self.assertTrue(th.all(relative_diff_neuron_weight <= 0.1))

    def test_perturb_neuron_cpu(self):
        self.perturb_neurons(th.device('cpu'))

    def test_perturb_neuron_cuda(self):
        self.perturb_neurons(th.device('cuda'))

    def reset_incoming_neurons(self, device):
        self.module.to(device)
        test_input = self.test_input.to(device)
        self.module.reset_incoming_neurons({1}, skip_initialization=True)
        module_test_output = self.module(test_input)
        self.assertTrue(module_test_output.shape[0] == 2)

        # print(module_test_output)
        self.assertTrue(th.all(th.isclose(
                module_test_output,
                th.tensor([[
                    [-3.0313, -2.9298, -2.8280],
                    [-2.6251, -2.5235, -2.4218],
                    [-2.2188, -2.1173, -2.0155]
                    ], [
                    [1.1563,  1.1328,  1.1093],
                    [1.0625,  1.0391,  1.0156],
                    [0.9688,  0.9453,  0.9218]]]).to(device),
                rtol=1e-03,
                atol=1e-03)))

    def test_reset_incoming_neurons_cpu(self):
        self.reset_incoming_neurons(th.device('cpu'))

    def test_reset_incoming_neurons_cuda(self):
        self.reset_incoming_neurons(th.device('cuda'))

    @th.no_grad
    def perturb_incoming_neurons(self, device):
        self.module.to(device)
        before_perturb_neuron_weights = deepcopy(self.module.weight[:, 1])
        self.module.reset_incoming_neurons(set([1,]), perturbation_ratio=0.1)
        after_perturb_neuron_weights = self.module.weight[:, 1]
        absolute_diff_neuron_weight = (
            after_perturb_neuron_weights - before_perturb_neuron_weights).abs()
        relative_diff_neuron_weight = (
            absolute_diff_neuron_weight / before_perturb_neuron_weights)
        self.assertTrue(th.all(relative_diff_neuron_weight <= 0.10001))

    def test_perturb_incoming_neurons_cpu(self):
        self.perturb_incoming_neurons(th.device('cpu'))

    def test_perturb_incoming_neurons_cuda(self):
        self.perturb_incoming_neurons(th.device('cuda'))

    def test_all_ops(self):
        pass

    def store_and_load(self, device):
        self.module.to(device)
        self.zeros_module.to(device)
        self.module.set_tracking_mode(TrackingMode.TRAIN)
        _ = self.module(self.test_input.unsqueeze(0).to(device))
        self.assertNotEqual(self.module, self.zeros_module)

        state_dict_file_path = path.join(self.test_dir, 'conv2d_module.txt')
        th.save(self.module.state_dict(), state_dict_file_path)
        state_dict = th.load(state_dict_file_path)
        self.zeros_module.load_state_dict(state_dict, strict=False)

        self.assertEqual(self.module, self.zeros_module)

    def test_store_and_load_cpu(self):
        self.store_and_load(th.device('cpu'))

    def test_store_and_load_cuda(self):
        self.store_and_load(th.device('cuda'))


class BatchNorm2dWithNeuronOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        # self.test_dir = tempfile.mkdtemp()
        self.module = BatchNorm2dWithNeuronOps(2)

        self.test_input = th.tensor([[
            [1.0000, 0.9688, 0.9375, 0.9062],
            [0.8750, 0.8438, 0.8125, 0.7812],
            [0.7500, 0.7188, 0.6875, 0.6562],
            [0.6250, 0.5938, 0.5625, 0.5312]
            ], [
            [0.5000, 0.4688, 0.4375, 0.4062],
            [0.3750, 0.3438, 0.3125, 0.2812],
            [0.2500, 0.2188, 0.1875, 0.1562],
            [0.1250, 0.0938, 0.0625, 0.0312]]])

        self.test_input_3chans = th.tensor([[
            [1.0000, 0.9688, 0.9375, 0.9062],
            [0.8750, 0.8438, 0.8125, 0.7812],
            [0.7500, 0.7188, 0.6875, 0.6562],
            [0.6250, 0.5938, 0.5625, 0.5312]
            ], [
            [0.5000, 0.4688, 0.4375, 0.4062],
            [0.3750, 0.3438, 0.3125, 0.2812],
            [0.2500, 0.2188, 0.1875, 0.1562],
            [0.1250, 0.0938, 0.0625, 0.0312]
            ], [
            [0.5555, 0.4444, 0.4444, 0.4444],
            [0.3333, 0.3333, 0.3333, 0.2222],
            [0.2222, 0.2222, 0.1111, 0.1111],
            [0.1111, 0.0000, 0.0000, 0.0000]]])

        self.test_input_reorderd = th.tensor([[
            [0.5000, 0.4688, 0.4375, 0.4062],
            [0.3750, 0.3438, 0.3125, 0.2812],
            [0.2500, 0.2188, 0.1875, 0.1562],
            [0.1250, 0.0938, 0.0625, 0.0312],
        ], [
            [1.0000, 0.9688, 0.9375, 0.9062],
            [0.8750, 0.8438, 0.8125, 0.7812],
            [0.7500, 0.7188, 0.6875, 0.6562],
            [0.6250, 0.5938, 0.5625, 0.5312]
        ]])

    def reorder_neurons(self, device):
        # TODO(rotaur): revisit this, this seems ... SUS'
        self.module.to(device)

        test_output = self.module(
            self.test_input.unsqueeze(0).to(device)
        )
        self.module.reorder([1, 0])
        test_output_bn_reordered = self.module(
            self.test_input_reorderd.unsqueeze(0).to(device)
        )

        self.assertTrue(
            th.all(th.isclose(
                test_output_bn_reordered, test_output, rtol=1e-03,
                atol=1e-03
            )),
            "reodered bn on reordered input does not yield the same results")

    def test_reorder_neurons_cpu(self):
        self.reorder_neurons(th.device('cpu'))

    def test_reorder_neurons_gpu(self):
        self.reorder_neurons(th.device('cuda'))

    def prune(self, device):
        self.module.to(device)
        test_output = self.module(
            self.test_input.unsqueeze(0).to(device)
        )
        self.module.prune(set([0]))
        test_output_pruned = self.module(
            self.test_input[1:].unsqueeze(0).to(device)
        )

        self.assertTrue(
            th.all(th.isclose(
                test_output_pruned, test_output[1:], rtol=1e-03,
                atol=1e-03
            )),
            "pruned bn does not yield the same results")

    def test_prune_cpu(self):
        self.prune(th.device('cpu'))

    def test_prune_gpu(self):
        self.prune(th.device('cuda'))

    def add_neurons(self, device):
        self.module.to(device)
        test_output = self.module(
            self.test_input.unsqueeze(0).to(device)
        )
        self.module.add_neurons(1)
        test_output_added = self.module(
            self.test_input_3chans.unsqueeze(0).to(device)
        )

        self.assertEqual(test_output_added.shape[1], 3)

        self.assertTrue(
            th.all(th.isclose(
                test_output, test_output_added[:, :2], rtol=1e-03,
                atol=1e-03
            )),
            "added bn does not yield the same results")

    def test_add_neurons_cpu(self):
        self.add_neurons(th.device('cpu'))

    def test_add_neurons_gpu(self):
        self.add_neurons(th.device('cuda'))


if __name__ == '__main__':
    unittest.main()