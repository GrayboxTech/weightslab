""" Tests for modules with operations. """
import time
import warnings; warnings.filterwarnings("ignore")
import tempfile
import unittest
import torch as th

from weightslab.models.monkey_patcher import monkey_patch
from weightslab.modules.modules_with_ops import \
    ArchitectureNeuronsOpType
from weightslab.utils.tools import \
    get_layer_trainable_parameters_neuronwise


# Set Global Default Settings
th.manual_seed(42)  # Set SEED


class LayerWiseOperationsTest(unittest.TestCase):
    def setUp(self) -> None:
        print(f"\n--- Start {self._testMethodName} ---\n")

        # Init Variables
        self.stamp = time.time()
        self.test_dir = tempfile.mkdtemp()
        self.all_layers = {}

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")

    # --- SETUP METHODS (To initialize layer instances) ---
    def _create_layers(self, device: str = 'cpu') -> None:
        # Use an input size that works across 1D/2D/3D (C_in=10, C_out=5)
        self.all_layers['Linear'] = th.nn.Linear(10, 5).to(device)

        self.all_layers['Conv1d'] = th.nn.Conv1d(10, 5, kernel_size=3).to(
            device
        )
        self.all_layers['Conv2d'] = th.nn.Conv2d(10, 5, kernel_size=3).to(
            device
        )
        self.all_layers['Conv3d'] = th.nn.Conv3d(10, 5, kernel_size=3).to(
            device
        )

        self.all_layers['ConvTranspose1d'] = th.nn.ConvTranspose1d(
            10,
            5,
            kernel_size=3
        ).to(device)
        self.all_layers['ConvTranspose2d'] = th.nn.ConvTranspose2d(
            10,
            5,
            kernel_size=3
        ).to(device)
        self.all_layers['ConvTranspose3d'] = th.nn.ConvTranspose3d(
            10,
            5,
            kernel_size=3
        ).to(device)

        # BatchNorm layers have 'num_features' which corresponds to the
        # channel count
        self.all_layers['BatchNorm1d'] = th.nn.BatchNorm1d(5).to(device)
        self.all_layers['BatchNorm2d'] = th.nn.BatchNorm2d(5).to(device)
        self.all_layers['BatchNorm3d'] = th.nn.BatchNorm3d(5).to(device)

        # Apply the required patch
        for layer in self.all_layers.values():
            layer.apply(monkey_patch)

    # --- CORE GENERIC HELPER ---
    def _test_operation_core(
            self,
            layer_key: str,
            op: ArchitectureNeuronsOpType,
            device: th.device
    ):

        self._create_layers(device=device)
        layer_instance = self.all_layers.get(layer_key)
        layer_instance.to(device)  # Update tracker device 

        if layer_instance is None:
            self.fail(f"Layer key '{layer_key}' not found in setup.")

        # Check if layer has module id
        # Will raise error if module_id is not set or function not found
        self.assertNotEqual(layer_instance.get_module_id(), -1)

        # Get initial state
        initial_nb_trainable_parameters = \
            get_layer_trainable_parameters_neuronwise(
                layer_instance
            )
        initial_nb_in_neurons = layer_instance.get_neurons(attr_name='in_neurons')
        initial_nb_out_neurons = layer_instance.get_neurons(attr_name='out_neurons')
        # Skip test if layer has no trainable params and
        # we test structural changes
        if initial_nb_trainable_parameters == 0 and \
                op in [
                    ArchitectureNeuronsOpType.ADD,
                    ArchitectureNeuronsOpType.PRUNE
                ]:
            return

        # --- ASSERTIONS ---
        op_name = op.name
        if op == ArchitectureNeuronsOpType.ADD:
            # --- Not Incoming ---
            # 1. Perform the Operation
            neuron_indices = {1, -2}
            layer_instance.operate(
                neuron_indices=neuron_indices,
                is_incoming=False,
                skip_initialization=False,
                neuron_operation=op
            )
            #
            # 2. Check Final State
            final_nb_trainable_parameters = \
                get_layer_trainable_parameters_neuronwise(
                    layer_instance
                )
            self.assertGreater(
                final_nb_trainable_parameters,
                initial_nb_trainable_parameters,
                f"[{layer_key}/{op_name}] failed to increase parameters." +
                f"Init:{initial_nb_trainable_parameters}," +
                f"Final:{final_nb_trainable_parameters}"
            )  # ADD must strictly increase the count
            self.assertEqual(
                layer_instance.get_neurons(attr_name='out_neurons'),
                initial_nb_out_neurons + len(neuron_indices),
                f"[{layer_key}/{op_name}] failed to increase out neurons" +
                "by 2." +
                f"Init:{initial_nb_out_neurons}," +
                f"Final:{layer_instance.get_neurons(attr_name='out_neurons')}"
            )  # ADD 2 neurons must increase the count by 2

            # --- Incoming ---
            if len(layer_instance.weight.shape) > 1:
                # 1. Perform the Operation
                neuron_indices = {1, -2}
                layer_instance.operate(
                    neuron_indices=neuron_indices,
                    is_incoming=True,
                    skip_initialization=True,
                    neuron_operation=op
                )
                #
                # 2. Check Final State
                final_nb_trainable_parameters = \
                    get_layer_trainable_parameters_neuronwise(
                        layer_instance
                    )
                self.assertGreater(
                    final_nb_trainable_parameters,
                    initial_nb_trainable_parameters,
                    f"[{layer_key}/{op_name}] failed to increase parameters." +
                    f"Init:{initial_nb_trainable_parameters}," +
                    f"Final:{final_nb_trainable_parameters}"
                )  # ADD must strictly increase the count
                self.assertEqual(
                    layer_instance.get_neurons(attr_name='in_neurons'),
                    initial_nb_in_neurons + len(neuron_indices),
                    f"[{layer_key}/{op_name}] failed to increase out neurons" +
                    "by 2." +
                    f"Init:{initial_nb_out_neurons}," +
                    f"Final:{layer_instance.get_neurons(attr_name='out_neurons')}"
                )  # ADD 2 neurons must increase the count by 2

        elif op == ArchitectureNeuronsOpType.PRUNE:
            # --- Not Incoming ---
            # 1. Perform the Operation
            neuron_indices = {1, -2}
            layer_instance.operate(
                neuron_indices=neuron_indices,
                is_incoming=False,
                skip_initialization=False,
                neuron_operation=op
            )
            #
            # 2. Check Final State
            final_nb_trainable_parameters = \
                get_layer_trainable_parameters_neuronwise(
                    layer_instance
                )
            self.assertLess(
                final_nb_trainable_parameters,
                initial_nb_trainable_parameters,
                f"[{layer_key}/{op_name}] failed to decrease parameters." +
                f"Init:{initial_nb_trainable_parameters}," +
                f"Final:{final_nb_trainable_parameters}"
            )  # PRUNE must strictly decrease the count
            self.assertEqual(
                layer_instance.get_neurons(attr_name='out_neurons'),
                initial_nb_out_neurons - len(neuron_indices),
                f"[{layer_key}/{op_name}] failed to decrease out neurons" +
                "by 2." +
                f"Init:{initial_nb_out_neurons}," +
                f"Final:{layer_instance.get_neurons(attr_name='out_neurons')}"
            )  # PRUNE 2 neurons must decrease the count by 2

            # --- Incoming ---
            if len(layer_instance.weight.shape) > 1:
                # 1. Perform the Operation
                neuron_indices = {1, -2}
                layer_instance.operate(
                    neuron_indices=neuron_indices,
                    is_incoming=True,
                    skip_initialization=True,
                    neuron_operation=op
                )
                #
                # 2. Check Final State
                final_nb_trainable_parameters = \
                    get_layer_trainable_parameters_neuronwise(
                        layer_instance
                    )
                self.assertLess(
                    final_nb_trainable_parameters,
                    initial_nb_trainable_parameters,
                    f"[{layer_key}/{op_name}] failed to decrease parameters." +
                    f"Init:{initial_nb_trainable_parameters}," +
                    f"Final:{final_nb_trainable_parameters}"
                )  # PRUNE must strictly decrease the count
                self.assertEqual(
                    layer_instance.get_neurons(attr_name='in_neurons'),
                    initial_nb_in_neurons - len(neuron_indices),
                    f"[{layer_key}/{op_name}] failed to decrease out neurons" +
                    "by 2." +
                    f"Init:{initial_nb_out_neurons}," +
                    f"Final:{layer_instance.get_neurons(attr_name='out_neurons')}"
                )  # PRUNE 2 neurons must decrease the count by 2

        elif op == ArchitectureNeuronsOpType.FREEZE:
            # --- Not Incoming ---
            # 1. Perform the Operation
            neuron_indices = {0, -1}
            layer_instance.operate(
                neuron_indices=neuron_indices,
                is_incoming=False,
                neuron_operation=op
            )
            #
            # 2. Check Final State
            final_nb_trainable_parameters = \
                get_layer_trainable_parameters_neuronwise(
                    layer_instance
                )
            self.assertLess(
                final_nb_trainable_parameters,
                initial_nb_trainable_parameters,
                f"[{layer_key}/{op_name}] failed to decrease parameters." +
                f"Init:{initial_nb_trainable_parameters}," +
                f"Final:{final_nb_trainable_parameters}"
            )  # FREEZE must strictly decrease the count
            #
            for tensor_name in layer_instance.learnable_tensors_name:
                # reverse neuron index
                neuron_indices_ = [
                    (
                        layer_instance.get_neurons(attr_name='out_neurons') + i
                    ) if i < 0 else i for i in neuron_indices
                ]
                neuron2lr = layer_instance.neuron_2_lr
                if tensor_name not in neuron2lr:
                    continue
                for index in neuron_indices_:
                    frozen_neuron = neuron2lr[
                        tensor_name
                    ][index]
                    self.assertEqual(
                        frozen_neuron,
                        0,
                        f"[{layer_key}/{op_name}/{tensor_name}] failed to" +
                        "freeze neuron {index}"
                    )
            #
            # UNFREEZE the neurons
            layer_instance.operate(
                neuron_indices=neuron_indices,
                is_incoming=False,
                neuron_operation=op
            )
            final_nb_trainable_parameters = \
                get_layer_trainable_parameters_neuronwise(
                    layer_instance
                )
            self.assertEqual(
                final_nb_trainable_parameters,
                initial_nb_trainable_parameters,
                f"[{layer_key}/{op_name}] failed to unfreeze parameters." +
                f"Init:{initial_nb_trainable_parameters}," +
                f"Final:{final_nb_trainable_parameters}"
            )  # UNFREEZE must match initial count
            #
            # FREEZE & UNFREEZE every neurons
            # # FREEZE
            layer_instance.operate(
                neuron_indices={},
                is_incoming=False,
                neuron_operation=op
            )
            final_nb_trainable_parameters = \
                get_layer_trainable_parameters_neuronwise(
                    layer_instance
                )
            self.assertLess(
                final_nb_trainable_parameters,
                initial_nb_trainable_parameters,
                f"[{layer_key}/{op_name}] failed to freeze every params." +
                f"Init:{layer_instance.get_neurons(attr_name='in_neurons')}," +
                f"Final:{final_nb_trainable_parameters}"
            )  # FREEZE every out neurons
            #
            # # UNFREEZE
            layer_instance.operate(
                neuron_indices={},
                is_incoming=False,
                neuron_operation=op
            )
            final_nb_trainable_parameters = \
                get_layer_trainable_parameters_neuronwise(
                    layer_instance
                )
            self.assertEqual(
                final_nb_trainable_parameters,
                initial_nb_trainable_parameters,
                f"[{layer_key}/{op_name}] failed to unfreeze every params." +
                f"Init:{initial_nb_trainable_parameters}," +
                f"Final:{final_nb_trainable_parameters}"
            )  # UNFREEZE every out neurons

            # --- Incoming ---
            if len(layer_instance.weight.shape) > 1:
                # 1. Perform the Operation
                neuron_indices = {0, -1}
                layer_instance.operate(
                    neuron_indices=neuron_indices,
                    is_incoming=True,
                    neuron_operation=op
                )
                #
                # 2. Check Final State
                final_nb_trainable_parameters = \
                    get_layer_trainable_parameters_neuronwise(
                        layer_instance
                    )
                self.assertLess(
                    final_nb_trainable_parameters,
                    initial_nb_trainable_parameters,
                    f"[{layer_key}/{op_name}] failed to decrease parameters." +
                    f"Init:{initial_nb_trainable_parameters}," +
                    f"Final:{final_nb_trainable_parameters}"
                )  # FREEZE must strictly decrease the count
                #
                for tensor_name in layer_instance.learnable_tensors_name:
                    # reverse neuron index
                    neuron_indices_ = [
                        (
                            layer_instance.get_neurons(attr_name='in_neurons') + i
                        ) if i < 0 else i for i in neuron_indices
                    ]
                    neuron2lr = layer_instance.incoming_neuron_2_lr
                    if tensor_name not in neuron2lr:
                        continue
                    for index in neuron_indices_:
                        frozen_neuron = neuron2lr[
                            tensor_name
                        ][index]
                        self.assertEqual(
                            frozen_neuron,
                            0,
                            f"[{layer_key}/{op_name}/{tensor_name}] failed " +
                            "to freeze neuron {index}"
                        )
                #
                # UNFREEZE the neurons
                layer_instance.operate(
                    neuron_indices=neuron_indices,
                    is_incoming=True,
                    skip_initialization=False,
                    neuron_operation=op
                )
                final_nb_trainable_parameters = \
                    get_layer_trainable_parameters_neuronwise(
                        layer_instance
                    )
                self.assertEqual(
                    final_nb_trainable_parameters,
                    initial_nb_trainable_parameters,
                    f"[{layer_key}/{op_name}] failed to unfreeze parameters." +
                    f"Init:{initial_nb_trainable_parameters}," +
                    f"Final:{final_nb_trainable_parameters}"
                )  # UNFREEZE must match initial count
                #
                # FREEZE & UNFREEZE every neurons
                # # FREEZE
                layer_instance.operate(
                    neuron_indices={},
                    is_incoming=True,
                    neuron_operation=op
                )
                final_nb_trainable_parameters = \
                    get_layer_trainable_parameters_neuronwise(
                        layer_instance
                    )
                self.assertLess(
                    final_nb_trainable_parameters,
                    initial_nb_trainable_parameters,
                    f"[{layer_key}/{op_name}] failed to freeze every params." +
                    f"Init:{layer_instance.get_neurons(attr_name='in_neurons')}," +
                    f"Final:{final_nb_trainable_parameters}"
                )  # FREEZE every out neurons
                #
                # # UNFREEZE
                layer_instance.operate(
                    neuron_indices={},
                    is_incoming=True,
                    neuron_operation=op
                )
                final_nb_trainable_parameters = \
                    get_layer_trainable_parameters_neuronwise(
                        layer_instance
                    )
                self.assertEqual(
                    final_nb_trainable_parameters,
                    initial_nb_trainable_parameters,
                    f"[{layer_key}/{op_name}] failed to unfreeze every" +
                    "params." +
                    f"Init:{initial_nb_trainable_parameters}," +
                    f"Final:{final_nb_trainable_parameters}"
                )  # UNFREEZE every out neurons

        elif op == ArchitectureNeuronsOpType.RESET:
            # RESET must preserve the number of parameters
            layer_instance.operate(
                neuron_indices={},
                is_incoming=False,
                neuron_operation=op
            )
            final_nb_trainable_parameters = \
                get_layer_trainable_parameters_neuronwise(
                    layer_instance
                )
            self.assertEqual(
                final_nb_trainable_parameters,
                initial_nb_trainable_parameters,
                f"[{layer_key}/{op_name}] changed parameter count (should " +
                f"be preserved). Init:{initial_nb_trainable_parameters}," +
                f"Final:{final_nb_trainable_parameters}"
            )
            # --- Incoming ---
            if len(layer_instance.weight.shape) > 1:
                # RESET must preserve the number of parameters
                initial_weights_sum = layer_instance.weight.sum()
                layer_instance.operate(
                    neuron_indices={},
                    is_incoming=False,
                    neuron_operation=op
                )
                final_nb_trainable_parameters = \
                    get_layer_trainable_parameters_neuronwise(
                        layer_instance
                    )
                self.assertEqual(
                    final_nb_trainable_parameters,
                    initial_nb_trainable_parameters,
                    f"[{layer_key}/{op_name}] changed parameter count (should " +
                    f"be preserved). Init:{initial_nb_trainable_parameters}," +
                    f"Final:{final_nb_trainable_parameters}"
                )
                self.assertNotEqual(
                    initial_weights_sum,
                    layer_instance.weight.sum(),
                    f"[{layer_key}/{op_name}] changed weights (should be " +
                    f"preserved). Init:{initial_weights_sum}," +
                    f"Final:{layer_instance.weight.sum()}"
                )


# --- DYNAMIC TEST GENERATION ---
LAYER_KEYS = [
    'Linear', 'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d'
]

OPERATIONS = [
    ArchitectureNeuronsOpType.ADD,
    ArchitectureNeuronsOpType.PRUNE,
    ArchitectureNeuronsOpType.RESET,
    ArchitectureNeuronsOpType.FREEZE
]


def _create_test_method(
        op: ArchitectureNeuronsOpType,
        layer_key: str,
        device: th.device
):
    """Factory function to create a named test method."""
    device_name = "cuda" if device.type == "cuda" else "cpu"

    def test_func(self):
        # Skip CUDA tests if CUDA is not available
        if device.type == 'cuda' and not th.cuda.is_available():
            self.skipTest("CUDA not available.")

        self._test_operation_core(layer_key, op, device)

    test_func.__name__ = f"test_{op.name}_{layer_key}_{device_name}"
    return test_func


# Dynamically attach all test functions to the class
for layer_key in LAYER_KEYS:
    for op in OPERATIONS:
        # 1. CPU Test
        test_method_cpu = _create_test_method(
            op,
            layer_key,
            th.device('cpu')
        )
        setattr(
            LayerWiseOperationsTest,
            test_method_cpu.__name__,
            test_method_cpu
        )

        # 2. CUDA Test
        if th.cuda.is_available():
            test_method_cuda = _create_test_method(
                op,
                layer_key,
                th.device('cuda')
            )
            setattr(
                LayerWiseOperationsTest,
                test_method_cuda.__name__,
                test_method_cuda
            )


if __name__ == '__main__':
    unittest.main()
