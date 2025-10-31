import os
import time
import inspect
import unittest
import warnings; warnings.filterwarnings("ignore")
import traceback
import torch as th
import importlib.util
import torch.nn as nn

from typing import Type

from weightslab.models.model_with_ops import ArchitectureNeuronsOpType
from weightslab.backend.watcher_editor import WatcherEditor
from weightslab.utils.tools import model_op_neurons, \
    get_model_parameters_neuronwise
from weightslab.utils.logs import print


# Set Global Default Settings
DEVICE = 'cpu' if not th.cuda.is_available() else 'cuda'
th.manual_seed(42)  # Set SEED


# 1. Utility function to dynamically find model classes (from previous answer)
def get_th_model_classes(file_path: str) -> list[Type[nn.Module]]:
    # ... (Implementation of get_th_model_classes) ...
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find module file at: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_classes = []

    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if issubclass(obj, nn.Module) and obj is not nn.Module and \
                    obj.__module__ == module_name:
                model_classes.append(obj)

    return model_classes


# 2. Define the path to your file containing the models
MODEL_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "torch_models.py"
)

# 3. Get the list of all model classes to parametrize the test
try:
    ALL_MODEL_CLASSES = get_th_model_classes(MODEL_FILE_PATH)
except FileNotFoundError:
    print(f"Could not find model file at: {MODEL_FILE_PATH}", level='ERROR')
except Exception as e:
    print(f"Error loading models from file: {e}", level='ERROR')

# If no models are found, you might want to skip or fail the test session
if not ALL_MODEL_CLASSES:
    print(
        f"No th.nn.Module classes found in {MODEL_FILE_PATH}",
        level='ERROR'
    )


# --- Test Class 1: Dynamic Inference and Shape Checks ---
class TestAllModelInference(unittest.TestCase):
    """
        Dynamically generated tests to check forward pass and output
        structure/shapes.
    """
    def setUp(self):
        print(f"\n--- Start {self._testMethodName} ---\n")
        self.stamp = time.time()

    def tearDown(self):
        """
        Runs AFTER every single test method (test_...).
        This is where you should place your final print('\n').
        """
        print(
            f"\n--- FINISHED: {self._testMethodName} in " +
            f"{time.time()-self.stamp}s ---\n")


def create_inference_test(ModelClass):
    """
        Helper to dynamically generate a test method
        for inference verification.
    """

    def test_inference(self, model, dummy_input, op=None):
        # Infer
        try:
            with th.no_grad():
                output = model(dummy_input)
        except Exception as e:
            print(f"Error during inference: {e}")
            output = None
            traceback.print_exc()
        # Test Inference
        self.assertNotEqual(
            output,
            None,
            f"[{model.get_name()}] Inference fails." + "" if op is None else
            f"\nOperation was {op}"
            ) if self is not None else None

    def model_test(self):
        # 1. Setup
        model = ModelClass()
        # # # Create dummy input tensor
        dummy_input = th.randn(model.input_shape).to(DEVICE)
        # # Interface the th model
        model = WatcherEditor(
            model,
            dummy_input=dummy_input,
            print_graph=False
        )
        model.to(DEVICE)
        model.eval()
        model_name = ModelClass.__name__
        print(f"\n--- Running Inference Test: {model_name} ---")

        # 2. Forward Pass Testing
        test_inference(self, model, dummy_input)

        # 3. Model Edition Testing
        # # Check ADD operation
        op = ArchitectureNeuronsOpType.ADD  # Get initial nb parameters
        layer_id = len(model.layers) // 2
        initial_nb_trainable_parameters = get_model_parameters_neuronwise(
            model
        )
        model_op_neurons(model, layer_id=layer_id, op=op)
        test_inference(self, model, dummy_input, op=op)
        # # Check nb trainable parameters (which should be greater)
        nb_trainable_parameters = get_model_parameters_neuronwise(model)
        self.assertGreater(
            nb_trainable_parameters,
            initial_nb_trainable_parameters,
            f"Neurons operation {op} didn\'t \
                generate new trainable parameters."
        ) if self is not None else None
        #
        # # Check PRUNING operation
        op = ArchitectureNeuronsOpType.PRUNE  # Get initial nb parameters
        layer_id = len(model.layers) // 2
        initial_nb_trainable_parameters = get_model_parameters_neuronwise(
            model
        )
        model_op_neurons(model, layer_id=layer_id, op=op)
        test_inference(self, model, dummy_input, op=op)
        # # Check nb trainable parameters (which should be greater)
        nb_trainable_parameters = get_model_parameters_neuronwise(model)
        self.assertLess(
            nb_trainable_parameters,
            initial_nb_trainable_parameters,
            f"Neurons operation {op} didn\'t \
                remove trainable parameters."
        ) if self is not None else None
        #
        # # Check RESET operation
        op = ArchitectureNeuronsOpType.RESET  # Get initial nb parameters
        layer_id = len(model.layers) // 2
        initial_nb_trainable_parameters = get_model_parameters_neuronwise(
            model
        )
        model_op_neurons(model, layer_id=layer_id, op=op)
        test_inference(self, model, dummy_input, op=op)
        # # Check nb trainable parameters (which should be greater)
        nb_trainable_parameters = get_model_parameters_neuronwise(model)
        self.assertEqual(
            nb_trainable_parameters,
            initial_nb_trainable_parameters,
            f"Neurons operation {op} change \
                the number of trainable parameters."
        ) if self is not None else None
        #
        # # Check FREEZE operation
        op = ArchitectureNeuronsOpType.FREEZE  # Get initial nb parameters
        initial_nb_trainable_parameters = get_model_parameters_neuronwise(
            model
        )
        model_op_neurons(model, layer_id=-1, op=op)
        test_inference(self, model, dummy_input, op=op)
        # # Check nb trainable parameters (which should be greater)
        nb_trainable_parameters = get_model_parameters_neuronwise(model)
        self.assertIn(
            nb_trainable_parameters,
            # TODO (GP): Estimate how many frozen neurons from res. connexion
            range(
                initial_nb_trainable_parameters,
                initial_nb_trainable_parameters-20,
                -1
            ),
            f"Neurons operation {op}: Wrong behavior with" +
            f"initially {initial_nb_trainable_parameters} trainable" +
            f" parameters, and now {nb_trainable_parameters}."
        )
        # # Unmasked the parameters
        model_op_neurons(model, layer_id=-1, op=op)
        test_inference(self, model, dummy_input, op=op)
        # # Check nb trainable parameters (which should be greater)
        nb_trainable_parameters = get_model_parameters_neuronwise(model)
        self.assertEqual(
            initial_nb_trainable_parameters,
            nb_trainable_parameters,
            "Unmasking parameters didn't restore the correct parameters."
        )
        #
        # # Check ALL operations on every layers
        print('Performing model parameters operations..', level='DEBUG')
        model_op_neurons(model)
        test_inference(self, model, dummy_input)

    model_test.__name__ = f'test_{ModelClass.__name__}_inference_check'

    return model_test


# Add dynamic tests to TestAllModelInference
for ModelClass in ALL_MODEL_CLASSES:
    test_method = create_inference_test(ModelClass)
    setattr(TestAllModelInference, test_method.__name__, test_method)


# --- Execution ---
if __name__ == '__main__':
    # Running unittest.main to execute all test classes
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
