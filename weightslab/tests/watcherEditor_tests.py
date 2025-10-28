import os
import pytest
import unittest
import torch
import inspect
import importlib.util
import torch.nn as nn
import warnings

from tqdm import tqdm
from typing import Type

from weightslab.models.model_with_ops import ArchitectureNeuronsOpType
from weightslab.backend.watcher_editor import WatcherEditor
from weightslab.utils.tools import model_op_neurons, \
    get_nb_parameters
from weightslab.utils.logs import print


NB_LAYERS_2_TEST = 20
# This suppresses all warnings globally
warnings.filterwarnings("ignore")


# 1. Utility function to dynamically find model classes (from previous answer)
def get_torch_model_classes(file_path: str) -> list[Type[nn.Module]]:
    # ... (Implementation of get_torch_model_classes) ...
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
    "test_utils.py"
)

# 3. Get the list of all model classes to parametrize the test
try:
    ALL_MODEL_CLASSES = get_torch_model_classes(MODEL_FILE_PATH)
except FileNotFoundError:
    pytest.fail(f"Could not find model file at: {MODEL_FILE_PATH}")
except Exception as e:
    pytest.fail(f"Error loading models from file: {e}")

# If no models are found, you might want to skip or fail the test session
if not ALL_MODEL_CLASSES:
    pytest.skip(
        f"No torch.nn.Module classes found in {MODEL_FILE_PATH}",
        allow_module_level=True
    )


# @pytest.mark.parametrize("ModelClass", ALL_MODEL_CLASSES)
# def test_model_loading_and_inference(ModelClass):
#     """
#     Checks that a given ModelClass can be instantiated and successfully
#     completes a forward pass (inference) with dummy data.
#     """

#     # 1. Check Loading/Initialization
#     try:
#         # Instantiate the model
#         model = ModelClass()
#         DUMMY_INPUT_SHAPE = model.input_shape
#     except Exception as e:
#         pytest.fail(
#             f"Model {ModelClass.__name__} failed to initialize. Error: {e}")

#     # 2. Check Inference (Forward Pass)
#     try:
#         # Create dummy input tensor
#         # NOTE: If your model requires a different input shape
#         # e.g., only 1 channel;
#         # you MUST customize the input_tensor based on the class type here.
#         dummy_input = torch.randn(DUMMY_INPUT_SHAPE)

#         # Convert the model - with_neuron_ops
#         model = WatcherEditor(
#             model,
#             dummy_input=dummy_input,
#             print_graph=False
#         )
#         get_nb_parameters(model)

#         # Run the forward pass
#         with torch.no_grad():
#             model(dummy_input)

#         print('Performing model parameters operations..', level='DEBUG')
#         model_op_neurons(model, dummy_input=None)
#         with torch.no_grad():
#             out = model(dummy_input)
#         get_nb_parameters(model)
#         return out

#     except RuntimeError as e:
#         # This typically catches dimension mismatch errors;
#         # the most common inference error.
#         pytest.fail(f"Model {ModelClass.__name__} failed inference due" +
#                     f"to a Runtime error (e.g., shape mismatch). Error: {e}")

#     except Exception:
#         pytest.fail(f"Model {ModelClass.__name__} failed inference." +
#                     "Error: {e}")


# if __name__ == "__main__":
#     import warnings
#     warnings.filterwarnings("ignore")

#     # Init logs
#     setup_logging('INFO')

#     err = list()
#     # ALL_MODEL_CLASSES = ALL_MODEL_CLASSES[5:9]
#     for model_cl in tqdm(ALL_MODEL_CLASSES, desc="Testing.."):
#         print(f'{model_cl}', level='DEBUG')
#         out = test_model_loading_and_inference(model_cl)
#         if out is None:
#             err.append(model_cl)


# --- Test Class 1: Dynamic Inference and Shape Checks ---
class TestAllModelInference(unittest.TestCase):
    """
        Dynamically generated tests to check forward pass and output
        structure/shapes.
    """
    pass  # Tests added dynamically below


def create_inference_test(ModelClass):
    """
        Helper to dynamically generate a test method
        for inference verification.
    """
    model_test.__name__ = f'test_{ModelClass.__name__}_inference_check'
    return model_test



def test_inference(self, model, dummy_input, op=None):
    # Infer
    try:
        with torch.no_grad():
            output = model(dummy_input)
    except Exception:
        output = None

    # Test Inference
    self.assertNotEqual(
        output,
        None,
        f"[{model.get_name()}] Inference fails." + "" if op is None else
        f"\nOperation was {op}"
        ) if self is not None else None


def model_test(self, ModelClass):
    # 1. Setup
    model = ModelClass()
    # # # Create dummy input tensor
    dummy_input = torch.randn(model.input_shape)
    # # Interface the torch model
    model = WatcherEditor(
        model,
        dummy_input=dummy_input,
        print_graph=False
    )
    model.eval()
    model_name = ModelClass.__name__
    print(f"\n--- Running Inference Test: {model_name} ---")

    # 2. Forward Pass Testing
    test_inference(self, model, dummy_input)

    # 3. Model Edition Testing
    # # Check ADD operation
    op = ArchitectureNeuronsOpType.ADD  # Get initial nb parameters
    layer_id = len(model.layers) // 2
    initial_nb_trainable_parameters = get_nb_parameters(model)
    model_op_neurons(model, layer_id=layer_id, op=op)
    test_inference(self, model, dummy_input, op=op)
    # # Check nb trainable parameters (which should be greater)
    nb_trainable_parameters = get_nb_parameters(model)
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
    initial_nb_trainable_parameters = get_nb_parameters(model)
    model_op_neurons(model, layer_id=layer_id, op=op)
    test_inference(self, model, dummy_input, op=op)
    # # Check nb trainable parameters (which should be greater)
    nb_trainable_parameters = get_nb_parameters(model)
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
    initial_nb_trainable_parameters = get_nb_parameters(model)
    model_op_neurons(model, layer_id=layer_id, op=op)
    test_inference(self, model, dummy_input, op=op)
    # # Check nb trainable parameters (which should be greater)
    nb_trainable_parameters = get_nb_parameters(model)
    self.assertEqual(
        nb_trainable_parameters,
        initial_nb_trainable_parameters,
        f"Neurons operation {op} change \
            the number of trainable parameters."
    ) if self is not None else None
    #
    # # Check FREEZE operation
    op = ArchitectureNeuronsOpType.FREEZE  # Get initial nb parameters
    layer_id = len(model.layers) // 2
    initial_nb_trainable_parameters = get_nb_parameters(model)
    model_op_neurons(model, layer_id=layer_id, op=op)
    test_inference(self, model, dummy_input, op=op)
    # # Check nb trainable parameters (which should be greater)
    nb_trainable_parameters = get_nb_parameters(model)
    # self.assertNotEqual(
    #     nb_trainable_parameters,
    #     initial_nb_trainable_parameters,
    #     f"Neurons operation {op} change \
    #         the number of trainable parameters."
    # )  # TODO (GP): Trainable neurons count based on tensor grad.
    #
    # # Check ALL operations on every layers
    print('Performing model parameters operations..', level='DEBUG')
    model_op_neurons(model)
    test_inference(self, model, dummy_input)


# Add dynamic tests to TestAllModelInference
for ModelClass in tqdm(
    ALL_MODEL_CLASSES,
    total=len(ALL_MODEL_CLASSES),
    desc="Testing model inference and parameters modification.."
):
    test_method = create_inference_test(ModelClass)
    setattr(TestAllModelInference, test_method.__name__, test_method)


# --- Execution ---
if __name__ == '__main__':
    # # Running unittest.main to execute all test classes
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)

    # Manual tests
    for ModelClass in tqdm(
        ALL_MODEL_CLASSES[:],
        total=len(ALL_MODEL_CLASSES),
        desc="Testing model inference and parameters modification.."
    ):
        model_test(None, ModelClass)
