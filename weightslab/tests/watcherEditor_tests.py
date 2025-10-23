import os
import pytest
import torch
import inspect
import importlib.util
import torch.nn as nn

from tqdm import tqdm
from typing import Type

from weightslab.backend.watcher_editor import WatcherEditor
from weightslab.utils.tools import model_add_neurons
from weightslab.utils.logs import print, setup_logging


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


@pytest.mark.parametrize("ModelClass", ALL_MODEL_CLASSES)
def test_model_loading_and_inference(ModelClass):
    """
    Checks that a given ModelClass can be instantiated and successfully
    completes a forward pass (inference) with dummy data.
    """

    # 1. Check Loading/Initialization
    try:
        # Instantiate the model
        model = ModelClass()
        DUMMY_INPUT_SHAPE = model.input_shape
    except Exception as e:
        pytest.fail(
            f"Model {ModelClass.__name__} failed to initialize. Error: {e}")

    # 2. Check Inference (Forward Pass)
    try:
        # Create dummy input tensor
        # NOTE: If your model requires a different input shape
        # e.g., only 1 channel;
        # you MUST customize the input_tensor based on the class type here.
        dummy_input = torch.randn(DUMMY_INPUT_SHAPE)

        # Convert the model - with_neuron_ops
        model = WatcherEditor(
            model,
            dummy_input=dummy_input,
            print_graph=False
        )

        # Run the forward pass
        with torch.no_grad():
            model(dummy_input)

        model_add_neurons(model)
        with torch.no_grad():
            out = model(dummy_input)

        print('#'+'-'*50+'\n')
        return out

    except RuntimeError as e:
        # This typically catches dimension mismatch errors;
        # the most common inference error.
        pytest.fail(f"Model {ModelClass.__name__} failed inference due" +
                    f"to a Runtime error (e.g., shape mismatch). Error: {e}")

    except Exception:
        pytest.fail(f"Model {ModelClass.__name__} failed inference." +
                    "Error: {e}")


if __name__ == "__main__":
    # Init logs
    setup_logging('INFO')

    err = list()
    for model_cl in tqdm(ALL_MODEL_CLASSES, desc="Testing.."):
        print(f'{model_cl}')
        out = test_model_loading_and_inference(model_cl)
        if out is None:
            err.append(model_cl)
