import types
import torch as th
import torch.nn as nn

from weightslab.modules.modules_with_ops import \
    NeuronWiseOperations, LayerWiseOperations
from weightslab.utils.logs import print, setup_logging
from weightslab.utils.tools import \
    get_shape_attribute_from_module, \
    what_layer_type, extract_in_out_params, \
    get_module_device, rename_with_ops, \
    is_learnable_module


def monkey_patch(module: nn.Module):
    """
    Dynamically injects LayerWiseOperations methods, wraps forward, and renames
    the module's displayed class name.
    """
    module_type = what_layer_type(module)
    # Check if module is model type, sequential, list, and iterate until
    # module is type nn.module and no children.
    if len(list(module.children())) > 0 or \
            isinstance(module, nn.modules.container.Sequential) or \
            not isinstance(module, nn.Module):  # or not is_learnable_module(module):
        return module

    # --- Step 0: Extract Input and Output Parameters from layers ---
    in_dim, out_dim, in_name, out_name = extract_in_out_params(module)
    if in_dim is None and out_dim is None and in_name is None and out_name is None:
        return module

    # --- Step 1: Inject Mixin Methods (As before) ---
    # First, set layer type attribute
    setattr(module, 'layer_type', module_type)
    # ... (Injection of NeuronWiseOperations and LayerWiseOperations methods)
    # NeuronWiseOperations
    for name, method in vars(NeuronWiseOperations).items():
        if isinstance(method, types.FunctionType):
            setattr(module, name, types.MethodType(method, module))
    # LayerWiseOperations
    for name, method in vars(LayerWiseOperations).items():
        if isinstance(method, types.FunctionType):
            setattr(module, name, types.MethodType(method, module))

    # --- Step 2: Custom Initialization Setup ---
    try:
        module.__init__(
            in_neurons=in_dim,
            out_neurons=out_dim,
            device=get_module_device(module),
            module_name=module._get_name(),
            super_in_name=in_name,
            super_out_name=out_name
        )
    except Exception as e:
        print(f'Exception raised during custom init for"\
               f"{module.__class__.__name__}: {e}', level='ERROR')
        pass

    # --- Step 3: Monkey patch the module name with "with_ops" suffix
    rename_with_ops(module)

    # --- Step 4: Wrap the 'forward' Method ---
    original_forward = module.forward

    def wrapped_forward(self, input):
        activation_map = original_forward(input)
        output = self.perform_layer_op(
            activation_map=activation_map,
            data=input
        )
        return output
    module.forward = types.MethodType(wrapped_forward, module)  # Monkey patch

    return module


if __name__ == "__main__":
    from torch import nn
    from weightslab.backend.watcher_editor import WatcherEditor
    from weightslab.tests.torch_models import \
        TwoLayerUnflattenNet as Model

    # Setup prints
    setup_logging('DEBUG')
    print('Hello World')

    # 1. Instantiate the standard model
    model = Model()
    dummy_input = th.randn(model.input_shape)
    model(dummy_input)  # test inference
    model = WatcherEditor(model, dummy_input=dummy_input, print_graph=False)
    model(dummy_input)  # test inference
    print("--- Model Instantiated (Pre-Patching) ---")

    print("\n--- Running Patched Forward Pass ---")
    # The output of this forward pass will trigger your custom print statements
    # inside perform_layer_op and perform_neuron_op (if you uncomment them).
    output = model(dummy_input)

    print("\n--- Final Output Shape ---")
    print(output.shape)
