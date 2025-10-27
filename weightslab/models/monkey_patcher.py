import types
import torch as th
import torch.nn as nn

from weightslab.layers.modules_with_ops import \
    NeuronWiseOperations, LayerWiseOperations
from weightslab.utils.logs import print, setup_logging
from weightslab.utils.tools import \
    check_learnable_module, extract_in_out_params, \
    get_module_device, rename_with_ops, model_op_neurons


def monkey_patch(module: nn.Module):
    """
    Dynamically injects LayerWiseOperations methods, wraps forward, and renames
    the module's displayed class name.
    """
    # Check if module is model type, sequential, list, and iterate until
    # module is type nn.module and no children.
    if len(list(module.children())) > 0 or \
            isinstance(module, nn.modules.container.Sequential) or \
            not isinstance(module, nn.Module) \
            or not check_learnable_module(module):
        return module

    # --- Step 1: Inject Mixin Methods (As before) ---
    # ... (Injection of NeuronWiseOperations and LayerWiseOperations methods)
    for name, method in vars(NeuronWiseOperations).items():
        if isinstance(method, types.FunctionType):
            setattr(module, name, types.MethodType(method, module))
    for name, method in vars(LayerWiseOperations).items():
        if isinstance(method, types.FunctionType):
            setattr(module, name, types.MethodType(method, module))

    # --- Step 2: Custom Initialization Setup ---
    in_dim, out_dim, in_name, out_name = extract_in_out_params(module)
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
    from weightslab.tests.test_utils import FashionCNNSequential

    # Setup prints
    setup_logging('DEBUG')
    print('Hello World')

    # 0. Test the forward pass to see the custom logic execute
    dummy_input = th.randn(1, 1, 28, 28)

    # 1. Instantiate the standard model
    model = FashionCNNSequential()
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

    print("\n--- Test Layer add_neurons")
    model_add_neurons(model)
    print(f'Inference test of the modified model is:\n{model(dummy_input)}')

    print('Bye World')
