import types
import torch.nn as nn

from weightslab.modules.modules_with_ops import \
    NeuronWiseOperations, LayerWiseOperations
from weightslab.utils.logs import print
from weightslab.utils.tools import \
    what_layer_type, extract_in_out_params, \
    get_module_device, rename_with_ops


def monkey_patch(module: nn.Module):
    """
        Dynamically injects LayerWiseOperations methods, wraps forward, and
        renames the module's displayed class name.
        Args:
            module (nn.Module): The module to be patched.
    """
    # Get module type, i.e., 'Linear', 'Conv2d', or unflattened layer.
    module_type = what_layer_type(module)

    # Check if module is model type, sequential, list, and iterate until
    # module is type nn.module and no children.
    if len(list(module.children())) > 0 or \
            isinstance(module, nn.modules.container.Sequential) or \
            not isinstance(module, nn.Module):
        return module

    # --- Step 0: Extract Input and Output Parameters from layers ---
    in_dim, out_dim, in_name, out_name = extract_in_out_params(module)
    if in_dim is None and out_dim is None and in_name is None and out_name is None:
        return module

    # --- Step 1: Inject Mixin Methods (As before) ---
    # # First, set layer type attribute
    setattr(module, 'layer_type', module_type)
    # # NeuronWiseOperations
    for name, method in vars(NeuronWiseOperations).items():
        if isinstance(method, types.FunctionType):
            setattr(module, name, types.MethodType(method, module))
    # # LayerWiseOperations
    for name, method in vars(LayerWiseOperations).items():
        if isinstance(method, types.FunctionType):
            setattr(module, name, types.MethodType(method, module))

    # --- Step 2: Initiliaze module inteface ---
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

    # --- Step 3: Update module name with "with_ops" suffix
    rename_with_ops(module)

    # --- Step 4: Wrap the 'forward' Method ---
    def wrapped_forward(self, input):
        activation_map = module.forward(input)
        output = self.perform_layer_op(
            activation_map=activation_map,
            data=input
        )
        return output
    module.forward = types.MethodType(wrapped_forward, module)  # Monkey patch

    return module
