import types
import inspect
import torch as th
import torch.nn as nn
from torch.fx import GraphModule, Node
from typing import Tuple, Optional, List, Any, Type

from weightslab.utils.modules_dependencies import DepType
from weightslab.utils.logs import print


# ----------------------------------------------------------------------------
# -------------------------- Utils Functions ---------------------------------
# ----------------------------------------------------------------------------
def check_learnable_module(module: nn.Module) -> bool:
    """
    Checks if a module is a learnable nn.Module with parameters that have grad.
    """
    # Check if it's a th.nn.Module instance
    if not isinstance(module, nn.Module):
        return

    has_learnable_params = False
    # Iterate over the parameters to check if any requires gradient
    for _, param in module.named_parameters():
        if param.requires_grad:
            has_learnable_params = True
            break
    return has_learnable_params


def extract_in_out_params(module: nn.Module) -> List[int | str]:
    """
    Detects and returns the primary input and output dimension parameters
    for a given PyTorch module instance, based on commmon templates.
    For single weight tensor (e.g., nn.BatchNorm), in=out.
    """

    # 1. Like Linear Layers use 'features' template
    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
        in_dim = module.in_features
        in_name = "in_features"
        out_dim = module.out_features
        out_name = "out_features"
        return in_dim, out_dim, in_name, out_name

    # 2. Like Convolutional Layers (Conv1d, Conv2d, Conv3d) use 'channels'
    # template
    if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
        in_dim = module.in_channels
        in_name = "in_channels"
        out_dim = module.out_channels
        out_name = "out_channels"
        return in_dim, out_dim, in_name, out_name

    # 3. Like BatchNorm Layers use 'num_features' template
    if hasattr(module, 'num_features'):
        # For BatchNorm, in_dim and out_dim are the same
        in_dim = module.num_features
        in_name = "num_features"
        out_dim = module.num_features
        out_name = "num_features"
        return in_dim, out_dim, in_name, out_name

    # 4. Catch all or return None for non-parameterized layers
    return None, None, None, None


def get_children(module: nn.Module):
    """
        Get module children (other modules).
    """
    if is_module_with_ops(module):
        return [module]
    flatt_children = []
    for child in module.children():
        flatt_children.extend(get_children(child))

    return flatt_children


def get_module_device(module: nn.Module) -> th.device:
    """
    Retrieves the device (CPU or CUDA) of a th.nn.Module.
    """
    # Use next(module.parameters()) to get the first parameter tensor
    try:
        # Check the device of the first parameter found
        return next(module.parameters()).device
    except StopIteration:
        # If the module has no parameters (e.g., nn.ReLU, nn.Sequential,
        # or containers),
        # it doesn't have a device of its own. It defaults to the CPU.
        # This is the safest fallback, though you might need context for
        # the exact device.
        return th.device("cpu")


def rename_with_ops(module: nn.Module) -> nn.Module:
    """
        Add WithNeuronOps string to each nn.module name.
    """
    # 1. Store the original class name
    original_name = module._get_name()

    # 2. Define the new name
    new_name = f"{original_name}WithNeuronOps"

    # 3. Create a custom method to return the new name
    def new_get_name(self):
        return new_name

    # 4. Monkey patch the module's _get_name() method
    # This is the function PyTorch calls when printing the model hierarchy.
    module._get_name = types.MethodType(new_get_name, module)


def is_module_with_ops(module: nn.Module) -> bool:
    return "WithNeuronOps" in module._get_name()


def get_all_classes_from_module(module):
    """
        Dynamically retrieves all class objects defined within a given module.
    """
    classes = []
    # Use inspect.getmembers to look at all attributes of the module
    for name, obj in inspect.getmembers(module):
        # Check if the object is a class and not an internal/private object
        if inspect.isclass(obj) and not name.startswith('_'):
            classes.append(obj)
    return tuple(classes)  # isinstance takes a tuple of classes


# Helper to retrieve module instance by its submodule path
def get_module_by_name(model: nn.Module, name: str) -> nn.Module | None:
    """
        Safely retrieves a module instance from the model based on its FX
        target name.
    """
    try:
        return model.get_submodule(name)
    except AttributeError:
        return getattr(model, name, None)


def is_module_learnable(module: Optional[Any]) -> bool:
    """
    Check if the module has learnable parameters.
    """
    return hasattr(module, 'weight') and module.weight is not None


def is_feature_producer(module: Optional[Any]) -> bool:
    """
    Checks if a module is a primary feature producer by checking for
    the presence of common input and output dimension attributes (in_*, out_*).
    This generalizes the check beyond specific nn.Module classes.
    """
    if module is None:
        return False

    # Check for convolutional-style feature definition
    # (e.g., in_channels, out_channels)
    has_conv_features = hasattr(module, 'in_channels') and \
        hasattr(module, 'out_channels')

    # Check for linear-style feature definition
    # (e.g., in_features, out_features)
    has_linear_features = hasattr(module, 'in_features') and \
        hasattr(module, 'out_features')

    # Any module defining both an input and an output feature dimension is
    # considered a "producer"
    return has_conv_features or has_linear_features


def get_feature_channel_size(node: Node) -> Optional[int]:
    """
        Retrieves the channel size (dimension 1) of the tensor output by the
        node.
    """
    if 'tensor_meta' in node.meta and node.meta['tensor_meta'] is not None:
        meta = node.meta['tensor_meta']
        if isinstance(meta, th.Tensor) or \
                isinstance(meta, th.fx.passes.shape_prop.TensorMetadata):
            # Assumes N, C, H, W or N, C, L format (channel is dim 1)
            if len(meta.shape) > 1:
                return int(meta.shape[1])
    return None


def has_multi_in_out(module):
    in_out_c = (
        hasattr(module, 'in_channels') or
        hasattr(module, 'in_features')
    ) and (
        hasattr(module, 'out_channels') or
        hasattr(module, 'out_features')
    )
    if not in_out_c:
        if hasattr(module, 'weight') or hasattr(module, 'weights'):
            return True
    return False


def make_safelist(x):
    return [x] if not isinstance(x, list) else x


def generate_graph_dependencies(
        model: nn.Module,
        traced_graph: GraphModule) -> \
            List[Tuple[nn.Module, nn.Module, DepType]]:
    """
        Infers dependencies from the traced graph, explicitly marking
        structuralSAME and INCOMING constraints.
    """
    dependencies = []

    # Map to store the last *structural module* (instance) that produced the
    # output for a given node.
    # This map is crucial for implementing the "pass-through" logic for
    # non-structural layers.
    node_to_module = {}
    bypass = []

    # Iterate over the nodes in the graph to find sources
    for node in traced_graph.graph.nodes:

        current_module = None
        if node.op == 'call_module':
            # Get current module from node
            current_module = get_module_by_name(model, node.target)
            if node.name in bypass:
                current_module.bypass = True  # bypass strategy for recursive update dependencies, like bypass = true for __add__ but false for cat

            # Find the input source node that came from a tracked module
            source_node = next(
                (arg for arg in node.args if isinstance(arg, th.fx.Node)),
                None
            )
            source_modules = node_to_module.get(source_node) if source_node \
                else None

            # --- 1. Dependency Creation (from last Structural Module to
            # current Structural Module) ---
            # A dependency edge (A -> B) is only created if B (current_module)
            # is a structural layer.
            is_dest_structural = is_feature_producer(current_module)
            is_learnable = is_module_learnable(current_module)
            if source_modules:
                for source_module in source_modules:
                    if source_module is not None and \
                            (is_dest_structural or is_learnable):
                        # 1.1. Determine Dependency Type based on Shape
                        # (for Pruning)
                        dep_type = DepType.INCOMING
                        source_out_channels = get_feature_channel_size(
                            source_node
                        )
                        dest_out_channels = get_feature_channel_size(node)
                        # 1.2. Check if current module should be target SAME
                        # path. It's a specific case where current module has
                        # in==out shapes
                        same_dep = has_multi_in_out(current_module)

                        # Check for SAME constraint (requires source to be a
                        # producer)
                        if is_feature_producer(source_module) and \
                                source_out_channels is not None and \
                                dest_out_channels is not None:
                            if same_dep and \
                                    source_out_channels == dest_out_channels:
                                dep_type = DepType.SAME
                        elif same_dep and \
                                source_out_channels == dest_out_channels:
                            dep_type = DepType.SAME

                        # 1.2. Append the dependency
                        # (Structural Source -> Structural Destination)
                        dependencies.append(
                            (
                                source_module,
                                current_module,
                                dep_type
                            )
                        )

            # --- 2. Update Tracking Map (Only track Structural Modules
            # or pass through) ---
            # Structural Modules are producers (Conv, Linear) or
            # size-constrainers (BN)
            if is_dest_structural or is_learnable:
                node_to_module[node] = make_safelist(current_module)
            elif source_node and source_node in node_to_module:
                # Pass through: For stateless layers (ReLU, MaxPool), point
                # back to their actual source
                node_to_module[node] = make_safelist(
                    node_to_module[source_node]
                )
            else:
                # Fallback (e.g., first node)
                node_to_module[node] = make_safelist(
                    current_module
                )  # Fallback to current module if source isn't tracked

        # --- Handle General Merge Operations (Any call_function with multiple
        # module inputs) ---
        elif node.op == 'call_function' or node.op == "call_method":
            # add next steps bypass if op. change next input dimension
            # (e.g., cat)
            bypass.append(str(node.next)) if node.name == 'cat' else None

            # 1. Identify all source modules that feed into this function node
            source_modules_ = []  # Collect modules to check for single input
            source_nodes = []  # Collect nodes to check for single input
            for arg in node.args:
                if not isinstance(arg, list):
                    arg = make_safelist(arg)
                for _arg in arg:
                    if isinstance(_arg, th.fx.Node):
                        source_nodes.append(_arg)
                        source_modules = node_to_module.get(_arg)
                        if source_modules is not None:
                            for ind in range(len(source_modules)):
                                source_modules_.append(source_modules[ind])

            # Remove duplicates while preserving the order/identity
            distinct_source_modules = source_modules_

            # 2. Check for multi-branch constraint
            # (e.g., residual merge, element-wise merge)
            # If two or more *different* modules feed into the function,
            # they impose a SAME constraint.
            if len(distinct_source_modules) >= 2:
                # Apply bidirectional SAME constraint between all pairs
                # of modules that merge at this function node.
                # This covers th.add, th.mul, etc.
                for i in range(len(distinct_source_modules)):
                    for j in range(i + 1, len(distinct_source_modules)):
                        mod_a = distinct_source_modules[i]
                        mod_b = distinct_source_modules[j]
                        dependencies.append((mod_a, mod_b, DepType.REC))

            # 3. Update the module map for the function node's output
            # (i.e., intelligent pass-through)
            if len(distinct_source_modules) == 1:
                # Single-input stateless function (e.g., th.sigmoid, view):
                # pass through the source
                node_to_module[node] = distinct_source_modules
            elif len(distinct_source_modules) >= 2:
                # Multi-input merge: The function output should be tracked as
                # dependent on the first module in the merge
                node_to_module[node] = distinct_source_modules
            else:
                node_to_module[node] = None  # Placeholder or constant input

    # Final cleanup of dependencies to remove duplicates and None entries
    dependencies = list(
        set(
            [d for d in dependencies if d[0] is not None and
                d[1] is not None]
        )
    )
    return dependencies


def get_original_torch_class(
        module_instance: nn.Module,
        replacement_map: dict) -> Type[nn.Module] | None:
    """
    Maps an instance of a custom wrapper module back to its original
    th.nn Class using the module's type.
    """
    # Get the class (type object) of the provided instance
    custom_class = type(module_instance)

    # Look up the original torch class in the replacement map
    return replacement_map.get(custom_class)


def model_add_neurons(model, x=None):
    """
        Test function to iteratively update neurons for each layer,
        then test inference. Everything match ?
    """
    n_layers = len(model.layers)
    for n in range(n_layers-1, -1, -1):
        if x is not None:
            if x >= 0:
                if n != x:
                    continue
            else:
                if n != n_layers + x:  # - -x != + -x
                    continue
        print(f'Adding neuron at layer {n}', level='DEBUG')
        with model:
            model.add_neurons(n, neuron_count=1)
