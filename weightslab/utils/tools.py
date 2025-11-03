import types
import inspect
import collections
import torch as th
import torch.nn as nn

from torch.fx import GraphModule, Node
from typing import Tuple, Optional, List, Any, Type

from weightslab.utils.modules_dependencies import DepType
from weightslab.utils.logs import print


# ----------------------------------------------------------------------------
# -------------------------- Utils Functions ---------------------------------
# ----------------------------------------------------------------------------
def is_learnable_module(module: nn.Module) -> bool:
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
        module.same_flag = True
        return in_dim, out_dim, in_name, out_name

    # 4. Layers using in or out shape/size attributes
    shape_attrs = [
        i for i in list(module.__dict__.keys())
        if '_size' in i or '_shape' in i
    ]
    if len(shape_attrs) and 'flatten' in module._get_name():
        in_shape_attrs = [attr for attr in shape_attrs if 'in_' in attr]
        out_shape_attrs = [attr for attr in shape_attrs if 'in_' in attr]
        # OneWay layers, e.g., UnFlatten or BatchNorm layers
        if not len(in_shape_attrs) or not len(out_shape_attrs):
            in_dim = getattr(module, shape_attrs[0])
            in_name = shape_attrs[0]
            out_dim = getattr(module, shape_attrs[0])
            out_name = shape_attrs[0]
        elif len(in_shape_attrs) == len(out_shape_attrs):
            in_dim = getattr(module, in_shape_attrs[0])
            in_name = in_shape_attrs[0]
            out_dim = getattr(module, out_shape_attrs[0])
            out_name = out_shape_attrs[0]
        module.same_flag = True
        return in_dim, out_dim, in_name, out_name

    # 5. Catch all or return None for non-parameterized layers
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


def get_shape_attribute_from_module(
        module: nn.Module,
        _in=False,
        _out=False,
        with_name=False
):
    attrs = [
        i for i in list(module.__dict__.keys())
        if '_size' in i or '_shape' in i
    ]
    res = [None] if not with_name else (None, None)
    if not len(attrs):
        return [None] if not with_name else (None, None)
    if _in:
        _in_attrs = [i for i in attrs if '_in' in i or 'in_' in i]
        if len(_in_attrs):
            res = getattr(module, _in_attrs[0])
            res = [res] if not with_name else (res, _in_attrs[0])
    if _out:
        _out_attrs = [i for i in attrs if '_out' in i or 'out_' in i]
        if len(_out_attrs):
            res = getattr(module, _out_attrs[0])
            res = [res] if not with_name else (res, _out_attrs[0])

    if _in == _out is False:
        res = getattr(module, attrs[0])
        res = [res] if not with_name else (res, attrs[0])

    return res


def what_layer_type(module: nn.Module):
    in_attrs = [i for i in list(module.__dict__.keys()) if 'in_' in i]
    out_attrs = [i for i in list(module.__dict__.keys()) if 'out_' in i]
    shape_attrs = [
        i for i in list(module.__dict__.keys())
        if 'size_' in i or '_size' in i or
        'shape_' in i or '_shape' in i
    ]

    # Find and return layer type based on the attributes found
    if len(in_attrs) and len(out_attrs):
        return 1
    elif len(shape_attrs):
        return 2
    else:
        return 0


def make_safelist(x):
    return [x] if not isinstance(x, list) else x


def generate_mappings(src_channels: int, dst_channels: int) -> Tuple[list]:
    """
    TODO (GP): Improve this function
    Generates index mappings between a source and destination layer.

    The mapping format is a list of [from_index, [to_indices_list]].
    This structure can represent one-to-one, one-to-many, and many-to-one
    relationships.

    Args:
        src_channels (int): The number of neurons/channels in the source layer.
        dst_channels (int): The number of neurons/channels in the destination
        layer.

    Returns:
        tuple: A tuple containing (src_to_dst_mapping, dst_to_src_mapping).

    Raises:
        ValueError: If one channel count is larger than the other but not
                    perfectly divisible by the smaller one.
    """
    src_to_dst_mapping = []
    dst_to_src_mapping = []
    src_channels = list(src_channels)
    dst_channels = list(dst_channels)

    if len(src_channels) == len(dst_channels):
        # Case 1: 1-to-1 mapping
        # Each source channel maps to the corresponding dstination channel.
        src_to_dst_mapping = {i: [j] for i, j in zip(src_channels, dst_channels)}
        dst_to_src_mapping = {i: [j] for i, j in zip(dst_channels, src_channels)}

    elif len(src_channels) > len(dst_channels):
        # Case 2: Many-to-one (src > dst)
        # A "batch" of source neurons maps to a single dstination neuron.
        if len(src_channels) % len(dst_channels) != 0:
            raise ValueError(
                f"Source channels ({src_channels}) must be perfectly \
                 divisible by dstination channels ({dst_channels}) \
                 for many-to-one mapping."
            )

        # 1. Calculate the block size.
        # This determines how many linear layer neurons map to one convolution channel.
        # We use integer division to ensure a clean split.
        # Example: 8192 keys // 32 values = 256 keys per value
        group_size = len(src_channels) // len(dst_channels)

        # src_to_dst: Many-to-one
        # [src_idx, [dst_idx]]
        # e.g., src 0, 1, 2 map to dst 0 (group_size=3)
        dependency_map = dict([[src_idx, src_idx // group_size]
                              for src_idx in src_channels])
        groups = collections.defaultdict(list)
        for key, value in dependency_map.items():
            groups[value].append(key)
        src_to_dst_mapping = {key: groups[dependency_map[key]]
                              for key in dependency_map}

        # dst_to_src: One-to-many
        # [dst_idx, [src_idx_list]]
        # e.g., dst 0 maps to src 0, 1, 2
        dst_to_src_mapping_ = []
        for dst_idx in dst_channels:
            start_src_idx = dst_idx * group_size
            end_src_idx = (dst_idx + 1) * group_size
            src_indices = list(range(start_src_idx, end_src_idx))
            dst_to_src_mapping_.append([dst_idx, src_indices])
        dst_to_src_mapping = {}
        # We iterate over the key (index) and value (the range of codes)
        for index, code_range in dict(dst_to_src_mapping_).items():
            # Then, we iterate over every single code within that range
            for code in code_range:
                # We map the individual code back to the original index
                dst_to_src_mapping[code] = [index]

    else:  # src_channels < dst_channels
        # 1. Calculate the block size.
        # This determines how many linear layer neurons map to one convolution channel.
        # We use integer division to ensure a clean split.
        # Example: 8192 keys // 32 values = 256 keys per value
        block_size = len(dst_channels) // len(src_channels)

        # 2. Generate the first mapping dictionary (a)
        # The key is the linear neuron index (0 to 8191)
        # The value is the convolution channel index (0 to 31)
        neuron_to_channel_map = {
            i: i // block_size
            for i in dst_channels
        }
        # 3. Generate the second mapping dictionary (b)
        # Since you requested it to be equal, we just copy the first one.
        # Using .copy() creates a new object in memory, which is usually safer
        # than just assigning a reference (map_conv_to_linear_copy = map_conv_to_linear),
        # unless you explicitly need them to share the *same* memory ID,
        # which is rare for simple immutable mappings.
        channel_to_neuron_map = collections.defaultdict(list)

        for neuron_id, channel_id in neuron_to_channel_map.items():
            channel_to_neuron_map[channel_id].append(neuron_id)

        dst_to_src_mapping = dict(channel_to_neuron_map)
        src_to_dst_mapping = {i: [i] for i in range(len(dst_to_src_mapping))}
        dst_to_src_mapping = {k: u if isinstance(u, list) else [u]
                              for k, u in dst_to_src_mapping.items()}

    return src_to_dst_mapping, dst_to_src_mapping


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
        bypassed = False
        current_module = None
        if node.op == 'call_module':
            # Get current module from node
            current_module = get_module_by_name(model, node.target)
            current_layer_type = current_module.layer_type if hasattr(current_module, 'layer_type') else -1

            # If the current module is a multi-input layer, flag as bypass
            if node.name in bypass:
                # bypass strategy for recursive update dependencies,
                # like bypass = true for __add__ but false for cat;
                # and cnt for neurons mapping src / dst
                current_module.bypass = 0

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
            is_dst_structural = is_feature_producer(current_module)
            is_learnable = is_module_learnable(current_module)
            has_layer_type = hasattr(current_module, 'layer_type')
            if source_modules:
                for source_module in source_modules:
                    if source_module is not None and \
                        (has_layer_type or is_dst_structural or is_learnable):
                        # 1.1. Determine Dependency Type based on Shape
                        # (for Pruning)
                        dep_type = DepType.INCOMING
                        source_out_channels = get_feature_channel_size(
                            source_node
                        )
                        dst_out_channels = get_feature_channel_size(node)

                        # 1.2. Check if current module should be target SAME
                        # path. It's a specific case where current module has
                        # in==out shapes
                        # Check for SAME constraint (requires source to be a
                        # producer)
                        if current_layer_type == 1 and \
                            source_out_channels is not None and \
                                dst_out_channels is not None:
                            if hasattr(current_module, 'same_flag'):
                                dep_type = DepType.SAME
                        else:
                            dep_type = DepType.SAME
                            # current_module.bypass = 1

                        # 1.3. Append the dependency
                        # (Structural Source -> Structural dstination)
                        dependencies.append(
                            (
                                source_module,
                                current_module,
                                dep_type
                            )
                        )
                        if hasattr(current_module, 'bypass'):
                            source_module.src_bypass = 1

            # --- 2. Update Tracking Map (Only track Structural Modules
            # or pass through) ---
            # Structural Modules are producers (Conv, Linear) or
            # size-constrainers (BN)
            if current_layer_type >= 1 or is_learnable:
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
            if node.name == 'cat':
                bypass.append(str(node.next))
                bypassed = True

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
                        if not bypassed:
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

    # Generate mapping tensor btw deps
    for dep in dependencies:
        # Get src and dst modules and type
        src_mod, dst_mod, dep_type = dep[0], dep[1], dep[2]

        # Is it a recursive dependency ?
        recursive_dep = dep_type == DepType.REC

        source_out_channels = range(src_mod.get_neurons(attr_name='out_neurons'))
        dst_in_channels = range(dst_mod.get_neurons(attr_name='in_neurons') if not recursive_dep else
                                dst_mod.get_neurons(attr_name='out_neurons'))
        if hasattr(dst_mod, 'bypass') or hasattr(dst_mod, 'bypassed'):
            dst_in_channels = range(
                dst_mod.bypass,
                len(source_out_channels) + dst_mod.bypass
            )
            dst_mod.bypass = len(source_out_channels)

        # 1.4. Generate mappings tnsr for src and dst layers
        src_to_dst_mapping_tnsr, dst_to_src_mapping_tnsr = \
            generate_mappings(
                source_out_channels,
                dst_in_channels
            )
        if not recursive_dep:
            # should be reverse mapping
            dst_mod.dst_to_src_mapping_tnsrs.update(
                {
                    src_mod.get_name_wi_id():
                        dst_to_src_mapping_tnsr
                }
            )
        else:
            dst_mod.src_to_dst_mapping_tnsrs.update(
                {
                    src_mod.get_name_wi_id():
                        dst_to_src_mapping_tnsr
                }

            )
        src_mod.src_to_dst_mapping_tnsrs.update(
            {
                dst_mod.get_name_wi_id():
                    src_to_dst_mapping_tnsr
            }
        )
        # Add to dst node, the parent indexs maps also
        dst_mod.parents_src_to_dst_mapping_tnsrs.update(
            {
                dst_mod.get_name_wi_id():
                    src_to_dst_mapping_tnsr
            }
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


def model_op_neurons(model, layer_id=None, dummy_input=None, op=None):
    """
        Test function to iteratively update neurons for each layer,
        then test inference. Everything match ?
    """
    n_layers = len(model.layers)
    for n in range(n_layers-1, -1, -1):
        if layer_id is not None:
            if layer_id >= 0:
                if n != layer_id:
                    continue
            else:
                if n != n_layers + layer_id:  # - -layer_id != + -layer_id
                    continue
        print(f'Operate on neurons at layer {n}', level='DEBUG')
        with model as m:
            if op is None:
                print('Adding operation - 2 neurons added.',
                      level='DEBUG')
                m.operate(n, {0, 0, 0, 0, 0}, neuron_operation=1)
                m(dummy_input) if dummy_input is not None else None
                print('Reseting operation - every neurons reset.',
                      level='DEBUG')
                m.operate(n, {}, neuron_operation=4)
                m(dummy_input) if dummy_input is not None else None
                print('Freezing operation - last neuron froze.',
                      level='DEBUG')
                m.operate(n, {-3}, neuron_operation=3)
                m(dummy_input) if dummy_input is not None else None
                print('Pruning operation - first neuron removed.',
                      level='DEBUG')
                m.operate(n, {0, 1}, neuron_operation=2)
                m(dummy_input) if dummy_input is not None else NotImplemented
            else:
                m.operate(
                    n,
                    {-1},
                    neuron_operation=op
                )
                m(dummy_input) if dummy_input is not None else None


def reindex_and_compress_blocks(data_dict, block_size, offset_index=0):
    """
    Re-indexes the dictionary keys and shifts the neuron value ranges to ensure
    they remain contiguous starting from 0, after removing an intermediate
    block.

    Args:
        data_dict (dict): The dictionary with non-contiguous keys and value
        ranges.
        block_size (int): The fixed size of each neuron block (e.g., 256).

    Returns:
        dict: The re-indexed dictionary with contiguous keys and compressed
        values.
    """
    # 1. Sort the remaining blocks by their original keys to maintain order
    # The dictionary keys must be sorted to ensure the blocks are processed
    # sequentially.
    sorted_blocks = collections.OrderedDict(sorted(data_dict.items()))

    reindexed_dict = {}

    # 2. Iterate through the remaining blocks, assigning a new contiguous index
    for new_index in range(len(list(sorted_blocks.items()))):
        index_batch = new_index // block_size
        # Calculate the new starting point for the range.
        # This point ensures the range is contiguous (0 * size, n * size, ...)
        new_start = block_size*index_batch
        new_end = new_start + block_size

        # Create the new contiguous range
        new_range = list(range(new_start, new_end))

        # Assign the new key and the compressed value range
        reindexed_dict[new_index+offset_index] = new_range

    return reindexed_dict


def get_layer_trainable_parameters_neuronwise(layer: th.nn.Module):
    """
        Count the number of neurons with associated lr != 0.
    """
    # TODO (GP) Review function; seems like not working as expected with conv.
    # TODO (GP) when having kernel size (counts now only in out params. wo.
    # TODO (GP) corr. to kernel weights).
    trainable_params = 0
    for learnable_tensor_name in layer.learnable_tensors_name:
        trainable_params += getattr(layer, learnable_tensor_name).numel()
        trainable_params -= len(
            layer.neuron_2_lr[
                learnable_tensor_name
            ]
        )
        if learnable_tensor_name in layer.incoming_neuron_2_lr:
            trainable_params -= len(
                layer.incoming_neuron_2_lr[
                    learnable_tensor_name
                ]
            )
    return trainable_params


def get_model_parameters_neuronwise(model: th.nn.Module, trainable_only=True):
    """
        Get the number of neurons with associated lr!= 0 in the model.
    """
    # Count only neurons with associated lr != 0
    # Basically parameters not masked
    params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = 0
    for layer in model.layers:
        trainable_params += get_layer_trainable_parameters_neuronwise(layer)

    # Since all parameters in your model currently have requires_grad=True:
    # trainable_params will also equal 8,367,235
    print(
        f"{params} paraeters with {trainable_params} trainable parameters.",
        level='DEBUG'
    )

    return (params, trainable_params) if not trainable_only else \
        trainable_params