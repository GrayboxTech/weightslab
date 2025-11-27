import collections
import torch as th
import torch.nn as nn

from copy import deepcopy
from typing import Tuple, List

from torch.fx import GraphModule

from weightslab.utils.modules_dependencies import DepType
from weightslab.utils.tools import *


def generate_mappings(
    src_channels: int,
    dst_channels: int,
    dst_groups: int = 1,
    src_groups: int = 1
) -> Tuple[list]:
    """
    Generates index mappings between a source and destination layer.

    The mapping format is a list of [from_index, [to_indices_list]].
    This structure can represent one-to-one, one-to-many, and many-to-one
    relationships.

    Args:
        src_channels (int): The number of neurons/channels in the source layer.
        dst_channels (int): The number of neurons/channels in the destination
        layer.
        groups (int): The number of groups to divide the channels into.

    Returns:
        tuple: A tuple containing (src_to_dst_mapping, dst_to_src_mapping).

    Raises:
        ValueError: If one channel count is larger than the other but not
                    perfectly divisible by the smaller one.
    """
    src_group_size = 1
    dst_group_size = 1
    src_to_dst_mapping = []
    dst_to_src_mapping = []
    src_channels = list(src_channels)
    dst_channels = list(dst_channels)

    # 1. Calculate the size of the block (group) for both input and output
    if src_groups is not None:
        src_group_size = len(src_channels) // max(src_groups, 1)
    if dst_groups is not None:
        dst_group_size = len(dst_channels) // max(1, dst_groups)

    if len(src_channels) == len(dst_channels):
        # Case 1: 1-to-1 mapping
        # Each source channel maps to the corresponding dstination channel.
        # src_to_dst_mapping = {i: [j] for i, j in zip(src_channels, dst_channels)}
        src_to_dst_mapping = {}
        # 2. Iterate through every source neuron
        for src_idx in src_channels:

            # Determine which group the current source neuron belongs to
            group_idx = src_idx // src_group_size

            # Calculate the starting index for the connected destination neurons
            dst_start_idx = group_idx * dst_group_size

            # Calculate the ending index (exclusive) for the connected destination neurons
            dst_end_idx = dst_start_idx + dst_group_size

            # Create the list of connected destination neuron indices for this group
            connected_dst_neurons = list(range(dst_start_idx, dst_end_idx))

            # Add the mapping to the dictionary
            src_to_dst_mapping[src_idx] = connected_dst_neurons
        # dst_to_src_mapping = {i: [j] for i, j in zip(dst_channels, src_channels)}
        dst_to_src_mapping = {}
        # 2. Iterate through every source neuron
        for dst_idx in dst_channels:
            # Determine which group the current source neuron belongs to
            group_idx = dst_idx // dst_group_size

            # Calculate the starting index for the connected destination neurons
            src_start_idx = group_idx * src_group_size

            # Calculate the ending index (exclusive) for the connected destination neurons
            src_end_idx = src_start_idx + src_group_size

            # Create the list of connected destination neuron indices for this group
            connected_src_neurons = list(range(src_start_idx, src_end_idx))

            # Add the mapping to the dictionary
            dst_to_src_mapping[dst_idx] = connected_src_neurons

    elif len(src_channels) > len(dst_channels):
        # Case 2: Many-to-one (src > dst)
        # A "batch" of source neurons maps to a single dstination neuron.
        # if len(src_channels) % len(dst_channels) != 0:
        #     raise ValueError(
        #         f"Source channels ({src_channels}) must be perfectly \
        #          divisible by dstination channels ({dst_channels}) \
        #          for many-to-one mapping."
        #     )

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
        group_size = len(dst_channels) // len(src_channels) * src_group_size

        # 2. Generate the first mapping dictionary (a)
        # The key is the linear neuron index (0 to 8191)
        # The value is the convolution channel index (0 to 31)
        neuron_to_channel_map = {
            i: i // group_size
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

        dst_to_src_mapping_ = dict(channel_to_neuron_map)
        # src_to_dst_mapping = {i: [i] for i in range(len(dst_to_src_mapping_))}
        dst_to_src_mapping_ = {k: u if isinstance(u, list) else [u]
                               for k, u in dst_to_src_mapping_.items()}
        dst_to_src_mapping = {
            input_neuron_index: input_range
            for input_range in dst_to_src_mapping_.values()
            for input_neuron_index in input_range
        }
        src_to_dst_mapping = {}
        # 2. Iterate through every source neuron
        for src_idx in src_channels:

            # Determine which group the current source neuron belongs to
            group_idx = src_idx // src_group_size

            # Calculate the starting index for the connected destination neurons
            dst_start_idx = group_idx * dst_group_size

            # Calculate the ending index (exclusive) for the connected destination neurons
            dst_end_idx = dst_start_idx + dst_group_size

            # Create the list of connected destination neuron indices for this group
            connected_dst_neurons = list(range(dst_start_idx, dst_end_idx))

            # Add the mapping to the dictionary
            src_to_dst_mapping[src_idx] = connected_dst_neurons
    return src_to_dst_mapping, dst_to_src_mapping


def generate_graph_dependencies(
        model: nn.Module,
        traced_graph: GraphModule,
        indexing_neurons: bool = True
) -> \
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
                            if hasattr(current_module, 'wl_same_flag'):
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
            if 'cat' in node.name or 'cat_' in node.name:
                bypass.append(str(node.next))
                # bypassed = True

            # 1. Identify all source modules that feed into this function node
            # TODO (GP): Find recursive approach to do that, if there are
            # TODO (GP): cat of cat of cat, should be nested list also ?
            # TODO (GP): e.g., cat([conv1, conv2, cat([conv3, cat([conv4,
            # TODO (GP): conv5])])])])
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
                    elif isinstance(_arg, (tuple, set, list)):
                        for __arg in _arg:
                            if isinstance(__arg, th.fx.Node):
                                source_nodes.append(__arg)
                                source_modules = node_to_module.get(__arg)
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
    if not indexing_neurons:
        return dependencies 
    for edge in dependencies:
        # Get src and dst modules and type
        src_mod, dst_mod, edge_label = edge[0], edge[1], edge[2]
        recursive_dep = edge_label == DepType.REC  # A recursive dependency ?

        # 1.1. Determine the number of neurons in each direction
        source_out_channels = range(
            src_mod.get_neurons(attr_name='out_neurons') if
            not hasattr(src_mod, 'wl_transposed')
            else src_mod.get_neurons(attr_name='in_neurons')
        )
        dst_in_channels = range(
            dst_mod.get_neurons(attr_name='in_neurons') if not recursive_dep
            and not hasattr(dst_mod, 'wl_transposed')
            else dst_mod.get_neurons(attr_name='out_neurons')
        )
        # # For multi-input / one output layers (e.g., Cat)
        if hasattr(dst_mod, 'bypass'):
            dst_in_channels = range(
                dst_mod.bypass,
                len(source_out_channels) + dst_mod.bypass
            )
            dst_mod.bypass += len(source_out_channels)

        # 1.2. Generate mappings tnsr for src and dst layers
        groups = dst_mod.groups if hasattr(dst_mod, 'groups') else (
            src_mod.groups
        ) if hasattr(src_mod, 'groups') else None
        src_to_dst_mapping_tnsr, dst_to_src_mapping_tnsr = \
            generate_mappings(
                source_out_channels,
                dst_in_channels,
                dst_groups=(len(dst_in_channels) // groups) if groups is
                not None else None,
                src_groups=len(source_out_channels) // groups if groups is
                not None else None
            )

        # 1.3 Update neurons mapping tensors
        # # Update edge dst node with neurons mapping tensor
        if not recursive_dep:
            # should be_ reverse mapping
            dst_mod.dst_to_src_mapping_tnsrs.update(
                {
                    src_mod.get_name_wi_id():
                        dst_to_src_mapping_tnsr
                }
            )
            # # Update edge child and parent node with neurons mapping tensor
            dst_mod.related_src_to_dst_mapping_tnsrs.update(
                {
                    dst_mod.get_name_wi_id():
                        deepcopy(src_to_dst_mapping_tnsr)
                } if not hasattr(dst_mod, 'bypass') else {}
            )
            dst_mod.dst_to_src_mapping_tnsrs = normalize_dicts(dst_mod.dst_to_src_mapping_tnsrs)
            dst_mod.related_src_to_dst_mapping_tnsrs = normalize_dicts(dst_mod.related_src_to_dst_mapping_tnsrs)

        else:
            # Recursive dependency: src & dst are reversed
            # here for mapping logic
            dst_mod.src_to_dst_mapping_tnsrs.update(
                {
                    src_mod.get_name_wi_id():
                        dst_to_src_mapping_tnsr
                }

            )
            # # Update edge child and parent node with neurons mapping tensor
            dst_mod.related_dst_to_src_mapping_tnsrs.update(
                {
                    dst_mod.get_name_wi_id():
                        deepcopy(src_to_dst_mapping_tnsr)
                } if not hasattr(dst_mod, 'bypass') else {}
            )  # Child equivalent here
            dst_mod.src_to_dst_mapping_tnsrs = normalize_dicts(dst_mod.src_to_dst_mapping_tnsrs)
            dst_mod.related_dst_to_src_mapping_tnsrs = normalize_dicts(dst_mod.related_dst_to_src_mapping_tnsrs)

        # # Update edge src node with neurons mapping tensor
        src_mod.src_to_dst_mapping_tnsrs.update(
            {
                dst_mod.get_name_wi_id():
                    src_to_dst_mapping_tnsr
            }
        )
        src_mod.related_dst_to_src_mapping_tnsrs.update(
            {
                src_mod.get_name_wi_id():
                    deepcopy(dst_to_src_mapping_tnsr)
            } if not hasattr(src_mod, 'bypass') else {}
        )
        src_mod.src_to_dst_mapping_tnsrs = normalize_dicts(src_mod.src_to_dst_mapping_tnsrs)
        src_mod.related_dst_to_src_mapping_tnsrs = normalize_dicts(src_mod.related_dst_to_src_mapping_tnsrs)

    return dependencies