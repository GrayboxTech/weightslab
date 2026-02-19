import tempfile
import torch.nn as nn
import torch as th
import onnx
import onnx.shape_inference
import collections
import logging


from torch.fx import GraphModule
from torch._C import _onnx as _C_onnx
from typing import Iterable, List, Tuple, Dict, Optional
from copy import deepcopy
from typing import Any

from weightslab.utils.tools import *
from weightslab.utils.modules_dependencies import DepType
from weightslab.utils.tools import (
    normalize_dicts
)


# Get Global Logger
logger = logging.getLogger(__name__)


def _export_model_to_onnx_temp(
    model: nn.Module,
    dummy_input: th.Tensor,
    opset_version: int = 17,
) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export
        dummy_input: A sample input tensor for the model
        opset_version: ONNX opset version to use (default: 16 for better compatibility)
                      Use 16+ for models with AdaptiveAvgPool (VGG, ResNet, etc.)
    """

    onnx_file_path = tempfile.mkstemp(suffix='.onnx')[1]
    try:
        th.onnx.export(
            model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=['input'],
            training=_C_onnx.TrainingMode.PRESERVE,
            output_names=['output'],
            verbose=False
        )
        logger.info(f"Model exported to {onnx_file_path} (opset {opset_version})")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}. Please try with higher version of opset_version or without ONNX export.")
        raise e

    return onnx_file_path

def _generate_mappings(
    src_channels: Iterable[int],
    dst_channels: Iterable[int],
    dst_groups: Optional[int] = None,
    src_groups: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
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

def _detect_layer_constraints(module: nn.Module) -> Dict[str, Any]:
    """
    Detect structural constraints on a layer by analyzing its properties.
    Uses introspection to detect constraints without hardcoding.

    Constraints detected:
    - 'grouped': Dict with 'groups' count and 'is_depthwise' flag

    Args:
        module: The module to analyze

    Returns:
        Dict mapping constraint_name -> constraint_value
    """
    constraints = {}

    # Encode groups constraints
    try:
        # Check for grouped convolution (Conv1d, Conv2d, Conv3d)
        if hasattr(module, 'groups') and isinstance(module.groups, int):
            if module.groups > 1:
                # Get in/out channels to determine if depthwise
                in_channels = None
                out_channels = None

                if hasattr(module, 'in_channels'):
                    in_channels = module.in_channels
                if hasattr(module, 'out_channels'):
                    out_channels = module.out_channels

                is_depthwise = False
                if in_channels is not None and in_channels == module.groups:
                    is_depthwise = True

                constraints['grouped'] = {
                    'cons_group_size': module.groups,
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'is_depthwise': is_depthwise,
                    'is_incoming': False
                }
    except Exception as e:
        logger.debug(f"Error detecting constraints for {type(module).__name__}: {e}")

    return constraints


def _propagate_constraints_through_dependencies(
    dependencies: List[Tuple[nn.Module, nn.Module, DepType]],
    constraint_check_fn = None
) -> Dict[int, Dict[str, Any]]:
    """
    Propagate constraints through the dependency graph.

    Constraints originating from a layer are propagated downstream through
    SAME dependencies. Propagation stops at INCOMING dependencies, which
    represent transformation boundaries (e.g., Conv -> Linear).

    Args:
        dependencies: List of (src_module, dst_module, dep_type) tuples
        constraint_check_fn: Optional custom function to detect constraints.
                           If None, uses _detect_layer_constraints.

    Returns:
        Dict mapping module_id -> {
            'incoming': {constraint_name -> constraint_value},
            'outgoing': {constraint_name -> constraint_value}
        }
    """
    if constraint_check_fn is None:
        constraint_check_fn = _detect_layer_constraints

    # First pass: detect native constraints on all modules
    module_constraints = {}
    unique_modules = {}

    for src, dst, _ in dependencies:
        src_id = id(src)
        dst_id = id(dst)

        if src_id not in unique_modules:
            unique_modules[src_id] = src
            module_constraints[src_id] = constraint_check_fn(src)

        if dst_id not in unique_modules:
            unique_modules[dst_id] = dst
            module_constraints[dst_id] = constraint_check_fn(dst)

    # Second pass: build adjacency list for forward propagation
    forward_edges = {}
    for src, dst, dep_type in dependencies:
        src_id = id(src)
        dst_id = id(dst)

        if src_id not in forward_edges:
            forward_edges[src_id] = []
        forward_edges[src_id].append((dst_id, dep_type))

    # Third pass: propagate constraints downstream
    # We track constraints separately for incoming (affecting dst inputs)
    # and outgoing (native to module or pass-through outputs).
    propagated_constraints: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for module_id, native_constraints in module_constraints.items():
        if not native_constraints:
            # No constraints on this module, skip propagation
            if module_id not in propagated_constraints:
                propagated_constraints[module_id] = {}
            continue

        # Seed native constraints as OUTGOING for the module itself
        if module_id not in propagated_constraints:
            propagated_constraints[module_id] = {
                'incoming': {},
                'outgoing': {}
            }
        for cname, cval in native_constraints.items():
            if cname not in propagated_constraints[module_id].get('outgoing', {}):
                propagated_constraints[module_id].get('outgoing', {})[cname] = cval

        # BFS to propagate OUTGOING constraints downstream
        queue = [(module_id, native_constraints.copy(), {module_id})]  # (current_id, outgoing_constraints, visited)

        while queue:
            current_id, current_constraints, visited_set = queue.pop(0)

            # Ensure dict structure exists for current module
            if current_id not in propagated_constraints:
                propagated_constraints[current_id] = {
                    'incoming': {},
                    'outgoing': {}
                }
            # Current node retains its OUTGOING constraints (merge once)
            for constraint_name, constraint_value in current_constraints.items():
                if constraint_name not in propagated_constraints[current_id].get('outgoing', {}):
                    propagated_constraints[current_id].get('outgoing', {})[constraint_name] = constraint_value

            # Propagate downstream with different rules for each dependency type
            if current_id in forward_edges:
                for next_id, dep_type in forward_edges[current_id]:
                    if next_id not in visited_set:
                        if next_id not in propagated_constraints:
                            propagated_constraints[next_id] = {
                                'incoming': {},
                                'outgoing': {}
                            }
                        if dep_type == DepType.INCOMING:
                            # INCOMING: dst receives current OUTGOING as INCOMING; do not continue further
                            for constraint_name, constraint_value in current_constraints.items():
                                if constraint_name not in propagated_constraints[next_id].get('incoming', {}):
                                    propagated_constraints[next_id].get('incoming', {})[constraint_name] = constraint_value
                            # stop propagation here (no enqueue)
                        else:
                            # SAME and REC: pass-through → dst gets only OUTGOING constraint (not incoming), and continue
                            for constraint_name, constraint_value in current_constraints.items():
                                if constraint_name not in propagated_constraints[next_id].get('outgoing', {}):
                                    propagated_constraints[next_id].get('outgoing', {})[constraint_name] = constraint_value
                                    propagated_constraints[next_id].get('incoming', {})[constraint_name] = constraint_value
                            visited_set_copy = visited_set.copy()
                            visited_set_copy.add(next_id)
                            queue.append((next_id, current_constraints.copy(), visited_set_copy))

    # Ensure all modules have an entry (even if empty)
    for module_id in module_constraints:
        if module_id not in propagated_constraints:
            propagated_constraints[module_id] = {
                'incoming': {},
                'outgoing': {}
            }

    return propagated_constraints


def _flag_layers_with_constraints(
    model: nn.Module,
    dependencies: List[Tuple[nn.Module, nn.Module, DepType]],
    constraint_check_fn = None
) -> Dict[str, Dict]:
    """
    Flag layers with constraints and propagate them through the graph.

    Attaches constraint information to each module as attributes:
    - module.wl_constraints: Dict of constraint_name -> value
    - module.wl_constraint_source_id: ID of the module where constraint originated

    Args:
        model: The PyTorch model
        dependencies: List of layer dependencies
        constraint_check_fn: Optional custom constraint detection function

    Returns:
        Dict mapping module_name -> constraint_info for reporting
    """
    # Get propagated constraints
    propagated = _propagate_constraints_through_dependencies(dependencies, constraint_check_fn)

    # Map module IDs to module names for reporting
    id_to_name = {}
    for name, module in model.named_modules():
        id_to_name[id(module)] = name

    # Apply constraints as attributes and collect reporting info
    constraint_report = {}

    for module_id, constraints in propagated.items():
        if not constraints:
            continue

        module_name = id_to_name.get(module_id, f"unknown_{module_id}")

        # Find the actual module object
        module = None
        for m in model.modules():
            if id(m) == module_id:
                module = m
                break

        if module is not None:
            # Attach unified constraints list with incoming/outcoming flags
            incoming_dict = constraints.get('incoming', {})
            outgoing_dict = constraints.get('outgoing', {})
            module.wl_constraint_source_id = module_id

            unified: List[Dict[str, Any]] = []

            # Helper to append entries (single boolean flag 'incoming')
            def append_entries(src: Dict[str, Any], is_incoming: bool):
                for cname, cval in src.items():
                    entry = {'name': cname, **cval, 'incoming': is_incoming}
                    unified.append(entry)

            append_entries(incoming_dict, True)
            append_entries(outgoing_dict, False)

            module.wl_constraints = unified

            # No cons_group_size_* attributes stored; consumer derives from wl_constraints

            # Build reporting info
            constraint_report[module_name] = {
                'wl_constraints': module.wl_constraints,
                'module_type': type(module).__name__,
            }

            logger.debug(f"Flagged layer '{module_name}' ({type(module).__name__}) with constraints: {constraints}")

    return constraint_report


def _alias_from_tensor_name(tensor_name: str) -> Optional[str]:
    """
    Extract the Pyth module name from an ONNX tensor name.

    ONNX tensor names follow the pattern:
    - Simple: '/module_name/Operation_output_0' -> 'module_name'
    - Nested: '/parent/child/Operation_output_0' -> 'parent.child'
    - Redundant: '/model/layer1/layer1.0/conv1/...' -> 'model.layer1.0.conv1'
      (ONNX adds redundant parent names in the path)

    Examples:
        '/conv1/Conv_output_0' -> 'conv1'
        '/block1/conv1/Conv_output_0' -> 'block1.conv1'
        '/model/conv1/Conv_output_0' -> 'model.conv1'
        '/model/layer1/layer1.0/conv1/Conv_output_0' -> 'model.layer1.0.conv1'
          (not 'model.layer1.layer1.0.conv1' - removes redundant 'layer1')

    The ONNX path structure is: /module1/module2/.../Operation_output_N
    We need to reconstruct: module1.module2... while removing redundancies
    """
    if not tensor_name.startswith("/"):
        # fallback: treat the whole name as alias
        return tensor_name

    # Remove leading '/' and split by '/'
    parts = tensor_name[1:].split("/")

    if len(parts) < 2:
        return None

    # The last part is always the operation (e.g., 'Conv_output_0', 'BatchNormalization_output_0')
    # Everything before that is the module path
    module_parts = parts[:-1]

    # Handle ONNX redundancy: /model/layer1/layer1.0/conv1/...
    # Here 'layer1' is redundant because 'layer1.0' already contains it
    # We need to skip parts that are prefixes of the next part
    deduplicated = []
    for i, part in enumerate(module_parts):
        # Check if this part is a redundant prefix of the next part
        if i + 1 < len(module_parts):
            next_part = module_parts[i + 1]
            # If next part starts with current part + '.', it's redundant
            if next_part.startswith(part + '.'):
                continue  # Skip this redundant part
        deduplicated.append(part)

    return '.'.join(deduplicated)

def _get_onnx_shapes_map(onnx_file_path: str) -> Dict[str, Optional[Tuple[int, ...]]]:
    """
    ONNX shape inference → tensor_name -> shape.
    """
    try:
        model = onnx.load(onnx_file_path)
        inferred_model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Error during ONNX shape inference. Proceeding with limited shape info: {e}")
        return {}

    shapes_map: Dict[str, Optional[Tuple[int, ...]]] = {}
    graph = inferred_model.graph
    all_value_infos = list(graph.input) + list(graph.output) + list(graph.value_info)

    for tensor_info in all_value_infos:
        type_info = tensor_info.type.tensor_type
        if not type_info.HasField("shape"):
            continue

        dims = []
        for d in type_info.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(-1)
            else:
                dims.append(-1)

        shape = tuple(dims)
        shapes_map[tensor_info.name] = shape if any(x > 0 for x in shape) else None

    return shapes_map

def _clean_dependencies(
    dependencies: List[Tuple[nn.Module, nn.Module, DepType]]
) -> List[Tuple[nn.Module, nn.Module, DepType]]:
    """Remove self-loops and duplicate dependency edges.

    - Self-loops (where src is dst) are removed.
    - Duplicate edges (same src object, same dst object, same DepType)
    are removed, preserving the first occurrence order.

    Args:
        dependencies: List of tuples (src_module, dst_module, DepType).

    Returns:
        Cleaned list of dependencies.
    """
    seen = set()
    cleaned = []
    for src, dst, dep in dependencies:
        # Remove self-loops
        if src is dst:
            continue
        key = (id(src), id(dst), dep)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((src, dst, dep))
    return cleaned

def _infer_dependency_type(
    dst_mod: nn.Module,
) -> DepType:
    """
    Infer dependency type by analyzing module structure only - NO HARDCODING.

    Pure introspection: Check if module has learnable weight parameters.
    If it has 2D+ weight matrix, it CAN transform dimensions independently -> INCOMING
    If it has only 1D parameters or no weights, it CANNOT -> SAME

    Args:
        dst_mod: Module to analyze

    Returns:
        DepType.INCOMING or DepType.SAME
    """
    try:
        # Check ALL parameters of the module
        has_2dp_weight = False

        for _, param in dst_mod.named_parameters(recurse=False):
            if param is None:
                continue

            param_dim = param.dim()

            # Found a 2D+ parameter (weight matrix) -> can transform
            if param_dim >= 2:
                has_2dp_weight = True
                break


        # Decision (no hardcoding):
        # 2D+ weight = transformation capability = INCOMING
        # Only 1D params or no params = no transformation = SAME

        if has_2dp_weight:
            return DepType.INCOMING
        else:
            return DepType.SAME

    except Exception:
        # Safe fallback
        return DepType.SAME

def generate_graph_dependencies_from_torchfx(
        model: nn.Module,
        graph: GraphModule,
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
    for node in graph.nodes:
        bypassed = False
        current_module = None
        if node.op == 'call_module':
            # Get current module from node
            current_module = get_module_by_name(model, node.target)

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

            # --- 1. Dependency Creation (from Source Module to Current Module) ---
            if source_modules:
                for source_module in source_modules:
                    if source_module is not None:
                        # 1.1. Determine Dependency Type using the same logic as ONNX
                        # This checks if the module has 2D+ weight parameters
                        # (INCOMING) or only 1D params/no params (SAME)
                        dep_type = _infer_dependency_type(current_module)

                        # 1.2. Append the dependency
                        # Include all modules in the graph (learnable, structural, and non-learnable)
                        dependencies.append(
                            (
                                source_module,
                                current_module,
                                dep_type
                            )
                        )

            # Track all modules (learnable, structural, and non-learnable) so they can be
            # referenced as sources by subsequent nodes.
            # For non-learnable pass-through layers (ReLU, MaxPool, etc.), we track them
            # directly instead of passing through to an earlier source.
            if current_module is not None:
                node_to_module[node] = make_safelist(current_module)

                # # SEED NEURONS: Use FX metadata to seed neurons if possible
                # for mod in make_safelist(current_module):
                #     if 'tensor_meta' in node.meta:
                #         meta = node.meta['tensor_meta']
                #         if hasattr(meta, 'shape') and len(meta.shape) >= 2:
                #             out_ch = meta.shape[1]
                #             if out_ch is not None and out_ch > 0:
                #                 mod.set_neurons('out_neurons', out_ch)
                #                 if getattr(mod, 'wl_same_flag', False):
                #                     mod.set_neurons('in_neurons', out_ch)

                #     # Also check inputs to seed in_neurons
                #     for arg in node.args:
                #         if isinstance(arg, th.fx.Node) and 'tensor_meta' in arg.meta:
                #             meta_in = arg.meta['tensor_meta']
                #             if hasattr(meta_in, 'shape') and len(meta_in.shape) >= 2:
                #                 in_ch = meta_in.shape[1]
                #                 if in_ch is not None and in_ch > 0:
                #                     mod.set_neurons('in_neurons', in_ch)

        # --- Handle General Merge Operations (Any call_function with multiple
        # module inputs) ---
        elif node.op == 'call_function' or node.op == "call_method":
            # add next steps bypass if op. change next input dimension
            # (e.g., cat)
            if 'cat' in node.name or 'cat_' in node.name:
                bypass.append(str(node.next))
                bypassed = True

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
                            dependencies.append((mod_b, mod_a, DepType.REC))

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

    # Clean dependencies (remove duplicates and self-loops)
    dependencies = _clean_dependencies(dependencies)

    # Flag layers with constraints and propagate them
    _flag_layers_with_constraints(model, dependencies)

    return dependencies

def generate_layer_dependencies_from_onnx(
    model: nn.Module,
    dummy_input: Optional[th.Tensor] = None,
) -> List[Tuple[nn.Module, nn.Module, DepType]]:
    """
    Generate a simplified list of layer dependencies from an ONNX graph.

    This function analyzes the ONNX computational graph to identify how layers
    are connected and how neuron count changes propagate through the network.

    Args:
        onnx_file_path: Path to the ONNX model file
        model: The Pyth model corresponding to the ONNX file

    Returns:
        List of tuples (layer1_module, layer2_module, dependency_type) where:
        - layer1_name: Name of the source layer
        - layer2_name: Name of the destination layer
        - dependency_type: DepType.SAME or DepType.INCOMING
            - SAME: Output neuron changes in layer1 propagate to both input AND output of layer2
                   (e.g., Conv2d -> BatchNorm2d, BatchNorm2d -> ReLU)
            - INCOMING: Output neuron changes in layer1 only affect input of layer2
                       (e.g., ReLU -> Conv2d, Conv2d -> Conv2d)

    Example:
        For a CNN: input > Conv2d > BatchNorm2d > ReLU > Conv2d > output

        Returns:
        [
            ('conv1', 'bn1', DepType.SAME),       # Conv output = BN input/output
            ('bn1', 'relu', DepType.SAME),         # BN output = ReLU input/output
            ('relu', 'conv2', DepType.INCOMING),   # ReLU output = Conv2 input only
        ]

    Note:
        Enable DEBUG logging to see detailed processing information:
        >>> import logging
        >>> logging.basicConfig(level=logging.DEBUG)
    """

    # First generate the onnx file from the model temporarily
    onnx_file_path = _export_model_to_onnx_temp(model, dummy_input=th.randn(model.input_shape) if dummy_input is None else dummy_input)

    # Load ONNX model and get shape information
    try:
        onnx_model = onnx.load(onnx_file_path)
        onnx_shapes_map = _get_onnx_shapes_map(onnx_file_path)
        graph = onnx_model.graph
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model from {onnx_file_path}: {e}")

    all_modules = dict(model.named_modules())
    # Filter to only leaf modules (actual operation layers, not containers)
    name_to_module: Dict[str, nn.Module] = {
        name: mod for name, mod in all_modules.items()
        if hasattr(mod, 'is_leaf') and mod.is_leaf
    }
    module_to_name: Dict[nn.Module, str] = {m: n for n, m in name_to_module.items()}

    logger.debug(f"Available leaf modules in model: {sorted([k for k in name_to_module.keys() if k])}")

    # Map tensor outputs to their producing nodes
    producer_for_tensor: Dict[str, onnx.NodeProto] = {}
    tensor_to_mod: Dict[str, nn.Module] = {}

    for node in graph.node:
        for out_name in node.output:
            producer_for_tensor[out_name] = node

            # Try to associate tensor with a module
            alias = _alias_from_tensor_name(out_name)
            if alias and alias in name_to_module:
                tensor_to_mod[out_name] = name_to_module[alias]
                logger.debug(f"Tensor {out_name[:40]} -> module {alias} (via alias)")
            else:
                # Fallback: extract from node name or weight parameters
                module_name = None

                if node.name and node.name.startswith('/'):
                    parts = node.name[1:].split('/')
                    if len(parts) > 1:
                        module_name = '.'.join(parts[:-1])
                    else:
                        module_name = parts[0]

                # For BatchNorm and other ops with parameters, check all inputs
                if not module_name or module_name not in name_to_module:
                    for inp in node.input:
                        if any(param in inp for param in ['.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']):
                            # Extract module name progressively
                            inp_parts = inp.split('.')
                            for i in range(len(inp_parts)-1, 0, -1):
                                potential_name = '.'.join(inp_parts[:i])
                                if potential_name in name_to_module:
                                    module_name = potential_name
                                    break
                            if module_name:
                                break

                if module_name and module_name in name_to_module:
                    tensor_to_mod[out_name] = name_to_module[module_name]
                    logger.debug(f"Tensor {out_name[:40]} -> module {module_name} (via parameters)")

    # Helper to get channel count from tensor
    def get_channel_count(tensor_name: str) -> Optional[int]:
        """Get channel count from ONNX shape or node attributes"""
        # Try shape inference first
        if onnx_shapes_map:
            shape = onnx_shapes_map.get(tensor_name)
            if shape and len(shape) >= 2:
                return shape[1]  # NCHW format, C is dimension 1

        # Fallback to node attributes
        producer = producer_for_tensor.get(tensor_name)
        if producer:
            if producer.op_type == 'Conv' and len(producer.input) >= 2:
                weight_name = producer.input[1]
                for init in graph.initializer:
                    if init.name == weight_name and len(init.dims) >= 1:
                        return init.dims[0]  # out_channels

            elif producer.op_type == 'Gemm' and len(producer.input) >= 2:
                weight_name = producer.input[1]
                for init in graph.initializer:
                    if init.name == weight_name and len(init.dims) >= 1:
                        return init.dims[0]  # out_features

            elif producer.op_type == 'BatchNormalization' and len(producer.input) >= 2:
                weight_name = producer.input[1]
                for init in graph.initializer:
                    if init.name == weight_name and len(init.dims) >= 1:
                        return init.dims[0]  # num_features

        return None

    # Helper to recursively find module for a tensor
    mod_cache: Dict[str, Optional[nn.Module]] = {}

    def module_for_tensor(tname: str) -> Optional[nn.Module]:
        """Walk backwards through producers to find the module that created this tensor.
        ONNX exporter may prefix tensor names with scopes (e.g., "/dw/Conv_output_0").
        We try a few normalized variants to resolve producers/modules reliably.
        """
        # Prepare candidate name variants for robust lookup
        candidates = [tname]
        stripped = tname.lstrip('/')
        if stripped not in candidates:
            candidates.append(stripped)
        basename = stripped.split('/')[-1] if '/' in stripped else stripped
        if basename not in candidates:
            candidates.append(basename)

        # Fast-path: return from cache if any variant already computed
        for cand in candidates:
            if cand in mod_cache:
                return mod_cache[cand]

        # Direct hit: this tensor is an output we've already associated to a module
        for cand in candidates:
            if cand in tensor_to_mod:
                mod_cache[cand] = tensor_to_mod[cand]
                return tensor_to_mod[cand]

        # Try to find the producer of this tensor using any variant
        prod = None
        for cand in candidates:
            prod = producer_for_tensor.get(cand)
            if prod is not None:
                # Use the resolved variant as the key for cache tracking
                tname = cand
                break
        if prod is None:
            # This might be a graph input - check if it's produced by any earlier node
            # by looking at all nodes and their outputs
            mod_cache[candidates[0]] = None
            return None

        # Initialize cache entry to collect upstream modules (may remain empty)
        if tname not in mod_cache:
            mod_cache[tname] = []

        # Walk backwards through all inputs of the producer
        for inp in prod.input:
            # Skip parameters
            if any(param in inp for param in ['.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']):
                continue

            up_mod = module_for_tensor(inp)
            if up_mod is not None:
                # If recursive call returned a list of upstream modules, extend; else append
                if isinstance(up_mod, list):
                    if tname not in mod_cache:
                        mod_cache[tname] = list(up_mod)
                    else:
                        mod_cache[tname].extend(up_mod)
                else:
                    if tname not in mod_cache:
                        mod_cache[tname] = [up_mod]
                    else:
                        mod_cache[tname].append(up_mod)
        # Return collected upstream module(s); empty list means unresolved
        return mod_cache.get(tname)


    # Build dependency list
    dependencies: List[Tuple[str, str, DepType]] = []
    seen_edges = set()
    bypassed = []
    logger.debug(f"Processing {len(graph.node)} ONNX nodes...")

    for node in graph.node:
        logger.debug(f"\nNode: {node.op_type} | name: {node.name}")
        logger.debug(f"  Inputs: {node.input[:3]}")  # Show first 3 inputs
        logger.debug(f"  Outputs: {node.output}")

        # Find source modules from inputs
        src_modules = []
        src_tensors = []

        for inp in node.input:
            # Skip constant/initializer inputs (weights, biases, etc.)
            if any(param in inp for param in ['.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']):
                logger.debug(f"  Skipping parameter input: {inp[:50]}")
                continue

            src_mods = module_for_tensor(inp)
            for src_mod in make_safelist(src_mods):
                if src_mod is not None and src_mod not in src_modules:
                    src_modules.append(src_mod)
                    src_tensors.append(inp)
                    src_name = module_to_name.get(src_mod, "<??>")
                    logger.debug(f"  Found source: {src_name} (from tensor: {inp[:50]})")
                else:
                    logger.debug(f"  Could not find source module for input: {inp[:50]}")

        if not src_modules:
            logger.debug(f"  -> No source modules found, skipping")
            continue

        # Handle merge operations (Add, Sub, Sum, Concat, Mul, Div) - create REC dependencies between branches
        is_merge = node.op_type.capitalize() in ("Add", "Sub", "Sum", "Mul", "Div")
        is_concat = "Concat" in node.op_type.capitalize() or "Cat" in node.op_type.capitalize()

        if is_merge:
            logger.debug(f"  Detected merge operation: {node.op_type} with {len(src_modules)} source modules")
            logger.debug(f"  Source modules: {[module_to_name.get(m, '?') for m in src_modules]}")

        if is_merge and len(src_modules) >= 2:
            # Create REC dependencies between all pairs of source modules
            # These modules must have matching output dimensions for the merge to work
            for i in range(len(src_modules)):
                for j in range(i + 1, len(src_modules)):
                    mod_a = src_modules[i]
                    mod_b = src_modules[j]

                    name_a = module_to_name.get(mod_a, "")
                    name_b = module_to_name.get(mod_b, "")

                    if not name_a or not name_b:
                        continue

                    # Get channel counts for both modules to verify they match
                    tensor_a = src_tensors[i]
                    tensor_b = src_tensors[j]
                    channels_a = get_channel_count(tensor_a)
                    channels_b = get_channel_count(tensor_b)

                    logger.debug(f"  Checking REC: {name_a} (ch={channels_a}) <-> {name_b} (ch={channels_b})")

                    # For Add/Sub/Mul/Div, channels must match
                    # For Concat, channels can differ (concatenated along channel dim)
                    create_rec = False
                    if is_concat:
                        create_rec = True  # Always create REC for concat
                    elif channels_a is not None and channels_b is not None and channels_a == channels_b:
                        create_rec = True  # Channels match for Add/Sub/etc
                    elif channels_a is None or channels_b is None:
                        # Can't verify channels, but merge operation requires compatibility
                        create_rec = True

                    if create_rec:
                        # Add bidirectional REC edges
                        edge_key_ab = (name_a, name_b)
                        edge_key_ba = (name_b, name_a)

                        if edge_key_ab not in seen_edges:
                            logger.debug(f"  ✓ Adding REC dependency: {name_a} <-> {name_b}")
                            dependencies.append((mod_a, mod_b, DepType.REC))
                            seen_edges.add(edge_key_ab)

                        if edge_key_ba not in seen_edges:
                            logger.debug(f"  ✓ Adding REC dependency: {name_b} <-> {name_a}")
                            dependencies.append((mod_b, mod_a, DepType.REC))
                            seen_edges.add(edge_key_ba)
                    else:
                        logger.debug(f"  ✗ Skipping REC dependency (channel mismatch: {channels_a} vs {channels_b})")

        # Find destination module from outputs
        # First try: direct alias from output tensor name
        dst_mod = None
        dst_tensor = None

        for out_name in node.output:
            alias = _alias_from_tensor_name(out_name)
            if alias and alias in name_to_module:
                dst_mod = name_to_module[alias]
                dst_tensor = out_name
                logger.debug(f"  Found dest (method 1 - alias): {alias}")
                break

        # Second try: For ops that correspond to nn.Module, extract module from node itself
        if dst_mod is None:
            # Try to extract module name from the node itself
            if node.name and node.name.startswith('/'):
                parts = node.name[1:].split('/')
                # For ops like '/bn1/BatchNormalization', we want 'bn1'
                if len(parts) >= 2:
                    # The module name is typically everything except the last part (op type)
                    potential_names = [
                        '.'.join(parts[:-1]),  # e.g., 'model.bn1' from '/model/bn1/BatchNormalization'
                        parts[-2] if len(parts) >= 2 else parts[0],  # e.g., 'bn1' from above
                    ]
                    for pname in potential_names:
                        if pname in name_to_module:
                            dst_mod = name_to_module[pname]
                            if node.output:
                                dst_tensor = node.output[0]
                            logger.debug(f"  Found dest (method 2 - node name): {pname}")
                            break

            # Third try: For weight-based ops (Conv, Gemm, BatchNorm), check input parameters
            if dst_mod is None:
                for inp in node.input:
                    if any(param in inp for param in ['.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked']):
                        # Try progressive path building for nested modules
                        inp_parts = inp.split('.')
                        for i in range(len(inp_parts)-1, 0, -1):
                            potential_name = '.'.join(inp_parts[:i])
                            if potential_name in name_to_module:
                                dst_mod = name_to_module[potential_name]
                                if node.output:
                                    dst_tensor = node.output[0]
                                logger.debug(f"  Found dest (method 3 - params): {potential_name} (from {inp})")
                                break
                        if dst_mod:
                            break

        # Set bypass flag for Concat operations
        # The destination module after a Concat needs bypass=0 to track channel offset
        if is_concat and len(src_modules) >= 2:
            # if not hasattr(src_mod, 'bypass'):
                # src_mod.bypass = 0
            bypassed.extend(make_safelist(list(node.output)))
            logger.debug(f"  Setting bypass=0 for module after Concat: {module_to_name.get(src_mod, '?')}")
        for k in make_safelist(list(node.input)):
            if k in bypassed:
                if dst_mod is not None:
                    dst_mod.bypass = 0
                    logger.debug(f"  Setting bypass=0 for destination module: {module_to_name.get(dst_mod, '?')}")
                break

        # If no destination module found, skip as it s the end
        if dst_mod is None:
            logger.debug(f"  -> No destination module found, skipping")
            continue

        # Determine dependency type for each source -> destination connection
        for src_mod, src_tensor in zip(src_modules, src_tensors):
            # Skip self-connections
            if src_mod is dst_mod:
                continue

            src_name = module_to_name.get(src_mod, "")
            dst_name = module_to_name.get(dst_mod, "")

            # Skip empty names
            if not src_name or not dst_name:
                continue

            # Skip if already seen
            edge_key = (src_name, dst_name)
            if edge_key in seen_edges:
                continue

            # Get current channel counts
            src_channels = get_channel_count(src_tensor)
            dst_channels = get_channel_count(dst_tensor) if dst_tensor else None

            # # SEED NEURONS: Use ONNX metadata to seed neurons if possible
            # if src_channels is not None and src_channels > 0:
            #     src_mod.set_neurons('out_neurons', src_channels)
            #     if getattr(src_mod, 'wl_same_flag', False):
            #         src_mod.set_neurons('in_neurons', src_channels)

            # if dst_channels is not None and dst_channels > 0:
            #     dst_mod.set_neurons('in_neurons', dst_channels)
            #     if getattr(dst_mod, 'wl_same_flag', False):
            #         dst_mod.set_neurons('out_neurons', dst_channels)

            logger.debug(f"Analyzing dependency {src_name} -> {dst_name}")
            logger.debug(f"  Source channels: {src_channels}, Destination channels: {dst_channels}")

            # Use helper function to infer dependency type
            dep_type = _infer_dependency_type(dst_mod)

            # Set SAME Flag for modules if applicable
            if dep_type == DepType.SAME:
                dst_mod.wl_same_flag = True

            logger.debug(f"  ✓ Adding dependency: {src_name} -> {dst_name} [{dep_type.name}]")

            dependencies.append((src_mod, dst_mod, dep_type))
            seen_edges.add(edge_key)


    logger.info(f"Found {len(dependencies)} total dependencies")

    # Clean dependencies (remove duplicates and self-loops)
    dependencies = _clean_dependencies(dependencies)

    # Flag layers with constraints and propagate them
    _flag_layers_with_constraints(model, dependencies)

    return dependencies

def generate_index_maps(
    dependencies: List[Tuple[nn.Module, nn.Module, DepType]]
) -> Dict[str, Dict[str, th.Tensor]]:
    """
    Generate index mapping tensors for all layer dependencies.

    Args:
        dependencies: List of tuples (layer1_module, layer2_module, dependency_type)

    Returns:
        Updated dependencies with index mapping tensors for neuron correspondences.
    """

    for edge in dependencies:
        # Get src and dst modules and type
        src_mod, dst_mod, edge_label = edge[0], edge[1], edge[2]
        recursive_dep = edge_label == DepType.REC  # A recursive dependency ?

        # 1.1. Determine the number of neurons in each direction
        # # Src - First will always be is not None and int
        src_nb_neurons = src_mod.get_neurons(attr_name='out_neurons') if \
            not hasattr(src_mod, 'wl_transposed') \
            else src_mod.get_neurons(attr_name='in_neurons')

        # # Sanity check on dst
        dst_nb_neurons = dst_mod.get_neurons(attr_name='in_neurons') if not recursive_dep \
            and not hasattr(dst_mod, 'wl_transposed') \
            else dst_mod.get_neurons(attr_name='out_neurons')
        if dst_nb_neurons is None and src_nb_neurons is not None:
            dst_nb_neurons = src_nb_neurons
            dst_mod.set_neurons(
                'in_neurons' if not recursive_dep and not hasattr(dst_mod, 'wl_transposed') else 'out_neurons',
                dst_nb_neurons
            )  # So next will have neurons
            dst_mod_out_neurons = dst_mod.get_neurons(
                'in_neurons' if not (not recursive_dep and not hasattr(dst_mod, 'wl_transposed')) else 'out_neurons'
            )
            if dst_mod_out_neurons is None:
                dst_mod.set_neurons(
                    'in_neurons' if not (not recursive_dep and not hasattr(dst_mod, 'wl_transposed')) else 'out_neurons',
                    dst_nb_neurons
                )
        # # Sanity check on src
        if dst_nb_neurons is not None and src_nb_neurons is None:
            src_nb_neurons = dst_nb_neurons
            src_mod.set_neurons(
                'out_neurons' if not hasattr(src_mod, 'wl_transposed') else 'in_neurons',
                src_nb_neurons
            )  # So next will have neurons
            src_mod_out_neurons = src_mod.get_neurons(
                'out_neurons' if not hasattr(src_mod, 'wl_transposed') else 'in_neurons'
            )
            if src_mod_out_neurons is None:
                src_mod.set_neurons(
                    'out_neurons' if not hasattr(src_mod, 'wl_transposed') else 'in_neurons',
                    src_nb_neurons
                )
        # # Final sanity check
        if dst_nb_neurons is None and src_nb_neurons is None:
            raise ValueError(
                f"Cannot generate index maps for edge {src_mod.get_name_wi_id()} -> "
                f"{dst_mod.get_name_wi_id()}: both src and dst neurons are None"
            )
        dst_in_channels = range(dst_nb_neurons)

        # Get source out channels
        source_out_channels = range(src_nb_neurons)
        # # For multi-input / one output layers (e.g., Cat)
        if hasattr(dst_mod, 'bypass'):
            dst_in_channels = range(
                dst_mod.bypass,
                len(source_out_channels) + dst_mod.bypass
            )
            dst_mod.bypass += len(source_out_channels)

        # 1.2. Generate mappings tnsr for src and dst layers
        # Derive group sizes from wl_constraints entries
        def extract_group_size(mod: nn.Module, incoming: bool) -> Optional[int]:
            if hasattr(mod, 'wl_constraints') and isinstance(mod.wl_constraints, list):
                for entry in mod.wl_constraints:
                    if isinstance(entry, dict) and entry.get('name') == 'grouped' and entry.get('incoming') == incoming:
                        return entry.get('cons_group_size', None)
            return None

        cons_group_size_dst = extract_group_size(dst_mod, incoming=True)
        cons_group_size_src = extract_group_size(src_mod, incoming=False)
        src_to_dst_mapping_tnsr, dst_to_src_mapping_tnsr = \
            _generate_mappings(
                source_out_channels,
                dst_in_channels,
                dst_groups=(len(dst_in_channels) // cons_group_size_dst) if cons_group_size_dst is not None else None,
                src_groups=(len(source_out_channels) // cons_group_size_src) if cons_group_size_src is not None else None
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


if __name__ == '__main__':
    """
    Test script for generate_layer_dependencies_from_onnx function.

    This demonstrates how to extract layer dependencies from an ONNX model,
    including support for REC (residual/recursive) dependencies.
    """
    # Enable debug logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to see detailed merge operation detection
        format='%(levelname)s - %(message)s'
    )
    logger.setLevel(logging.DEBUG)

    import tempfile
    import torch.nn as nn
    from weightslab.utils.modules_dependencies import DepType
    from weightslab.backend.model_interface import ModelInterface


    # Test 1: Simple CNN without residual connections
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.relu2 = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 16 * 16, 256)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            return x


    # Test 2: Model with residual block (REC dependencies)
    class ResidualBlock(nn.Module):
        """Simple residual block with skip connection using Add operation"""
        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            # Residual connection using Add operation (creates REC dependency)
            out = out + identity
            out = th.relu(out)

            return out


    class ModelWithResidual(nn.Module):
        def __init__(self):
            super(ModelWithResidual, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()
            self.residual = ResidualBlock(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.residual(x)
            x = self.conv2(x)
            return x


    # Test 3: Residual CNN with Upsampling
    class ResidualBlockWithUpsampling(nn.Module):
        """Residual block with upsampling for encoder-decoder architectures"""
        def __init__(self, in_channels, out_channels):
            super(ResidualBlockWithUpsampling, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

            # Skip connection with channel adjustment if needed
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        def forward(self, x):
            identity = x
            if self.skip_conv is not None:
                identity = self.skip_conv(identity)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            out = out + identity
            out = th.relu(out)

            return out


    class ResidualCNNWithUpsampling(nn.Module):
        """
        Residual CNN with upsampling layers for semantic segmentation or super-resolution tasks.
        Architecture: Encoder (downsampling) -> Bottleneck -> Decoder (upsampling with residual blocks)
        """
        def __init__(self, in_channels=3, num_classes=10):
            super(ResidualCNNWithUpsampling, self).__init__()

            # Encoder: Downsampling path
            self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.enc_bn1 = nn.BatchNorm2d(64)
            self.enc_relu1 = nn.ReLU()

            self.enc_residual1 = ResidualBlockWithUpsampling(64, 64)
            self.enc_pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

            self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.enc_bn2 = nn.BatchNorm2d(128)
            self.enc_relu2 = nn.ReLU()

            self.enc_residual2 = ResidualBlockWithUpsampling(128, 128)
            self.enc_pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

            # Bottleneck
            self.bottleneck_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bottleneck_bn = nn.BatchNorm2d(256)
            self.bottleneck_relu = nn.ReLU()
            self.bottleneck_residual = ResidualBlockWithUpsampling(256, 256)

            # Decoder: Upsampling path
            self.dec_upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 8x8 -> 16x16
            self.dec_conv1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)  # Concatenate with encoder features
            self.dec_bn1 = nn.BatchNorm2d(128)
            self.dec_residual1 = ResidualBlockWithUpsampling(128, 128)

            # Final classification layer
            self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

        def forward(self, x):
            # Encoder
            enc1 = self.enc_conv1(x)
            enc1 = self.enc_bn1(enc1)
            enc1 = self.enc_relu1(enc1)
            enc1 = self.enc_residual1(enc1)
            enc1_pool = self.enc_pool1(enc1)

            enc2 = self.enc_conv2(enc1_pool)
            enc2 = self.enc_bn2(enc2)
            enc2 = self.enc_relu2(enc2)
            enc2 = self.enc_residual2(enc2)
            enc2_pool = self.enc_pool2(enc2)

            # Bottleneck
            bottleneck = self.bottleneck_conv(enc2_pool)
            bottleneck = self.bottleneck_bn(bottleneck)
            bottleneck = self.bottleneck_relu(bottleneck)
            bottleneck = self.bottleneck_residual(bottleneck)

            # Decoder with skip connections
            dec1 = self.dec_upsample1(bottleneck)
            dec1 = th.cat([dec1, enc2], dim=1)  # Skip connection via concatenation
            dec1 = self.dec_conv1(dec1)
            dec1 = self.dec_bn1(dec1)
            dec1 = self.dec_residual1(dec1)

            # Final output
            output = self.final_conv(dec1)

            return output


    class ResidualCNNWithUpsampling3X(nn.Module):
        """
        Residual CNN with 3x upsampling layers for semantic segmentation or super-resolution tasks.
        Architecture: Encoder (3x downsampling) -> Bottleneck -> Decoder (3x upsampling with residual blocks)
        Input 27x27 -> Encoder -> 3x3 -> Decoder -> 27x27
        """
        def __init__(self, in_channels=3, num_classes=10):
            super(ResidualCNNWithUpsampling3X, self).__init__()

            # Encoder: 3x Downsampling path using stride=3
            self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.enc_bn1 = nn.BatchNorm2d(64)
            self.enc_relu1 = nn.ReLU()

            self.enc_residual1 = ResidualBlockWithUpsampling(64, 64)
            self.enc_pool1 = nn.MaxPool2d(3, 3)  # 27x27 -> 9x9

            self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.enc_bn2 = nn.BatchNorm2d(128)
            self.enc_relu2 = nn.ReLU()

            self.enc_residual2 = ResidualBlockWithUpsampling(128, 128)
            self.enc_pool2 = nn.MaxPool2d(3, 3)  # 9x9 -> 3x3

            # Bottleneck
            self.bottleneck_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bottleneck_bn = nn.BatchNorm2d(256)
            self.bottleneck_relu = nn.ReLU()
            self.bottleneck_residual = ResidualBlockWithUpsampling(256, 256)

            # Decoder: Mixed upsampling (3x and 2x)
            self.dec_upsample1 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)  # 3x3 -> 9x9
            self.dec_conv1 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)  # Concatenate with encoder features
            self.dec_bn1 = nn.BatchNorm2d(128)
            self.dec_residual1 = ResidualBlockWithUpsampling(128, 128)

            self.dec_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 9x9 -> 18x18
            self.dec_conv2 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)  # Concatenate with encoder features
            self.dec_bn2 = nn.BatchNorm2d(64)
            self.dec_residual2 = ResidualBlockWithUpsampling(64, 64)

            # Final classification layer
            self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        def forward(self, x):
            # Encoder
            enc1 = self.enc_conv1(x)
            enc1 = self.enc_bn1(enc1)
            enc1 = self.enc_relu1(enc1)
            enc1 = self.enc_residual1(enc1)
            enc1_pool = self.enc_pool1(enc1)

            enc2 = self.enc_conv2(enc1_pool)
            enc2 = self.enc_bn2(enc2)
            enc2 = self.enc_relu2(enc2)
            enc2 = self.enc_residual2(enc2)
            enc2_pool = self.enc_pool2(enc2)

            # Bottleneck
            bottleneck = self.bottleneck_conv(enc2_pool)
            bottleneck = self.bottleneck_bn(bottleneck)
            bottleneck = self.bottleneck_relu(bottleneck)
            bottleneck = self.bottleneck_residual(bottleneck)

            # Decoder with skip connections
            dec1 = self.dec_upsample1(bottleneck)
            dec1 = th.cat([dec1, enc2], dim=1)  # Skip connection via concatenation
            dec1 = self.dec_conv1(dec1)
            dec1 = self.dec_bn1(dec1)
            dec1 = self.dec_residual1(dec1)

            dec2 = self.dec_upsample2(dec1)
            dec2 = th.cat([dec2, enc1], dim=1)  # Skip connection via concatenation
            dec2 = self.dec_conv2(dec2)
            dec2 = self.dec_bn2(dec2)
            dec2 = self.dec_residual2(dec2)

            # Final output
            output = self.final_conv(dec2)

            return output

    print("\n" + "="*80)
    print("TEST 3: Residual CNN with Upsampling (Encoder-Decoder)")
    print("="*80)

    model = ResidualCNNWithUpsampling(in_channels=3, num_classes=10)
    model.eval()
    dummy_input = th.randn(1, 3, 32, 32)

    print("\nModel Architecture:")
    print(model)

    model = ModelInterface(
        model,
        dummy_input=dummy_input,
        print_graph=False,
        use_onnx=True,
        compute_dependencies=True
    )
    dependencies3 = model.mapped_dependencies_with_ops

    # #############################
    # ######## OP. mix ############
    # #############################
    print('Performing model parameters operations..')
    model_op_neurons(model, dummy_input=dummy_input, rand=False, layer_id=15)

    print(f"\nFound {len(dependencies3)} dependencies in Residual CNN with Upsampling:")
    # Separate by type
    same_deps = [(s, d) for s, d, t in dependencies3 if t == DepType.SAME]
    incoming_deps = [(s, d) for s, d, t in dependencies3 if t == DepType.INCOMING]
    rec_deps = [(s, d) for s, d, t in dependencies3 if t == DepType.REC]

    print(f"\n  SAME Dependencies ({len(same_deps)}):")
    for src, dst in same_deps:  # Show first 5
        src_name = next((name for name, mod in model.named_modules() if mod is src), "<??>")
        dst_name = next((name for name, mod in model.named_modules() if mod is dst), "<??>")
        print(f"    [{src_name:30s}] --SAME-----> [{dst_name:30s}]")

    print(f"\n  INCOMING Dependencies ({len(incoming_deps)}):")
    for src, dst in incoming_deps:  # Show first 5
        src_name = next((name for name, mod in model.named_modules() if mod is src), "<??>")
        dst_name = next((name for name, mod in model.named_modules() if mod is dst), "<??>")
        print(f"    [{src_name:30s}] --INCOMING-> [{dst_name:30s}]")

    print(f"\n  REC Dependencies ({len(rec_deps)}):")
    for src, dst in rec_deps:  # Show first 5
        src_name = next((name for name, mod in model.named_modules() if mod is src), "<??>")
        dst_name = next((name for name, mod in model.named_modules() if mod is dst), "<??>")
        print(f"    [{src_name:30s}] <--REC----> [{dst_name:30s}]")

    print("""
Model Architecture Notes:
- Encoder: Downsamples spatial dimensions (Conv2d + MaxPool2d)
- Bottleneck: Central processing layer
- Decoder: Upsamples spatial dimensions (Upsample + Conv2d + Concatenate)
- Skip Connections: Concatenate encoder features with decoder features
- Residual Blocks: Used throughout for better gradient flow

Key Features:
- Bilinear interpolation for smooth upsampling
- Skip connections via concatenation (creates REC dependencies between branches)
- Residual blocks for identity shortcuts (creates REC dependencies)
    """)
