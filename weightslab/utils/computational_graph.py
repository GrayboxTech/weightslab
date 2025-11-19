import torch.nn as nn
import onnx
import onnx.shape_inference
from onnx import GraphProto
from typing import Iterable, List, Tuple, Dict, Optional
from copy import deepcopy
import collections


from weightslab.utils.modules_dependencies import DepType
from weightslab.utils.tools import (
    is_feature_producer,
    is_module_learnable,
    normalize_dicts,
)

def generate_mappings(
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


def _alias_from_tensor_name(tensor_name: str) -> Optional[str]:
    """
    Given an ONNX tensor name like '/c1/Conv_output_0', return 'c1'.

    If there's no leading '/', fall back to using the full name as alias.
    """
    if tensor_name.startswith("/"):
        parts = tensor_name.split("/")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return None
    # fallback: treat the whole name as alias
    return tensor_name


def get_onnx_shapes_map(onnx_file_path: str) -> Dict[str, Optional[Tuple[int, ...]]]:
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


def generate_graph_dependencies(
    model: nn.Module,
    graph: GraphProto,
    indexing_neurons: bool = True,
    onnx_shapes_map: Dict[str, Optional[Tuple[int, ...]]] = None,
) -> List[Tuple[nn.Module, nn.Module, DepType]]:
    """
    Pure ONNX version that:
    1) Traverses ONNX graph to produce raw (src_mod, dst_mod, DepType) edges for ALL modules.
    2) Bridges over non-ID modules (e.g. MaxPool, ReLU) so that final deps only connect
       neuron-aware modules (modules with get_module_id).
    3) Optionally (if indexing_neurons=True) builds neuron-level mappings
       between neuron-aware modules (i.e., modules that implement `get_neurons`).
    """

    name_to_module: Dict[str, nn.Module] = dict(model.named_modules())

    # Map tensor -> producing ONNX node
    producer_for_tensor: Dict[str, onnx.NodeProto] = {}
    tensor_to_mod: Dict[str, nn.Module] = {}

    for node in graph.node:
        for out_name in node.output:
            producer_for_tensor[out_name] = node

            # Try to see if this tensor belongs directly to a module
            alias = _alias_from_tensor_name(out_name)
            if alias is not None and alias in name_to_module:
                tensor_to_mod[out_name] = name_to_module[alias]

    
    mod_cache: Dict[str, Optional[nn.Module]] = {}

    def module_for_tensor_name(tname: str) -> Optional[nn.Module]:
        """
        Given any tensor name in the ONNX graph, walk backwards through its producers
        until we find the nearest nn.Module that produced something in that chain.
        """
        if tname in mod_cache:
            return mod_cache[tname]

        # Direct hit: this tensor is an output we've already associated to a module
        if tname in tensor_to_mod:
            mod_cache[tname] = tensor_to_mod[tname]
            return tensor_to_mod[tname]

        # Otherwise, see which node produced this tensor
        prod = producer_for_tensor.get(tname)
        if prod is None:
            mod_cache[tname] = None
            return None

        # Recursively walk through the producer's inputs
        for inp in prod.input:
            up_mod = module_for_tensor_name(inp)
            if up_mod is not None:
                mod_cache[tname] = up_mod
                return up_mod

        mod_cache[tname] = None
        return None


    def module_for_alias(alias: Optional[str]) -> Optional[nn.Module]:
        if alias is None:
            return None
        return name_to_module.get(alias, None)

    

    def chan_for_tensor(tname: str) -> Optional[int]:
        if onnx_shapes_map is None:
            return None
        shape = onnx_shapes_map.get(tname)
        if shape and len(shape) >= 2:
            # assume NCHW, C is dim 1
            return shape[1]
        return None


    raw_edges: List[Tuple[nn.Module, nn.Module, DepType]] = []      # ALL module edges (for bridging)
    filtered_edges: List[Tuple[nn.Module, nn.Module, DepType]] = [] # only structural/learnable dst (optional)


    for node in graph.node:
        # ---- STEP 1: Find source modules from node.inputs ----
        src_modules: List[nn.Module] = []
        src_tensors_for_shape: List[str] = []

        for inp in node.input:
            src_mod = module_for_tensor_name(inp)  # <--- NEW: deep mapping
            if src_mod is not None:
                src_modules.append(src_mod)
                src_tensors_for_shape.append(inp)

        # If no source modules, nothing to do for this node
        if not src_modules:
            continue

        # ---- STEP 2: Handle merge ops (REC) purely based on src_modules ----
        op_type = node.op_type
        is_merge = op_type in ("Add", "Sum", "Concat")
        if is_merge and len(src_modules) >= 2:
            for i in range(len(src_modules)):
                for j in range(i + 1, len(src_modules)):
                    mod_a = src_modules[i]
                    mod_b = src_modules[j]
                    raw_edges.append((mod_a, mod_b, DepType.REC))
                    filtered_edges.append((mod_a, mod_b, DepType.REC))

        # ---- STEP 3: Try to find destination module from node.outputs ----
        dst_alias = None
        dst_tensor_for_shape = None
        for out_name in node.output:
            a = _alias_from_tensor_name(out_name)
            if a is not None and a in name_to_module:
                dst_alias = a
                dst_tensor_for_shape = out_name
                break

        dst_mod = module_for_alias(dst_alias)
        if dst_mod is None:
            # e.g. Add/Concat/Relu that isn't directly tied to a module.
            # We have already captured REC edges above if this is a merge op.
            continue

        dst_is_structural = is_feature_producer(dst_mod)
        dst_is_learnable = is_module_learnable(dst_mod)

        dst_c = chan_for_tensor(dst_tensor_for_shape) if dst_tensor_for_shape else None

        # ---- STEP 4: SAME / INCOMING edges as before ----
        for src_mod, src_tname in zip(src_modules, src_tensors_for_shape):
            dep_type = DepType.INCOMING
            src_c = chan_for_tensor(src_tname) if src_tname else None

            if src_c is not None and dst_c is not None and src_c == dst_c:
                dep_type = DepType.SAME

            raw_edges.append((src_mod, dst_mod, dep_type))

            if dst_is_structural or dst_is_learnable:
                filtered_edges.append((src_mod, dst_mod, dep_type))


        # ---- REC edges between branches for merge ops ----
        op_type = node.op_type
        is_merge = op_type in ("Add", "Sum", "Concat")
        if is_merge and len(src_modules) >= 2:
            for i in range(len(src_modules)):
                for j in range(i + 1, len(src_modules)):
                    mod_a = src_modules[i]
                    mod_b = src_modules[j]
                    raw_edges.append((mod_a, mod_b, DepType.REC))
                    filtered_edges.append((mod_a, mod_b, DepType.REC))

        # DEBUG: print raw connectivity before bridging
    module_to_name = {m: n for n, m in name_to_module.items()}

    print("\n[DEBUG] RAW EDGES FROM ONNX:")
    for src_mod, dst_mod, dep_type in raw_edges:
        sname = module_to_name.get(src_mod, f"<?{type(src_mod).__name__}>")
        dname = module_to_name.get(dst_mod, f"<?{type(dst_mod).__name__}>")
        print(f"  {dep_type.name:8} {sname:15} -> {dname:15}")

    # ---- 1.5 BRIDGING over non-ID modules (use RAW edges) ----

    def is_id_module(m: nn.Module) -> bool:
        # Modules that are monkey-patched / neuron-aware and have IDs
        return hasattr(m, "get_module_id")

    out_edges: Dict[nn.Module, List[Tuple[nn.Module, DepType]]] = collections.defaultdict(list)
    for src_mod, dst_mod, dep_type in raw_edges:
        out_edges[src_mod].append((dst_mod, dep_type))

    bridged_deps: List[Tuple[nn.Module, nn.Module, DepType]] = []
    seen_edges = set()

    for src_mod in name_to_module.values():
        if not is_id_module(src_mod):
            continue

        # BFS from src_mod through non-ID modules
        queue = [src_mod]
        visited: set[nn.Module] = set()

        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)

            for nxt, dep_type in out_edges.get(cur, []):
                if nxt is src_mod:
                    continue

                if is_id_module(nxt):
                    key = (id(src_mod), id(nxt), dep_type)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        bridged_deps.append((src_mod, nxt, dep_type))
                else:
                    # Non-ID module (e.g., MaxPool, ReLU): keep walking
                    queue.append(nxt)

    # ---- 2. NEURON-LEVEL MAPPING (optional) ----

    if not indexing_neurons:
        return bridged_deps

    for (src_mod, dst_mod, edge_label) in bridged_deps:
        recursive_dep = edge_label == DepType.REC  # residual / multi-branch

        # Only build neuron mappings if BOTH modules are neuron-aware.
        if not hasattr(src_mod, "get_neurons") or not hasattr(dst_mod, "get_neurons"):
            continue

        # 2.1 Determine number of neurons for each side
        if not hasattr(src_mod, "wl_transposed"):
            src_n = src_mod.get_neurons(attr_name="out_neurons")
        else:
            src_n = src_mod.get_neurons(attr_name="in_neurons")

        if (not recursive_dep) and (not hasattr(dst_mod, "wl_transposed")):
            dst_n = dst_mod.get_neurons(attr_name="in_neurons")
        else:
            dst_n = dst_mod.get_neurons(attr_name="out_neurons")

        source_out_channels = range(src_n)
        dst_in_channels = range(dst_n)

        # Multi-input bypass (e.g. cat) handling if you still use dst_mod.bypass
        if hasattr(dst_mod, "bypass"):
            dst_in_channels = range(
                dst_mod.bypass,
                dst_mod.bypass + len(source_out_channels)
            )
            dst_mod.bypass += len(source_out_channels)

        # 2.2 Generate mappings, taking groups into account if present
        groups = None
        if hasattr(dst_mod, "groups"):
            groups = dst_mod.groups
        elif hasattr(src_mod, "groups"):
            groups = src_mod.groups

        if groups is not None:
            dst_groups = len(dst_in_channels) // groups
            src_groups = len(source_out_channels) // groups
        else:
            dst_groups = None
            src_groups = None

        src_to_dst_mapping_tnsr, dst_to_src_mapping_tnsr = generate_mappings(
            source_out_channels,
            dst_in_channels,
            dst_groups=dst_groups,
            src_groups=src_groups,
        )

        # 2.3 Register mappings on dst_mod and src_mod
        if not recursive_dep:
            # dst: child, src: parent
            dst_mod.dst_to_src_mapping_tnsrs.update(
                {src_mod.get_name_wi_id(): dst_to_src_mapping_tnsr}
            )
            if not hasattr(dst_mod, "bypass"):
                dst_mod.related_src_to_dst_mapping_tnsrs.update(
                    {dst_mod.get_name_wi_id(): deepcopy(src_to_dst_mapping_tnsr)}
                )
            dst_mod.dst_to_src_mapping_tnsrs = normalize_dicts(dst_mod.dst_to_src_mapping_tnsrs)
            dst_mod.related_src_to_dst_mapping_tnsrs = normalize_dicts(
                dst_mod.related_src_to_dst_mapping_tnsrs
            )
        else:
            # recursive: roles are swapped logically
            dst_mod.src_to_dst_mapping_tnsrs.update(
                {src_mod.get_name_wi_id(): dst_to_src_mapping_tnsr}
            )
            if not hasattr(dst_mod, "bypass"):
                dst_mod.related_dst_to_src_mapping_tnsrs.update(
                    {dst_mod.get_name_wi_id(): deepcopy(src_to_dst_mapping_tnsr)}
                )
            dst_mod.src_to_dst_mapping_tnsrs = normalize_dicts(dst_mod.src_to_dst_mapping_tnsrs)
            dst_mod.related_dst_to_src_mapping_tnsrs = normalize_dicts(
                dst_mod.related_dst_to_src_mapping_tnsrs
            )

        # src_mod side (always)
        src_mod.src_to_dst_mapping_tnsrs.update(
            {dst_mod.get_name_wi_id(): src_to_dst_mapping_tnsr}
        )
        if not hasattr(src_mod, "bypass"):
            src_mod.related_dst_to_src_mapping_tnsrs.update(
                {src_mod.get_name_wi_id(): deepcopy(dst_to_src_mapping_tnsr)}
            )
        src_mod.src_to_dst_mapping_tnsrs = normalize_dicts(src_mod.src_to_dst_mapping_tnsrs)
        src_mod.related_dst_to_src_mapping_tnsrs = normalize_dicts(
            src_mod.related_dst_to_src_mapping_tnsrs
        )

    return bridged_deps
