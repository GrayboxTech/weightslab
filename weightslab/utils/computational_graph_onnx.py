import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import GraphModule
import onnx
import onnx.shape_inference # Needed for infer_shapes
from typing import List, Tuple, Dict, Any, Optional
from weightslab.utils.modules_dependencies import DepType
from weightslab.utils.tools import *


def get_onnx_shapes_map(onnx_file_path: str) -> Dict[str, Optional[Tuple[int, ...]]]:
    """
    Loads the ONNX model, performs shape inference, and extracts a map of 
    intermediate tensor names (which match FX Node names) to their inferred shapes.
    """
    try:
        model = onnx.load(onnx_file_path)
        # Perform static shape inference to populate the graph's value_info
        inferred_model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        # Fallback if shape inference fails (e.g., dynamic axes not handled)
        print(f"Error during ONNX shape inference. Proceeding with limited shape info: {e}")
        return {}

    # Map: tensor_name (FX Node Name) -> (shape_tuple)
    shapes_map: Dict[str, Optional[Tuple[int, ...]]] = {}

    # Get shapes for graph inputs, outputs, and intermediate tensors
    for tensor_info in list(inferred_model.graph.input) + \
                        list(inferred_model.graph.output) + \
                        list(inferred_model.graph.value_info):

        type_info = tensor_info.type.tensor_type
        if type_info.shape:
            shape = tuple(d.dim_value for d in type_info.shape.dim)
            # Only record valid, fully determined shapes
            shapes_map[tensor_info.name] = shape if all(d > 0 for d in shape) else None

    return shapes_map

# --- CORE ADAPTED FUNCTION ---

def generate_graph_dependencies_onnx_aware(
    model: nn.Module,
    traced_graph: GraphModule,
    onnx_shapes_map: Dict[str, Optional[Tuple[int, ...]]]
) -> List[Tuple[nn.Module, nn.Module, str]]:
    """
    Infers dependencies from the PyTorch FX graph, using shapes retrieved 
    from the ONNX model to determine the dependency type (INCOMING, SAME, or REC).
    """
    dependencies = []
    # Map: FX Node -> list_of_structural_module_instances
    node_to_module = {}

    def get_feature_channel_size(node: fx.Node) -> Optional[int]:
        """Looks up the channel dimension (index 1) from the ONNX shape map."""
        shape = onnx_shapes_map.get(node.name)
        # Assuming NCHW format, the channel size is at index 1 (C)
        return shape[1] if shape and len(shape) >= 2 else None

    # Iterate over the nodes in the FX graph
    for node in traced_graph.graph.nodes:
        current_module = None

        if node.op == 'call_module':
            # 1. Get current module instance and properties
            current_module = get_module_by_name(model, node.target)
            
            is_dst_structural = is_feature_producer(current_module)
            is_learnable = is_module_learnable(current_module)
            current_module.layer_type = 1 if is_dst_structural else -1 

            # Find the primary input source node 
            source_node = next(
                (arg for arg in node.args if isinstance(arg, fx.Node)),
                None
            )
            source_modules = node_to_module.get(source_node) if source_node else None

            # --- 1. Dependency Creation (Structural Source -> Structural Destination) ---
            if source_modules:
                for source_module in source_modules:
                    if source_module is not None and (is_dst_structural or is_learnable):
                        
                        dep_type = DepType.INCOMING 
                        source_out_channels = get_feature_channel_size(source_node)
                        dst_out_channels = get_feature_channel_size(node)

                        # Logic to check for SAME constraint (e.g., identity or simple sequential same size)
                        if source_out_channels is not None and dst_out_channels is not None:
                            # If input and output channels are the same, it often implies a SAME constraint
                            if source_out_channels == dst_out_channels:
                                dep_type = DepType.SAME
                        
                        dependencies.append((source_module, current_module, dep_type))

            # --- 2. Update Tracking Map ---
            if current_module.layer_type >= 1 or is_learnable:
                # Structural layer: tracks itself
                node_to_module[node] = make_safelist(current_module)
            elif source_node and source_node in node_to_module:
                # Stateless layer (e.g., ReLU, MaxPool): passes through source module(s)
                node_to_module[node] = node_to_module[source_node]
            else:
                # Fallback
                node_to_module[node] = make_safelist(current_module) 

        # --- Handle Merge Operations (call_function/call_method like add, cat) ---
        elif node.op == 'call_function' or node.op == "call_method":
            
            # Check for merge operations
            if 'add' in str(node.target) or 'cat' in str(node.target):
                
                all_source_producers = []
                
                # Recursive helper to find all structural modules feeding into this merge node
                def find_source_modules_recursive(arg):
                    if isinstance(arg, fx.Node):
                        producers = node_to_module.get(arg)
                        if producers:
                            all_source_producers.extend(producers)
                    elif isinstance(arg, (tuple, list)):
                        for item in arg:
                            find_source_modules_recursive(item)
                
                for arg in node.args:
                    find_source_modules_recursive(arg)

                distinct_source_modules = list(set(all_source_producers))

                # 2. Check for multi-branch constraint
                if len(distinct_source_modules) >= 2:
                    # Apply bidirectional SAME constraint (REC) between all merging pairs
                    for i in range(len(distinct_source_modules)):
                        for j in range(i + 1, len(distinct_source_modules)):
                            mod_a = distinct_source_modules[i]
                            mod_b = distinct_source_modules[j]
                            # Use REC for the constraint imposed by the merge operation (e.g., residual path)
                            dependencies.append((mod_a, mod_b, DepType.REC))
                
                # 3. Update the module map (The output is tracked as dependent on all sources)
                node_to_module[node] = distinct_source_modules

            # Handle single-input stateless functions (like torch.sigmoid, view, etc.)
            else:
                source_node = next((arg for arg in node.args if isinstance(arg, fx.Node)), None)
                if source_node and source_node in node_to_module:
                    node_to_module[node] = node_to_module[source_node]
                else:
                    node_to_module[node] = [] 

    return dependencies


# --- EXAMPLE WORKFLOW ---
# 1. Define a dummy model with a residual connection
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # Merge point enforcing the SAME channel constraint between 'out' and 'identity'
        out = torch.add(out, identity) 
        return out

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.block1 = ResidualBlock(16)
        self.final_linear = nn.Linear(16 * 32 * 32, 10) 

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block1(x) 
        x = torch.flatten(x, 1)
        return self.final_linear(x)


if __name__ == '__main__':
    # Setup
    model = MyModel().eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    onnx_file_path = "onnx_traced_dependencies.onnx"

    # --- Step 1 & 2: FX Trace and Export to ONNX ---
    traced_model = fx.symbolic_trace(model)
    print("PyTorch FX tracing complete.")
    
    print(f"Exporting model to {onnx_file_path}...")
    try:
        torch.onnx.export(
            model, dummy_input, onnx_file_path, 
            opset_version=14,
            input_names=['input'], output_names=['output']
        )
        print("ONNX export complete.")
    except Exception as e:
        print(f"ONNX Export Error: {e}")
        print("Could not proceed with dependency analysis as ONNX export failed.")
        exit() 

    # --- Step 3: Infer Shapes from ONNX ---
    onnx_shapes = get_onnx_shapes_map(onnx_file_path)
    print("ONNX shapes map generated.")

    # --- Step 4: Run Combined Dependency Analysis ---
    dependencies = generate_graph_dependencies_onnx_aware(
        model=model,
        traced_graph=traced_model,
        onnx_shapes_map=onnx_shapes
    )

    print("\n" + "="*80)
    print("FINAL MODULE-TO-MODULE DEPENDENCIES (FX + ONNX Shape Awareness)")
    print("="*80)
    
    # Print the dependencies using the PyTorch module names
    for src, dst, dep_type in dependencies:
        # Find the module name using named_modules()
        src_name = next((name for name, mod in model.named_modules() if mod is src), "Input")
        dst_name = next((name for name, mod in model.named_modules() if mod is dst), "Output")
        
        print(f"{src_name:<25} -> {dst_name:<25} | Type: {dep_type:<10}")
    
    print("\nREC dependencies indicate a cross-branch, 'SAME channel size' constraint (like a residual connection).")