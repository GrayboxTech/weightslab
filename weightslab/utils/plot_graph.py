import graphviz
import logging
import torch as th
import torch.nn as nn

from typing import Tuple, List
from torch.fx import GraphModule, Node
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

from weightslab.models.model_with_ops import DepType


# Global logger
logger = logging.getLogger(__name__)


# Helper to retrieve module instance (needed for custom deps)
def get_module_by_name(model, name):
    try:
        return model.get_submodule(name)
    except AttributeError:
        return getattr(model, name, None)


def get_shape_string(node: Node) -> str:
    """Safely extracts and formats the tensor shape from a node's metadata."""
    output_shape = "N/A"
    if 'tensor_meta' in node.meta and node.meta['tensor_meta'] is not None:
        meta = node.meta['tensor_meta']
        if isinstance(meta, th.Tensor) or isinstance(meta, TensorMetadata):
            output_shape = str(tuple(meta.shape))
        # Fix DOT syntax error by replacing parentheses with brackets
        # output_shape = output_shape.replace('(', '[').replace(')', ']')
    return output_shape


def make_safelist(x):
    return [x] if not isinstance(x, list) else x


def plot_fx_graph_with_details(
        traced_model: GraphModule,
        custom_dependencies: List[Tuple[nn.Module, nn.Module, DepType]],
        filename="fx_trace_detailed"
        ):
    """
    Generates a graphviz plot of the th.fx graph, displaying tensor shapes
    on nodes and dependency types (INCOMING/SAME) on edges.
    Includes DOT syntax fixes.
    """
    comment = traced_model.name if hasattr(traced_model, 'name') else 'Model'
    graph_attr = {
        'rankdir': 'TB',
        'splines': 'ortho',
        'nodesep': '0.5',
        'ranksep': '0.75'
    }
    node_attr = {
        'shape': 'box',
        'style': 'filled',
        'color': 'lightblue'
    }

    # Instanciate graphviz
    dot = graphviz.Digraph(
        comment=comment,
        graph_attr=graph_attr,
        node_attr=node_attr
    )

    # 1. Build Nodes and INCOMING Edges (Standard FX Trace + Shapes)
    fx_to_gv_node, module_to_fx_node, module_id = {}, {}, 0
    for node in traced_model.graph.nodes:
        node_name = str(node)

        # --- LABELING ---
        current_module = None
        if node.op == 'call_module':
            current_module = get_module_by_name(traced_model, node.target)
            if current_module:
                module_to_fx_node[current_module] = node_name
            module_id_str = current_module.get_module_id() if \
                hasattr(
                    current_module, 'get_module_id'
                ) else None
            if 'conv' in current_module._get_name().lower() or \
                    'batch' in current_module._get_name().lower() or \
                    'linear' in current_module._get_name().lower():
                if module_id_str is None:
                    module_id_str = str(module_id)
                    module_id += 1
                else:
                    module_id = int(module_id_str) + 1

        # Define label and color
        label = f"{node.name}\n({node.op})"
        color = '#DCE6F1'
        if node.op == 'call_module':
            module_class = type(current_module).__name__ if current_module \
                else node.target
            label = f"ID={module_id_str}\n({node.target})\n{module_class}"
            color = '#8EBAD9'
        elif node.op == 'call_function':
            call_function_name = node.target.__name__ if \
                hasattr(node.target, '__name__') else str(node.target)
            label = f"Function: {call_function_name}"
            color = '#A6D9A6'
        elif node.op == 'placeholder':
            label = f"Input: {node.name}"
            color = '#F5D1A2'
        elif node.op == 'output':
            label = "Output"
            color = '#F5A2A2'
        label_content = label

        # Add the node
        dot.node(node_name, label=label_content, fillcolor=color)
        fx_to_gv_node[node] = node_name

        # Add INCOMING Edges (Data Flow with Source and Destination Shapes)
        for arg in node.args:
            if isinstance(arg, th.fx.Node):
                # Get the output shape of the source node (arg)
                src_shape = get_shape_string(arg)
                # Get the output shape of the destination node (node)
                # This shows the shape transformation at the destination node
                dst_shape = get_shape_string(node)
                # Edge label now includes the dependency type,
                # source output shape, and destination output shape
                edge_label = f"Src Output: {src_shape}" + \
                    f"\nDst Output: {dst_shape}"
                dot.edge(
                    str(arg),
                    node_name,
                    label=edge_label,
                    style="solid",
                    color="#2F4F4F"
                )

    logger.info("\n--- Plotting Phase: Detected Constraints/Dependencies ---")
    for srcs_mod, dests_mod, dep_type in custom_dependencies:
        if not isinstance(dests_mod, list):
            dests_mod = make_safelist(dests_mod)
        if not isinstance(srcs_mod, list):
            srcs_mod = make_safelist(srcs_mod)
        for src_mod in srcs_mod:
            for dest_mod in dests_mod:
                src_node_name = module_to_fx_node.get(src_mod)
                dest_node_name = module_to_fx_node.get(dest_mod)

                if src_node_name and dest_node_name:
                    if dep_type == DepType.SAME:
                        # Sequential SAME constraint (e.g., Conv -> BN)
                        label_text = "SAME"

                    elif dep_type == DepType.REC:
                        # Sequential REC constraint:
                        # e.g., Conv1 -> Add_1; Conv2 -> Add_1
                        # so REC dep. are here: Conv1 <> REC <> Conv2
                        label_text = "REC"

                    elif dep_type == DepType.INCOMING:
                        # Data flow dependency (e.g., BN -> Conv) - PLOT THIS
                        label_text = "INCOMING"

                    # Plot labeled edge
                    dot.edge(
                        src_node_name, dest_node_name,
                        label=label_text,
                        style="dashed",
                        dir="forward",
                        color="#2F4F4F"
                    )

    # Render the graph
    try:
        dot.render(filename, view=True, format='pdf', overwrite_source=True)
        logger.debug(f'Graph rendered at {filename}')
    except graphviz.backend.execute.CalledProcessError as e:
        logger.error(f'No graph rendering as {e}')


if __name__ == "__main__":
    from weightslab.utils.tools import generate_graph_dependencies
    from torch.fx import symbolic_trace, GraphModule, Node

    print('Hello World')

    # Define a standard model architecture
    class FashionCNNSequential(nn.Module):
        def __init__(self):
            super().__init__()

            # Feature Blocks (Same as before)
            # Block 1
            self.c1 = nn.Conv2d(1, 4, 3, padding=1)
            self.b1 = nn.BatchNorm2d(4)
            self.r1 = nn.ReLU()
            self.m1 = nn.MaxPool2d(2)

            # Block 2
            self.c2 = nn.Conv2d(4, 4, 3)  # Default stride=1, no padding
            self.b2 = nn.BatchNorm2d(4)
            self.r2 = nn.ReLU()
            self.m2 = nn.MaxPool2d(2)

            # Classifier Block (Includes Flatten)
            self.f3 = nn.Flatten()  # Automatically flattens to (B, CxHxW)
            self.fc3 = nn.Linear(in_features=4 * 6 * 6, out_features=10)
            self.s3 = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.m1(self.r1(self.b1(self.c1(x))))
            x = self.m2(self.r2(self.b2(self.c2(x))))
            x = self.s3(self.fc3(self.f3(x)))
            return x

    model = FashionCNNSequential()
    # 1. Trace the model and propagate tensor shapes:
    # required for accurate 'output_shape'
    traced_model = symbolic_trace(model)  # symbolic_trace(model)
    dummy_input = th.randn(1, 1, 28, 28)
    ShapeProp(traced_model).propagate(dummy_input)

    # 2. Generate the dependencies
    dependencies = generate_graph_dependencies(model, traced_model)

    # 3. Plot the final graph
    print("--- 3. Plotting Graph with Details ---")
    plot_fx_graph_with_details(
        traced_model=traced_model,
        custom_dependencies=dependencies
    )
    print('Bye World')
