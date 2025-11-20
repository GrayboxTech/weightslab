import torch as th

from torch.fx.passes.shape_prop import ShapeProp
# from weightslab.utils.shape_prop import ShapeProp
from torch.fx import symbolic_trace

from weightslab.components.tracking import TrackingMode
from weightslab.models.model_with_ops import NetworkWithOps
from weightslab.modules.neuron_ops import NeuronWiseOperations

from weightslab.utils.plot_graph import plot_fx_graph_with_details
from weightslab.utils.logs import print, setup_logging
from weightslab.models.monkey_patcher import monkey_patch
from weightslab.utils.tools import model_op_neurons
from weightslab.utils.computational_graph import \
    generate_graph_dependencies


class WatcherEditor(NetworkWithOps):
    def __init__(
            self,
            model: th.nn.Module,
            dummy_input: th.Tensor = None,
            device: str = 'cpu',
            print_graph: bool = False,
            print_graph_filename: str = None):
        """
        Initializes the WatcherEditor instance.

        This constructor sets up the model for watching and editing by tracing
        it, propagating shapes, generating graph visualizations, patching the
        model with WeightsLab features, and defining layer dependencies.

        Args:
            model (th.nn.Module): The PyTorch model to be wrapped and edited.
            dummy_input (th.Tensor, optional): A dummy input tensor required for
                symbolic tracing and shape propagation. Defaults to None.
            device (str, optional): The device ('cpu' or 'cuda') on which the model
                and dummy input should be placed. Defaults to 'cpu'.
            print_graph (bool, optional): If True, a visualization of the model's
                computational graph will be generated. Defaults to False.
            print_graph_filename (str, optional): The filename for saving the
                generated graph visualization. Required if `print_graph` is True.
                Defaults to None.

        Returns:
            None: This method initializes the object and does not return any value.
        """
        super(WatcherEditor, self).__init__()

        # Reinit IDS when instanciating a new torch model
        NeuronWiseOperations().reset_id()

        # Define variables
        # # Disable tracking for implementation
        self.tracking_mode = TrackingMode.DISABLED
        self.name = "Test Architecture Model"
        self.model = model.to(device)
        if dummy_input is not None:
            self.dummy_input = dummy_input.to(device)
        else:
            self.dummy_input = th.randn(model.input_shape).to(device)
        self.print_graph = print_graph
        self.print_graph_filename = print_graph_filename
        self.traced_model = None  # symbolic_trace(model)
        # self.traced_model.name = "N.A."

        # # Propagate the shape over the graph
        # self.shape_propagation()

        # # Generate the graph vizualisation
        # self.generate_graph_vizu()

        # Patch the torch model with WeightsLab features
        # self.monkey_patch_model()

        # Generate the graph dependencies
        self.define_deps()

        # Clean
        del self.traced_model

    def __enter__(self):
        """
        Executed when entering the 'with' block.

        This method is part of the context manager protocol. It is called
        when the 'with' statement is entered, allowing for setup operations
        or resource acquisition.

        Returns:
            WatcherEditor: The instance of the WatcherEditor itself, which
            will be bound to the variable after 'as' in the 'with' statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Executed when exiting the 'with' block (whether by success or error).

        This method is part of the context manager protocol. It is called
        when the 'with' statement is exited, allowing for cleanup operations
        or resource release. It resets the `visited_nodes` set and handles
        any exceptions that might have occurred within the 'with' block.

        Args:
            exc_type (Optional[Type[BaseException]]): The type of the exception
                that caused the 'with' block to be exited. None if no exception occurred.

        Returns:
            bool: False if an exception occurred and it should be re-raised,
            or if no exception occurred and the context manager handled its exit.
            True if an exception occurred and it was successfully handled by
            this method, preventing it from being re-raised.
        """
        self.visited_nodes = set()  # Reset NetworkWithOps nodes visited
        if exc_type is not None:
            print(
                f"[{self.__class__.__name__}]: An exception occurred: \
                    {exc_type.__name__} with {exc_val} and {exc_tb}.")
            return False
        return False

    def monkey_patch_model(self):
        """Applies monkey patching to the model's modules.

        This method iterates through all submodules of the `self.model` and applies
        a monkey patch. The purpose of this patching is to inject additional
        functionality, specifically `LayerWiseOperations`, into each `torch.nn.Module`
        instance, enabling features like neuron-wise tracking and manipulation.

        Args:
            self: The instance of the WatcherEditor class.

        Returns:
            None: This method modifies the `self.model` in-place and does not
            return any value.
        """
        self.model.apply(monkey_patch)

    def shape_propagation(self):
        """Propagates shapes through the traced model.

        This method uses `torch.fx.passes.shape_prop.ShapeProp` to infer and
        attach shape information (input and output dimensions) to each node
        in the `self.traced_model`'s computational graph. This shape information
        is crucial for generating graph visualizations and defining dependencies
        between layers.

        Args:
            self: The instance of the WatcherEditor class.

        Returns:
            None: This method modifies the `self.traced_model` in-place by
            adding shape metadata to its nodes and does not return any value.
        """
        ShapeProp(self.traced_model).propagate(self.dummy_input)

    def children(self):
        """
        Generates a list of all immediate child modules (layers) of the wrapped model.

        This method provides access to the direct submodules of the `self.model`,
        which are typically the individual layers or sequential blocks defined
        within the PyTorch model.

        Returns:
            list[torch.nn.Module]: A list containing all immediate child modules
            of the `self.model`.
        """
        # Return every model layers with ops
        childs = list(self.model.children())
        return childs

    def generate_graph_vizu(self):
        """Generates a visualization of the model's computational graph.

        This method creates a visual representation of the `self.traced_model`'s
        computational graph, including details about dependencies between layers.
        The visualization is generated only if `self.print_graph` is True.
        It uses `generate_graph_dependencies` to determine the connections
        and `plot_fx_graph_with_details` to render the graph to a file.

        Args:
            self: The instance of the WatcherEditor class.

        Returns:
            None: This method does not return any value; it generates a file
            as a side effect if `self.print_graph` is True.
        """
        if self.print_graph:
            print("--- Generated Graph Dependencies (FX Tracing) ---")
            or_dependencies = generate_graph_dependencies(
                self.model,
                self.traced_model,
                indexing_neurons=False
            )
            plot_fx_graph_with_details(
                self.traced_model,
                custom_dependencies=or_dependencies,
                filename=self.print_graph_filename
            )

    def define_deps(self):
        """Generates and registers the computational graph dependencies for the model.

        This method first calls `generate_graph_dependencies` to determine the
        connections and data flow between the layers of the `self.model` based
        on its `self.traced_model`. These dependencies are then stored in
        `self.dependencies_with_ops` and subsequently registered with the
        `WatcherEditor` instance using `self.register_dependencies`.
        This registration is crucial for operations that require understanding
        the model's structure and layer relationships.

        Args:
            self: The instance of the WatcherEditor class.

        Returns:
            None: This method modifies the instance's state by setting
            `self.dependencies_with_ops` and calling `self.register_dependencies`.
        """
        # --- Step 1 & 2: FX Trace and Export to ONNX ---
        print("PyTorch FX tracing complete.")

        onnx_file_path = "onnx_traced_dependencies.onnx"
        print(f"Exporting model to {onnx_file_path}...")
        th.onnx.export(
            self.model, self.dummy_input, onnx_file_path,
            opset_version=14,
            input_names=['input'], output_names=['output'],
            do_constant_folding=False
        )
        print("ONNX export complete.")

        import builtins
        def debug_print_onnx_nodes(onnx_file_path: str):
            import onnx

            model = onnx.load(onnx_file_path)
            graph = model.graph

            builtins.print("\n[DEBUG] ONNX NODES:")
            for i, node in enumerate(graph.node):
                builtins.print(f"Node {i}: op_type={node.op_type}")
                builtins.print(f"  inputs : {list(node.input)}")
                builtins.print(f"  outputs: {list(node.output)}")

            builtins.print("\n[DEBUG] ONNX VALUE_INFOS (tensor names + shapes if available):")
            for v in list(graph.input) + list(graph.value_info) + list(graph.output):
                t = v.type.tensor_type
                if t.HasField("shape"):
                    dims = []
                    for d in t.shape.dim:
                        if d.HasField("dim_value"):
                            dims.append(d.dim_value)
                        elif d.HasField("dim_param"):
                            dims.append(f"{d.dim_param}")
                        else:
                            dims.append("?")
                    builtins.print(f"  {v.name}: {dims}")
                else:
                    builtins.print(f"  {v.name}: <no shape>")

        builtins.print("=== ONNX GRAPH ===")
        debug_print_onnx_nodes(onnx_file_path)

        self.monkey_patch_model()
        import onnx
        import onnx.shape_inference
        from typing import Tuple, Dict, Optional
        from weightslab.utils.computational_graph_onnx import \
            generate_graph_dependencies_onnx_aware

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
                    name = tensor_info.name
                    if name.startswith("/"):
                        # TODO (GP): here we do not handle several outputs shapes for same layers
                        shapes_map[name.split('/')[1]] = shape if all(d > 0 for d in shape) else None
                    else:
                        shapes_map[name] = shape if all(d > 0 for d in shape) else None
            return shapes_map
        
        from typing import Dict, Tuple, Optional, Any
        import onnx 
        # Note: You may need to install the 'onnx' Python package: pip install onnx

        def get_intermediate_onnx_shapes(onnx_file_path: str) -> Dict[str, Tuple[int, ...]]:
            """
            Loads an ONNX model, infers all intermediate tensor shapes using the 
            'onnx' library, and generates the dictionary mapping tensor names to shapes.

            Args:
                onnx_file_path: Path to the ONNX model file.

            Returns:
                A dictionary mapping ONNX tensor output paths (e.g., '/c1/Conv_output_0') 
                to their inferred shape tuples.
            """
            try:
                # 1. Load the ONNX model from the file
                model = onnx.load(onnx_file_path)
                
                # 2. Run Shape Inference (The Critical Step)
                # This calculates and populates the shapes for all intermediate tensors.
                inferred_model = onnx.shape_inference.infer_shapes(model)
                graph = inferred_model.graph
                
            except FileNotFoundError:
                print(f"Error: ONNX file not found at {onnx_file_path}")
                return {}
            except Exception as e:
                print(f"An error occurred during ONNX loading or shape inference: {e}")
                return {}

            shapes_map: Dict[str, Tuple[int, ...]] = {}

            # 3. Collect all tensors whose shape we need (intermediate and final outputs)
            # 'value_info' holds the metadata, including shapes, for most intermediate tensors.
            all_value_infos = list(graph.value_info)
            all_value_infos.extend(list(graph.output)) # Include the final outputs as well

            # 4. Iterate and extract the shape data
            for value_info in all_value_infos:
                tensor_name = value_info.name
                dims = []
                
                # Extract the shape from the tensor_type field
                if value_info.type.tensor_type.HasField('shape'):
                    for dim in value_info.type.tensor_type.shape.dim:
                        if dim.HasField('dim_value'):
                            # Static dimension (e.g., 4)
                            dims.append(dim.dim_value)
                        elif dim.HasField('dim_param'):
                            # Dynamic dimension (e.g., 'batch_size'). Use -1 as a placeholder.
                            dims.append(-1) 
                        else:
                            dims.append(-1) # Unknown
                
                if dims:
                    shapes_map[tensor_name] = tuple(dims)
                    
            print(f"Successfully extracted shapes for {len(shapes_map)} tensors from the ONNX graph.")
            return shapes_map

        # --- The Pure ONNX Lookup Function (To use the generated dictionary) ---

        def get_onnx_shape_by_alias(
            alias_name: str,
            onnx_shapes_map: Dict[str, Tuple[int, ...]]
        ) -> Optional[Tuple[int, ...]]:
            """
            Queries the generated dictionary using the FX-derived alias (e.g., 'c1', 'fc4').
            """
            if not onnx_shapes_map:
                return None

            # We assume the alias is embedded in the ONNX tensor name as a prefix: /alias_name/
            expected_prefix = f"/{alias_name}/"
            
            for key in onnx_shapes_map.keys():
                # This finds the shape for the first tensor output by the node 'alias_name'.
                if key.startswith(expected_prefix):
                    return onnx_shapes_map[key]
                    
            return None
        onnx_shapes_map = get_intermediate_onnx_shapes(onnx_file_path)

        model_onnx = onnx.load(onnx_file_path)
        graph = model_onnx.graph

        self.dependencies_with_ops = generate_graph_dependencies(#_onnx_aware(
            model=self.model,
            graph=graph,
            onnx_shapes_map=onnx_shapes_map
        )

        # Generate the dependencies
        # self.dependencies_with_ops = generate_graph_dependencies(
        #     self.model,
        #     self.traced_model
        # )

        # Register the layers dependencies
        self.register_dependencies(self.dependencies_with_ops)
        
        # Print the dependency list for debugging
        self._print_dependency_list()

    def _print_dependency_list(self):
        """Print the registered dependencies for debugging purposes."""
        import builtins
        
        builtins.print("\n" + "="*80)
        builtins.print("WEIGHTSLAB DEPENDENCY GRAPH")
        builtins.print("="*80)
        
        if not hasattr(self, 'dependencies_with_ops') or not self.dependencies_with_ops:
            builtins.print("No dependencies found!")
            return
        
        # Create a mapping from module to name for readable output
        module_to_name = {}
        for name, module in self.model.named_modules():
            module_to_name[id(module)] = name if name else '<root>'
        
        builtins.print(f"\nTotal dependencies: {len(self.dependencies_with_ops)}\n")
        
        # Group by dependency type
        from collections import defaultdict
        deps_by_type = defaultdict(list)
        
        for src_mod, dst_mod, dep_type in self.dependencies_with_ops:
            deps_by_type[dep_type].append((src_mod, dst_mod))
        
        # Print dependencies grouped by type
        for dep_type, deps in deps_by_type.items():
            builtins.print(f"\n{dep_type.name} Dependencies ({len(deps)}):")
            builtins.print("-" * 80)
            
            for src_mod, dst_mod in deps:
                src_name = module_to_name.get(id(src_mod), f"<unknown {type(src_mod).__name__}>")
                dst_name = module_to_name.get(id(dst_mod), f"<unknown {type(dst_mod).__name__}>")
                
                # Get module IDs if available
                src_id = src_mod.get_module_id() if hasattr(src_mod, 'get_module_id') else 'N/A'
                dst_id = dst_mod.get_module_id() if hasattr(dst_mod, 'get_module_id') else 'N/A'
                
                builtins.print(f"  [{src_id:3}] {src_name:40} -> [{dst_id:3}] {dst_name:40}")
                
                # Show neuron mappings if available
                if hasattr(dst_mod, 'dst_to_src_mapping_tnsrs') and dst_mod.dst_to_src_mapping_tnsrs:
                    src_key = src_mod.get_name_wi_id() if hasattr(src_mod, 'get_name_wi_id') else None
                    if src_key and src_key in dst_mod.dst_to_src_mapping_tnsrs:
                        mapping = dst_mod.dst_to_src_mapping_tnsrs[src_key]
                        builtins.print(f"      └─ Neuron mapping: {len(mapping)} connections")
        
        builtins.print("\n" + "="*80 + "\n")


    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Performs a forward pass through the wrapped model, optionally updating its age.

        This method first calls `self.maybe_update_age(x)` to potentially update
        internal state related to the model's "age" or tracking, and then
        executes the forward pass of the underlying `self.model` with the
        provided input tensor `x`.

        Args:
            x (th.Tensor): The input tensor to the model.

        Returns:
            th.Tensor: The output tensor from the model's forward pass.
        """
        self.maybe_update_age(x)
        out = self.model(x)

        return out

    def state_dict(self):
        """
        Returns the state dictionary of the wrapped model.

        This method provides a way to access the `state_dict` of the underlying
        `self.model`, which is essential for saving and loading model parameters
        (weights, biases, etc.). It acts as a proxy to the original PyTorch
        model's `state_dict` method.

        Returns:
            dict: A dictionary containing a whole state of the module.
        """
        return super().state_dict()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Loads the model's parameters and buffers from a state dictionary.

        This method is a wrapper around the underlying PyTorch model's
        `load_state_dict` method. It allows for loading a pre-trained model's
        state, including weights and biases, into the current model instance.

        Args:
            state_dict (dict): A dictionary containing parameters and
                persistent buffers.
            strict (bool, optional): Whether to strictly enforce that the keys
                in `state_dict` match the keys returned by this module's
                `state_dict()` method. Defaults to True.
            assign (bool, optional): Whether to copy the data from `state_dict`
                into the module's parameters and buffers. Defaults to False.

        Returns:
            NamedTuple: A named tuple with `missing_keys` and `unexpected_keys` fields,
            detailing any discrepancies between the provided `state_dict` and the
            model's own state dictionary.
        """
        return super().load_state_dict(state_dict, strict, assign)

    def __repr__(self):
        """
        Overrides the behavior of print(model).
        It mimics the standard PyTorch format but includes a custom module ID.
        """
        string = f"{self.__class__.__name__}(\n"

        # Iterate over all named child modules
        for name, module in self.model.named_children():
            # Standard PyTorch module representation
            module_repr = repr(module)

            # --- Custom Logic to Inject ID ---
            # Check if the module has the get_module_id method
            # (i.e., if it's one of your custom layers)
            if hasattr(module, 'get_module_id'):
                try:
                    module_id = module.get_module_id()
                    # Inject the ID into the module's representation string
                    module_repr = f"ID={module_id} | {module_repr}"
                except Exception:
                    # Fallback if get_module_id fails
                    pass
            elif isinstance(module, th.nn.modules.container.Sequential):
                seq_string = "\n"
                for seq_name, seq_module in module.named_children():
                    seq_module_repr = repr(seq_module)
                    if hasattr(seq_module, 'get_module_id'):
                        try:
                            seq_module_id = seq_module.get_module_id()
                            # Inject the ID into the module's representation string
                            seq_module_repr = f"ID={seq_module_id} | {seq_module_repr}"
                        except Exception:
                            # Fallback if get_module_id fails
                            pass
                    seq_lines = seq_module_repr.split('\n')
                    # The first line is formatted with the name, the rest are indented
                    seq_string += f"  ({seq_name}): {seq_lines[0]}\n"
                    for seq_line in seq_lines[1:]:
                        seq_string += f"  {seq_line}\n"
                module_repr = f"{seq_string}"
            else:
                module_repr = f"ID=None | {module_repr}"

            # -----------------------------------
            # Indent and append the module's details
            # We use string manipulation to correctly format and indent nested
            # modules
            lines = module_repr.split('\n')

            # The first line is formatted with the name, the rest are indented
            string += f"  ({name}): {lines[0]}\n"
            for line in lines[1:]:
                string += f"  {line}\n"

        string += ")"
        return string


if __name__ == "__main__":
    from weightslab.tests.torch_models import \
        UNet as Model

    # Setup prints
    setup_logging('DEBUG')
    print('Hello World')

    # 0. Get the model
    model = Model()
    print(model)

    # 2. Create a dummy input and transform it
    dummy_input = th.randn(model.input_shape)

    # 3. Test the model inference
    model(dummy_input)

    # --- Example ---
    model = WatcherEditor(model, dummy_input=dummy_input, print_graph=False)
    print(f'Inference results {model(dummy_input)}')  # infer
    print(model)

    # Model Operations
    # # Test: add neurons
    print("--- Test: Op Neurons ---")
    model_op_neurons(model, op=1, rand=False)
    model(dummy_input)  # Inference test
    model_op_neurons(model, op=2, rand=False)
    model(dummy_input)  # Inference test
    model_op_neurons(model, op=3, rand=False)
    model(dummy_input)  # Inference test
    model_op_neurons(model, op=4, rand=False)
    model(dummy_input)  # Inference test
    print(f'Inference test of the modified model is:\n{model(dummy_input)}')
