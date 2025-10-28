import torch as th

from torch.fx.passes.shape_prop import ShapeProp
# from weightslab.utils.shape_prop import ShapeProp
from torch.fx import symbolic_trace

from weightslab.components.tracking import TrackingMode
from weightslab.models.model_with_ops import NetworkWithOps
from weightslab.layers.neuron_ops import NeuronWiseOperations

from weightslab.utils.plot_graph import plot_fx_graph_with_details
from weightslab.utils.logs import print, setup_logging
from weightslab.models.monkey_patcher import monkey_patch
from weightslab.utils.tools import generate_graph_dependencies, \
    model_op_neurons


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
        self.dummy_input = dummy_input.to(device)
        self.print_graph = print_graph
        self.print_graph_filename = print_graph_filename
        self.traced_model = symbolic_trace(model)
        self.traced_model.name = "N.A."

        # Propagate the shape over the graph
        self.shape_propagation()

        # Generate the graph vizualisation
        self.generate_graph_vizu()

        # Patch the torch model with WeightsLab features
        self.monkey_patch_model()

        # Generate the graph dependencies
        self.define_deps()

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

    def __exit__(self, exc_type, **_):
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
                    {exc_type.__name__}")
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
                self.traced_model
            )
            plot_fx_graph_with_details(
                self.traced_model,
                custom_dependencies=or_dependencies,

                filename=self.print_graph_filename
            )

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
                self.traced_model
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

        # Generate the dependencies
        self.dependencies_with_ops = generate_graph_dependencies(
            self.model,
            self.traced_model
        )

        # Register the layers dependencies
        self.register_dependencies(self.dependencies_with_ops)

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

    def __repr__(self):
        """
        Generates a string representation of the WatcherEditor instance.

        This method overrides the default `__repr__` behavior to provide a
        human-readable string that mimics the standard PyTorch `nn.Module`
        representation. It enhances this representation by injecting a custom
        module ID (if available via `get_module_id`) for each submodule,
        including those within `Sequential` containers. This implementation
        handles nested containers recursively.
        """
        def _get_module_repr(module, prefix=""):
            """Recursively builds the string representation for a module."""
            # Base case: not a container, or an empty container
            if not list(module.children()):
                module_repr = repr(module)
                if hasattr(module, 'get_module_id'):
                    try:
                        module_id = module.get_module_id()
                        return f"ID={module_id} | {module_repr}"
                    except Exception:
                        return f"ID=err | {module_repr}"
                else:
                    # For standard layers without get_module_id
                    return f"ID=None | {module_repr}"

            # Recursive step for containers
            child_lines = []
            for name, child_module in module.named_children():
                child_repr = _get_module_repr(child_module, prefix + "  ")
                child_lines.append(f"{prefix}  ({name}): {child_repr}")

            # Get the class name of the container
            container_name = module.__class__.__name__
            # Join the child representations
            children_str = '\n'.join(child_lines)
            return f"{container_name}(\n{children_str}\n{prefix})"

        # Start with the model's class name
        string = f"{self.model.__class__.__name__}(\n"

        # Iterate over all top-level child modules
        for name, module in self.model.named_children():
            module_repr = _get_module_repr(module, "  ")
            string += f"  ({name}): {module_repr}\n"

        string += ")"
        return string


if __name__ == "__main__":
    from weightslab.tests.test_utils import FCNResNet50 as Model

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
    print("--- Test: Add Neurons ---")
    model_op_neurons(model)
    model(dummy_input)  # Inference test
    print(f'Inference test of the modified model is:\n{model(dummy_input)}')
