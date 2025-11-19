import functools
import types
import torch as th

from torch.fx.passes.shape_prop import ShapeProp
from torch.fx import symbolic_trace

from weightslab.components.tracking import TrackingMode
from weightslab.models.model_with_ops import NetworkWithOps
from weightslab.modules.neuron_ops import NeuronWiseOperations

from weightslab.utils.plot_graph import plot_fx_graph_with_details
from weightslab.models.monkey_patcher import monkey_patch_modules
from weightslab.utils.logs import print, setup_logging
from weightslab.utils.tools import model_op_neurons
from weightslab.utils.computational_graph import \
    generate_graph_dependencies
from weightslab.components.global_monitoring import pause_controller


class ModelInterface(NetworkWithOps):
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
        super(ModelInterface, self).__init__()

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
        self.traced_model = symbolic_trace(model)
        self.traced_model.name = "N.A."
        self.pause_ctrl = pause_controller

        # Propagate the shape over the graph
        self.shape_propagation()

        # Generate the graph vizualisation
        self.generate_graph_vizu()

        # Patch the torch model with WeightsLab features
        self.monkey_patching()

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

    def monkey_patching(self):
        """
        Applies monkey patching to the model's modules.

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

        # Monkey patch every nn.Module of the model
        self.model.apply(monkey_patch_modules)

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

        self.pause_ctrl.wait_if_paused()  # Wait until resume
        
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
    model = ModelInterface(model, dummy_input=dummy_input, print_graph=False)
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
