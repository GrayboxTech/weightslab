import torch as th

from torch.fx.passes.shape_prop import ShapeProp
from torch.fx import symbolic_trace

from weightslab.components.tracking import TrackingMode
from weightslab.models.model_with_ops import NetworkWithOps
from weightslab.layers.neuron_ops import NeuronWiseOperations

from weightslab.utils.plot_graph import plot_fx_graph_with_details
from weightslab.utils.logs import print, setup_logging
from weightslab.models.monkey_patcher import monkey_patch
from weightslab.utils.tools import generate_graph_dependencies, \
    model_add_neurons


class WatcherEditor(NetworkWithOps):
    def __init__(
            self,
            model,
            dummy_input=None,
            device='cpu',
            print_graph=False,
            print_graph_filename=None):

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
        Used to set up the resource or state.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Executed when exiting the 'with' block (whether by success or error).
        Used to clean up or reset the resource.
        """
        self.visited_nodes = set()  # Reset NetworkWithOps nodes visited
        if exc_type is not None:
            print(
                f"[{self.__class__.__name__}]: An exception occurred: \
                    {exc_type.__name__}")
            return False
        return False

    def monkey_patch_model(self):
        """
            Use the monkey strategy to supplement
            each torch.nn.modules with LayerWiseOperations.
        """
        self.model.apply(monkey_patch)

    def shape_propagation(self):
        """
            Generate in/out meta info. in graph.
            Use to generate deps. and for the graph viz.
        """
        ShapeProp(self.traced_model).propagate(self.dummy_input)

    def children(self):
        """
            Generate a list of every model children, i.e., layers
        """
        # Return every model layers with ops
        childs = list(self.model.children())
        return childs

    def generate_graph_vizu(self):
        """
            Define the original dependencies between the layers.
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
        """
            Generate the graph dependencies and register them.
        """

        # Generate the dependencies
        self.dependencies_with_ops = generate_graph_dependencies(
            self.model,
            self.traced_model
        )

        # Register the layers dependencies
        self.register_dependencies(self.dependencies_with_ops)

    def forward(self, x):
        """
            Forward method that will first maybe update the model age.
            Then model forwarding.
        """
        self.maybe_update_age(x)
        out = self.model(x)

        return out

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
    from weightslab.tests.test_utils import FashionCNNSequential

    # Setup prints
    setup_logging('DEBUG')
    print('Hello World')

    # 0. Get the model
    model = FashionCNNSequential()

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
    model_add_neurons(model)
    model(dummy_input)  # Inference test
    print(f'Inference test of the modified model is:\n{model(dummy_input)}')
    print('#'+'-'*50)
