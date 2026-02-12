import logging
import os
import torch as th
import weightslab as wl

from torch.fx.passes.shape_prop import ShapeProp
from torch.fx import symbolic_trace

from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.components.tracking import TrackingMode
from weightslab.models.model_with_ops import NetworkWithOps
from weightslab.modules.neuron_ops import NeuronWiseOperations

from weightslab.utils.plot_graph import plot_fx_graph_with_details
from weightslab.models.monkey_patcher import monkey_patch_modules
from weightslab.utils.computational_graph import \
    generate_graph_dependencies_from_torchfx, \
    generate_layer_dependencies_from_onnx, \
    generate_index_maps
from weightslab.components.global_monitoring import guard_training_context, guard_testing_context
from weightslab.backend.ledgers import get_optimizer, get_optimizers, register_model
from weightslab.backend import ledgers
from weightslab.utils.tools import restore_rng_state


# Global logger
logger = logging.getLogger(__name__)


class ModelInterface(NetworkWithOps):
    def __init__(
            self,
            model: th.nn.Module,
            dummy_input: th.Tensor | dict = None,
            device: str = 'cpu',
            print_graph: bool = False,
            print_graph_filename: str = None,
            name: str = None,
            register: bool = True,
            use_onnx: bool = False,
            compute_dependencies: bool = True,
            weak: bool = False,
            skip_previous_auto_load: bool = False,
            **_
    ):
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
            name (str, optional): The name to assign to this model interface.
                Defaults to None.
            register (bool, optional): If True, the model interface will be
                registered in the global ledger. Defaults to True.
            use_onnx (bool, optional): If True, ONNX export will be used for
                dependency extraction instead of torch.fx tracing. Defaults to False.
            compute_dependencies (bool, optional): If True, computes the graph
            weak (bool, optional): If True, registers the model with a weak
                reference in the ledger. Defaults to False.
            skip_previous_auto_load (bool, optional): If True, skips the automatic loading
                of previous checkpoints during initialization. Defaults to False.
            
        Returns:
            None: This method initializes the object and does not return any value.
        """
        super(ModelInterface, self).__init__()

        # Reinit IDS when instanciating a new torch model
        NeuronWiseOperations().reset_id()

        # Define variables
        # # Disable tracking for implementation
        self.tracking_mode = TrackingMode.DISABLED
        self.name = "Default Name" if name is None else name
        self.device = device
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.skip_previous_auto_load = skip_previous_auto_load

        # Generate dummy input if not provided and sanity check
        if compute_dependencies:
            # First ensure that the model has module input_shape
            if not hasattr(model, 'input_shape'):
                raise ValueError("Model object must have 'input_shape' attribute for proper registration with WeightsLab.")

            # Move dummy input to the correct device, or create a default one if not provided
            if dummy_input is not None:
                self.dummy_input = dummy_input.to(device)
            else:
                self.dummy_input = th.randn(model.input_shape).to(device)

        # Initialize checkpoint manager and attempt early auto-load before any model-dependent setup
        self._checkpoint_manager = None
        _checkpoint_auto_every_steps = 0
        _root_log_dir = None
        try:
            from weightslab.backend.ledgers import list_hyperparams, get_hyperparams
            names = list_hyperparams()
            chosen = None
            if 'main' in names:
                chosen = 'main'
            elif 'experiment' in names:
                chosen = 'experiment'
            elif len(names) > 1:
                chosen = names[-1]

            if chosen:
                hp = get_hyperparams(chosen)
                if hasattr(hp, 'get') and not isinstance(hp, dict):
                    try:
                        hp = hp.get()
                    except Exception:
                        hp = None
                if isinstance(hp, dict):
                    _root_log_dir = hp.get('root_log_dir') or hp.get('root-log-dir') or hp.get('root')
                    _checkpoint_auto_every_steps = hp.get('experiment_dump_to_train_steps_ratio') or hp.get('experiment-dump-to-train-steps-ratio') or 0
        except Exception:
            _root_log_dir = None
            _checkpoint_auto_every_steps = 0
        self._checkpoint_auto_every_steps = int(_checkpoint_auto_every_steps or 0)

        # Initialize CheckpointManager if we have a root dir (fallback to default root)
        root_log_dir = _root_log_dir or os.path.join('.', 'root_log_dir')
        try:
            # Check if a checkpoint manager is already registered in ledger
            try:
                existing_manager = ledgers.get_checkpoint_manager()
                if existing_manager != None and not isinstance(existing_manager, ledgers.Proxy):
                    self._checkpoint_manager = existing_manager
                    logger.info("Using checkpoint manager from ledger")
                else:
                    raise KeyError("No manager in ledger")
            except (KeyError, AttributeError):
                # Create new manager and register it
                self._checkpoint_manager = CheckpointManager(root_log_dir=root_log_dir)
                try:
                    ledgers.register_checkpoint_manager(self._checkpoint_manager)
                    logger.info("Registered new checkpoint manager in ledger")
                except Exception:
                    pass
        except Exception:
            self._checkpoint_manager = None

            # Early auto-load latest model architecture and weights if checkpoints exist
            if self._checkpoint_manager != None and not self.skip_previous_auto_load:
                try:
                    # Try to get the latest experiment hash
                    latest_hash = None
                    if hasattr(self._checkpoint_manager, 'current_exp_hash') and self._checkpoint_manager.current_exp_hash:
                        latest_hash = self._checkpoint_manager.current_exp_hash
                    elif hasattr(self._checkpoint_manager, 'manifest') and self._checkpoint_manager.manifest:
                        manifest = self._checkpoint_manager.manifest
                        latest_hash = getattr(manifest, 'latest_hash', None)

                    if latest_hash:
                        # Use checkpoint manager's load_checkpoint to get architecture and weights
                        checkpoint_data = self._checkpoint_manager.load_checkpoint(
                            exp_hash=latest_hash,
                            load_model=True,
                            load_weights=True,
                            load_config=False,
                            load_data=False,
                            force=True
                        )

                        # Apply loaded model if architecture was loaded
                        if checkpoint_data.get('model'):
                            self = checkpoint_data['model']
                            weights = checkpoint_data.get('weights')
                            checkpoint_rng_state = checkpoint_data.get('weights', {}).get('rng_state')

                            # Restore RNG state if available
                            restore_rng_state(checkpoint_rng_state)
                            logger.debug(f"Restored RNG state from checkpoint")

                        elif checkpoint_data.get('weights'):
                            # Only weights available, load into existing model
                            weights = checkpoint_data['weights']
                            if 'model_state_dict' in weights:
                                self.load_state_dict(weights['model_state_dict'], strict=True)
                                self.current_step = weights.get('step', -1)
                                logger.info(f"Auto-loaded model weights from checkpoint {latest_hash[:16]} (step {self.current_step})")

                        # As model architecture as has been loaded, and it's an instance of the ModelInterface,
                        # we can set its current step if available in weights
                        if isinstance(self.model, self.__class__):
                            self._registration(
                                weak=weak
                            )
                            return

                except Exception as e:
                    logger.debug(f"Could not auto-load model checkpoint: {e}")

        if compute_dependencies and not use_onnx:
            self.print_graph = print_graph
            self.print_graph_filename = print_graph_filename
            self.traced_model = symbolic_trace(self.model)
            self.traced_model.name = "N.A."
        self.guard_training_context = guard_training_context
        self.guard_testing_context = guard_testing_context

        # Init attributes from super object (i.e., self.model)
        self.init_attributes(self.model)

        # Compute dependencies and generate graph visualization if enabled
        if compute_dependencies: 
            if not use_onnx:
                # Only propagate shapes if we need them for visualization
                if self.print_graph:
                    self.shape_propagation()
                    # Clean up any leftover threads from shape propagation
                    # import gc
                    # gc.collect()

                # Generate the graph vizualisation
                self.generate_graph_vizu()

            # Generate the graph dependencies
            self.define_deps(use_onnx=use_onnx, dummy_input=self.dummy_input)

        # Clean - Optionally register wrapper in global ledger
        if register:
            self._registration(
                weak=weak
            )
        
        # Set the optimizer hook for model architecture changes if we
        # are computing dependencies (i.e., we have the graph info to
        # know when they happen)
        if compute_dependencies:
            if not use_onnx:
                del self.traced_model

            # Hook optimizer update on architecture change
            self.register_hook_fn_for_architecture_change(
                lambda model: self._update_optimizer(model)
            )

        # Set Model Training Guard
        self.guard_training_context.model = self
        self.guard_testing_context.model = self

    def _registration(self, weak: bool = False):
        register_model(self, weak=weak)

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads the state dictionary into the wrapped model.

        This method forwards the provided `state_dict` to the underlying
        model's `load_state_dict` method, allowing for the restoration
        of model parameters and buffers from a saved state.

        Args:
            state_dict (dict): A state dictionary containing model parameters
                and buffers to be loaded.
            strict (bool, optional): Whether to strictly enforce that the keys
                in `state_dict` match the keys returned by the model's
                `state_dict()` function. Defaults to True.

        Returns:
            None: This method does not return any value; it modifies the
            state of the wrapped model in-place.
        """
        super().load_state_dict(state_dict, strict=strict)

    def init_attributes(self, obj):
        """Expose attributes and methods from the wrapped `obj`.

        Implementation strategy (direct iteration):
        - Iterate over `vars(obj)` to obtain instance attributes and
          create class-level properties that forward to `obj.<attr>`.
        - Iterate over `vars(obj.__class__)` to find callables (methods)
          and bind the model's bound method to this wrapper instance so
          calling `mi.method()` invokes `mi.model.method()`.

        This avoids using `dir()` and directly inspects the object's
        own dictionaries. Existing attributes on `ModelInterface` are
        preserved and not overwritten.
        """
        # Existing names on the wrapper instance/class to avoid overwriting
        existing_instance_names = set(self.__dict__.keys())
        existing_class_names = set(getattr(self.__class__, '__dict__', {}).keys())

        # 1) Expose model instance attributes as properties on the wrapper class
        model_vars = getattr(obj, '__dict__', {})
        for name, value in model_vars.items():
            if name.startswith('_'):
                continue
            if name in existing_instance_names or name in existing_class_names:
                continue

            # Create a property on the ModelInterface class that forwards to
            # the underlying model attribute. Using a property keeps the
            # attribute live (reads reflect model changes).
            try:
                def _make_getter(n):
                    return lambda inst: getattr(inst.model, n)

                getter = _make_getter(name)
                prop = property(fget=getter)
                setattr(self.__class__, name, prop)
            except Exception:
                # Best-effort: skip if we cannot set the property
                continue

        # 2) Bind model class-level callables (methods) to this instance
        model_cls_vars = getattr(obj.__class__, '__dict__', {})
        for name, member in model_cls_vars.items():
            if name.startswith('_'):
                continue
            if name in existing_instance_names or name in existing_class_names:
                continue

            # Only consider callables defined on the class (functions/descriptors)
            if callable(member):
                try:
                    # getattr(obj, name) returns the bound method
                    bound = getattr(obj, name)
                    # Attach the bound method to the wrapper instance so that
                    # calling mi.name(...) calls model.name(...)
                    setattr(self, name, bound)
                except Exception:
                    # If we cannot bind, skip gracefully
                    continue

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
            logger.error(
                f"[{self.__class__.__name__}]: An exception occurred: \
                    {exc_type.__name__} with {exc_val} and {exc_tb}.")
            return False
        return False

    def _update_optimizer(self, model):
        for opt_name in get_optimizers():
            # Overwrite the optimizer with the same class and lr, updated
            opt = get_optimizer(opt_name)
            lr = opt.get_lr()[0]
            optimizer_class = type(opt.optimizer)
            _optimizer = optimizer_class(
                model.parameters(),
                lr=lr
            )

            wl.watch_or_edit(_optimizer, flag='optimizer', name=opt_name)

    def _maybe_auto_dump(self):
        # Called from base class hook after seen_samples updates.
        # Auto-dump: save model weights only (and architecture if changed).
        try:
            if not self.is_training() or self._checkpoint_manager == None or self._checkpoint_auto_every_steps <= 0:
                return
            batched_age = int(self.get_batched_age())
            if batched_age > 0 and (batched_age % self._checkpoint_auto_every_steps) == 0:
                try:
                    # Update hash for current experiment state (marks changes as pending, doesn't dump)
                    _, _, changed_components = self._checkpoint_manager.update_experiment_hash()
                    # If model architecture changed, save it
                    if 'model' in changed_components:
                        try:
                            self._checkpoint_manager.save_model_architecture(self.model)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    # Save model weights checkpoint (no pending dump here)
                    self._checkpoint_manager.save_model_checkpoint(
                        model=self.model,
                        save_optimizer=True,
                        step=batched_age,
                        force_dump_pending=False,
                        update_manifest=False
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def eval(self):
        try:
            return super().eval()
        except (RuntimeError, Exception):
            logger.warning(
                f"[{self.__class__.__name__}]: Caught RuntimeError during eval(): {Exception}. \
                This may be due to certain layers not supporting eval mode. Continuing without eval()."
            )
            return self

    def train(self):
        try:
            return super().train()
        except (RuntimeError, Exception):
            logger.warning(
                f"[{self.__class__.__name__}]: Caught RuntimeError during train(): {Exception}. \
                This may be due to certain layers not supporting train mode. Continuing without train()."
            )
            return self

    def is_training(self) -> bool:
        """
        Checks if the model is currently in training mode.

        This method returns a boolean indicating whether the wrapped model
        is set to training mode (`True`) or evaluation mode (`False`).

        Returns:
            bool: `True` if the model is in training mode, `False` otherwise.
        """
        return self.training

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
            logger.info("--- Generated Graph Dependencies (FX Tracing) ---")
            or_dependencies = generate_graph_dependencies_from_torchfx(
                self.model,
                self.traced_model.graph
            )
            plot_fx_graph_with_details(
                self.traced_model,
                custom_dependencies=or_dependencies,
                filename=self.print_graph_filename
            )

    def define_deps(self, use_onnx: bool = False, dummy_input: th.Tensor = None):
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

        # Patch the torch model with WeightsLab features
        self.monkey_patching()

        # Generate the graph dependencies
        if not use_onnx:
            # self.shape_propagation()
            self.dependencies_with_ops = generate_graph_dependencies_from_torchfx(
                self.model,
                self.traced_model.graph
            )
        else:
            self.dependencies_with_ops = generate_layer_dependencies_from_onnx(
                self.model,
                dummy_input=dummy_input
            )

        # Map dependencies between layers and their operations
        self.mapped_dependencies_with_ops = generate_index_maps(
            self.dependencies_with_ops
        )

        # Register the dependencies
        self.register_dependencies(self.mapped_dependencies_with_ops)

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

        # Check device
        if x.device != self.device:
            x = x.to(self.device)

        self.maybe_update_age(x)
        out = self.model(x)

        return out

    def apply_architecture_op(self, op_type, layer_id, neuron_indices=None):
        """
            Applies an architecture operation to the model within a managed context.
        """
        with self as m:
            m.operate(layer_id=layer_id, op_type=op_type, neuron_indices=neuron_indices)

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
