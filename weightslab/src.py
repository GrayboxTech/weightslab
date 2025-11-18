""" The Experiment class is the main class of the graybox package.
It is used to train and evaluate models. """

import functools
import torch as th

from tqdm import trange
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Callable, List
from collections import namedtuple
from threading import Lock, RLock

from weightslab.components.checkpoint import CheckpointManager
from weightslab.data.data_samples_with_ops import \
    DataSampleTrackingWrapper
from weightslab.weightslab.backend.model_interface import ModelInterface
from weightslab.utils.logs import print
from weightslab.weightslab.components.global_monitoring import GuardContext


class WeightsLab:
    """
        Experiment class is the main class of the graybox package.
        It is used to train and evaluate models. Every change to the models, or
        the experiment parameters are made through this class
    """

    # Global hyperparameters dictionary
    HyperParam = namedtuple(
        "HyperParam",
        [
            "data_type",
            "value",
            "default_value",
            "fct2call",
            "related_functions"
        ]
    )
    HyperParam.__new__.__defaults__ = (
        None,
        None,
        (),
        ()
    )
    GLOBAL_HYPER_PARAMETERS: Dict = {}

    def __init__(
            self,
            config: Dict[str, Any] = None
    ):

        # Read hyperparameters from the config
        self.set_global_hyperparam(
            config_data=config
        )

        self.training_steps_to_do = 2048
        self.eval_full_to_train_steps_ratio = 64
        self.experiment_dump_to_train_steps_ratio = 1024
        self.occured_train_steps = 0
        self.occured_eval__steps = 0
        self.train_loop_callbacks = []
        self.train_loop_clbk_freq = 50
        self.train_loop_clbk_call = True
        self.learning_rate = 1e-2
        self.root_log_dir = Path(self.get_global_hyperparam('root_log_dir'))
        if not self.root_log_dir.exists():
            self.root_log_dir.mkdir(parents=True, exist_ok=True)

        # Init Logger and CheckpointManager
        self.logger = SummaryWriter(self.root_log_dir)
        self.chkpt_manager = CheckpointManager(self.root_log_dir)
        self.chkpt_manager.load(
            self.chkpt_manager.get_latest_experiment(), self
        ) if not self.get_global_hyperparam('skip_loading') else None

        # Thread safeguards
        self.lock = Lock()
        self.architecture_guard = RLock()
        self.training_guard = GuardContext(self, for_training=True)
        self.testing_guard = GuardContext(self, for_training=False)

    def logger_add_scalars(self, name: str, data: Dict, model_age: int):
        self.logger.add_scalars(
            name,
            data,
            global_step=model_age
        )

    def get_model_age(self):
        return self.model.get_age()

    def update_data_statistics(
        self,
        model_age: int,
        batch_ids: th.Tensor,
        losses_batch: th.Tensor,
        preds: th.Tensor,
        is_training: bool = True
    ):
        with self.lock:
            # Get batch data
            pred_np = preds.detach().cpu().numpy()
            batch_ids_np = batch_ids.detach().cpu().numpy()
            if not isinstance(losses_batch, dict):
                per_sample_loss_np = losses_batch.detach().cpu().numpy()
            else:
                for k in losses_batch:
                    losses_batch[k] = losses_batch[k].detach().cpu().numpy()
                per_sample_loss_np = losses_batch

            # Update batch sample stats
            self.train_dataset_loader.dataset.update_batch_sample_stats(
                model_age,
                batch_ids_np,
                per_sample_loss_np,
                pred_np
            )
            self.train_dataset_loader.dataset.update_sample_stats_ex_batch(
                batch_ids_np,
                {
                    "loss/combined": per_sample_loss_np,
                    "pred": pred_np
                }
            )

    def set_global_hyperparam(self, config_data: Dict) -> None:
        self.GLOBAL_HYPER_PARAMETERS.update(config_data)

    def get_global_hyperparam(
        self,
        name: str = None,
        p: Any = None,
        default_value: Any = None
    ) -> HyperParam:
        """
            Get the global hyperparameter for the given name, supporting nested 
            key paths (e.g., 'optimizer/Adam/lr').

            If 'p' is provided, it takes the highest precedence.
        """
        # 1. Highest Precedence: If 'p' is provided, return it immediately.
        if p is not None:
            return p

        # 2. Path Traversal: If a 'name' (path) is provided, attempt to traverse the structure.
        if name:
            # Clean the path and split into components (e.g., '/opt/Adam/lr' -> ['opt', 'Adam', 'lr'])
            path_components = name.strip('/').split('/')
            current_level = self.GLOBAL_HYPER_PARAMETERS

            try:
                # Iterate through the components to traverse the nested dictionary
                for key in path_components:
                    # Check if the current level is a dictionary and contains the key
                    if isinstance(current_level, dict) and key in current_level:
                        current_level = current_level[key]
                    else:
                        # Path segment not found or structure is not a dictionary
                        return default_value

                # If loop completes, 'current_level' holds the final value
                return current_level

            except Exception:
                # Catch any unexpected errors during traversal (e.g., if a non-dict was indexed)
                return default_value

        # 3. Fallback: If no name was provided, or if retrieval logic above failed, 
        # return the default value.
        return default_value

    # ========================================================================
    # ========================================================================
    # Main functions
    # Basic idea here: generate wl_exp (instance of Experiment, or Experiment directly),
    # and set the objects here (model, optimizer, data, etc.)
    # TODO (GP): Create abstract classes for Experiment with get_model, get_data, or whatever object maybe ?
    # Or maybe not.
    def watch_or_edit(self, obj: Callable, obj_name: str = None, flag: str = None, **kwargs) -> None:
        """
        Watch or edit the given object.

        Args:
            obj (Callable): The object to watch or edit.
            flag (str): The flag specifying the type of object to watch or
            edit.
            kwargs (Any): Additional keyword arguments to pass.
        """

        # Sanity check
        if not hasattr(obj, '__name__'):
            if obj_name is None:
                obj.__name__ = 'anonymous'
                print(
                    "Warning: Watching or editing anonymous object '" +
                    f"{obj.__name__}'."
                )
                print(
                    "Please add a 'name' attribute to the object."
                )
            else:
                obj.__name__ = obj_name

        # Related functions
        if flag == 'model' or 'model' in obj.__name__.lower():
            return self._watch_or_edit_model(obj, **kwargs)
        elif flag == 'data' or 'data' in obj.__name__.lower():
            return self._watch_or_edit_dataset(obj, **kwargs)
        elif flag == 'optimizer' or 'optimizer' in obj.__name__.lower():
            return self._watch_or_edit_optimizer(obj, **kwargs)

    def _watch_or_edit_model(self, model: Callable, **_) -> None:
        """
            Set up the model for tracking and validation

            Args:
                model (Callable): The torch model to watch or edit.
        """

        # Interface the torch model with weightslab's ModelInterface
        self._interface_model(model)
        # Register function to update optimizer when model
        # architecture changes
        self.model.register_hook_fn_for_architecture_change(
            lambda model: self._update_optimizer(model)
        )

        return self.model

    def _watch_or_edit_dataset(self, dataset: Callable, **kwargs) -> None:
        """
            Set up the dataset for tracking and validation

            Args:
                dataset (Callable): The dataset to watch or edit.
                kwargs (Any): Additional keyword arguments to pass to the
                DataLoader.
        """

        # TODO (GP): iter(loader) not working with n_workers > 0
        n_workers = self.get_global_hyperparam(
                f'/data/{dataset.__name__}/num_workers',
                kwargs.get('num_workers'),
                0
            )
        if n_workers > 0:
            raise ValueError("Invalid number of workers. Please use a positive integer.")

        # Data Loader
        tracked_dataset = DataSampleTrackingWrapper(
            dataset
        )
        tracked_dataset._map_updates_hook_fns.append(
            (self.reset_data_iterators, kwargs)
        )
        loader = th.utils.data.DataLoader(
            tracked_dataset,
            batch_size=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/batch_size',
                kwargs.get('batch_size'),
                1
            ),
            shuffle=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/train_shuffle',
                kwargs.get('train_shuffle'),
                True
            ),
            num_workers=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/num_workers',
                kwargs.get('num_workers'),
                0
            ),
            drop_last=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/drop_last',
                kwargs.get('drop_last'),
                True
            ),
            pin_memory=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/pin_memory',
                kwargs.get('pin_memory'),
                False
            ),
            **self.get_global_hyperparam(
                f'/data/{dataset.__name__}/kwargs',
                kwargs.get('kwargs'),
                {}
            ),
        )

        # Set instance variables
        # # set {train|test}_dataset_loader
        setattr(self, f'{dataset.__name__}_tracker', loader)
        # # set {train|test}_dataset_loader
        setattr(self, f'{dataset.__name__}_loader', loader)
        # # set {train|test}_dataset_iterator
        setattr(self, f'{dataset.__name__}_iterator', iter(loader))

        return getattr(self, f'{dataset.__name__}_iterator')

    def _watch_or_edit_optimizer(self, obj: Callable, **kwargs) -> None:
        """
            Set up the model optimizer.

            Args:
                dataset (Callable): The Optimizer class to watch or edit.
                kwargs (Any): Additional keyword arguments to pass to the
                DataLoader.
        """
        self.optimizer_class = obj
        self.optimizer = obj(
            self.model.parameters(),
            **(self.get_global_hyperparam(
                f'/optimizer/{obj.__name__}/',
                None
            ) or kwargs)
        )
        return self.optimizer

    def watch(self, obj: Callable, flag: str = "default", log=False, **kwargs) -> None:
        """
        Monkey patches the 'forward' method of the input object (obj) to inject
        logging action after the original calculation.

        Args:
            obj: The callable object (e.g., nn.CrossEntropyLoss) to patch.
            flag: The name to use for logging (e.g., 'loss/bin_loss').

        Returns:
            The patched object.
        """
        if 'loss' in flag.lower():
            return self._watch_loss(obj, flag, log, **kwargs)
        elif 'metric' in flag.lower():
            return self._watch_metric(obj, flag, log, **kwargs)

    def _watch_loss(self, obj: Callable, flag: str, log: bool, **kwargs): 
        # Ensure the object has a forward method to patch
        if not hasattr(obj, 'forward') \
                or not callable(getattr(obj, 'forward')):
            raise ValueError(
                f"Object {obj} does not have a callable 'forward' method \
                to patch."
            )

        # Capture the original forward method.
        original_forward = obj.forward

        # Define the new wrapper function (closure).
        # It captures 'self' (ExperimentLogger instance), 'original_forward',
        # and 'flag'.
        @functools.wraps(original_forward)
        def new_forward_func(*args, **kwargs):
            _flag = None
            if 'flag' in kwargs:
                _flag = kwargs.pop('flag', None)

            # --- EXECUTE ORIGINAL LOGIC ---
            # Call the original method with all passed arguments.
            # This returns the raw loss tensor needed for backpropagation.
            losses_value = original_forward(*args, **kwargs)

            # --- INJECT CUSTOM LOGGING ACTION ---
            try:
                tag = flag if _flag is None else _flag

                # 1. Get logging context
                model_age = self.get_model_age()

                # 2. Extract loss value and format for logger
                # Detach, move to CPU, convert to numpy scalar (item())
                flat_losses_value = losses_value.flatten().mean()

                # Log results
                self.logger.add_scalars(
                    tag,
                    {tag: flat_losses_value},
                    global_step=model_age
                ) if log else None

            except Exception as e:
                print(f"Logging Warning: Failed to log value for flag '{tag}': {e}")

            # --- RETURN ORIGINAL RESULT ---
            # This MUST return the raw tensor for the optimizer's backward pass.
            return losses_value

        # Apply the Monkey Patch
        print(
            "Successfully patched 'forward' method for object: " +
            f"{obj.__class__.__name__} with flag '{flag}'",
            level='DEBUG'
        )
        obj.forward = new_forward_func

        return obj

    def _watch_metric(self, obj: Callable, flag: str, log: bool, **kwargs): 
        # Ensure the object has a compute method to patch
        if not hasattr(obj, 'compute') \
                or not callable(getattr(obj, 'compute')):
            raise ValueError(
                f"Object {obj} does not have a callable 'compute' method \
                to patch."
            )

        # Capture the original compute method.
        original_compute = obj.compute

        # Define the new wrapper function (closure).
        # It captures 'self' (ExperimentLogger instance), 'original_compute',
        # and 'flag'.
        @functools.wraps(original_compute)
        def new_compute_func(*args, **kwargs):
            _flag = None
            if 'flag' in kwargs:
                _flag = kwargs.pop('flag', None)

            # --- EXECUTE ORIGINAL LOGIC ---
            # Call the original method with all passed arguments.
            # This returns the raw loss tensor needed for backpropagation.
            value = original_compute(*args, **kwargs)

            # --- INJECT CUSTOM LOGGING ACTION ---
            try:
                tag = flag if _flag is None else _flag

                # 1. Get logging context
                model_age = self.get_model_age()

                # 2. Extract loss value and format for logger
                # Detach, move to CPU, convert to numpy scalar (item())
                metric_value = value.detach().cpu().numpy().item()

                # Log results
                self.logger.add_scalars(
                    tag,
                    {tag: metric_value},
                    global_step=model_age
                ) if log else None

            except Exception as e:
                print(f"Logging Warning: Failed to log value for flag '{tag}': {e}")

            # --- RETURN ORIGINAL RESULT ---
            # This MUST return the raw tensor for the optimizer's backward pass.
            return value

        # Apply the Monkey Patch
        print(
            "Successfully patched 'compute' method for object: " +
            f"{obj.__class__.__name__} with flag '{flag}'",
            level='DEBUG'
        )
        obj.compute = new_compute_func

        return obj

    # =========================================
    # Training & Inference functions
    # Model state


    # ========================================================================
    # ========================================================================
    # Private functions
    def __repr__(self):
        with self.lock:
            return f"Experiment[{id(self)}, {self.name}] " + \
                f"is_train: {self.is_training} " + \
                f"steps: {self.training_steps_to_do}"

    def _update_optimizer(self, model):
        self.optimizer = self.optimizer_class(
            model.parameters(),
            lr=self.learning_rate
        )

    def _pick_legacy_dense_pred(self, preds, x):
        HxW = None
        if isinstance(x, th.Tensor) and x.ndim >= 4:
            HxW = (int(x.shape[-2]), int(x.shape[-1]))

        best, best_score = None, -1.0
        for p in preds.values():
            if not isinstance(p, th.Tensor):
                continue
            if not (p.ndim >= 3 or (p.ndim == 2 and p.shape[1] >= 64)):
                continue
            if p.ndim == 3:  # [N, H, W]
                H, W = int(p.shape[-2]), int(p.shape[-1])
            elif p.ndim >= 4:
                H, W = int(p.shape[-2]), int(p.shape[-1])
            else:
                continue
            score = float(H * W)
            if HxW and (H, W) == HxW:
                score += 1e9
            if score > best_score:
                best, best_score = p, score

        # fallback: first pred if nothing dense
        return best if best is not None else next(iter(preds.values()))

    def _interface_model(self, model):
        self.model = ModelInterface(
            model,
            dummy_input=th.randn(model.input_shape),
            device=self.get_global_hyperparam('device')
        )

    def register_train_loop_callback(self, callback):
        """Add callback that will be called every train_loop_clbk_freq steps
        during the training loop

        Args:
            callback (function): a function that will be called in training
        """
        self.train_loop_callbacks.append(callback)

    def unregister_train_loop_callback(self, callback):
        """Remove callback from the list of callbacks that are called during
        training.

        Args:
            callback (function): the function handle to be removed
        """
        self.train_loop_callbacks.remove(callback)

    def toggle_train_loop_callback_calls(self):
        """Toggle the calling of the callbacks during training loop
            This either enables or disables the callbacks.
        """
        self.train_loop_clbk_call = not self.train_loop_clbk_call

    def toggle_is_training(self):
        """Toggle the calling of the callbacks during training loop
            This either enables or disables the callbacks.
        """
        if not self.pause_controller.is_paused():
            self.pause_controller.pause()
        else:
            self.pause_controller.resume()

    def performed_train_steps(self):
        """Return the number of training steps that have been performed.

        Returns:
            int: the number of training steps that have been performed
        """
        return self.occured_train_steps

    def performed_eval_steps(self):
        """Return the number of evaluation steps that have been performed.

        Returns:
            int: the number of evaluation steps that have been performed
        """
        return self.occured_eval__steps

    def dump(self):
        """Dump the experiment into a checkpoint. Marks the checkpoint on the
        plots."""
        self.chkpt_manager.dump(self)
        graph_names = self.logger.get_graph_names()
        self.logger.add_annotations(
            graph_names, self.name, "checkpoint", self.model.get_age(),
            {
                "checkpoint_id": self.chkpt_manager.get_latest_experiment()
            }
        )

    def load(self, checkpoint_id: int):
        """Loads the given checkpoint with a given id.

        Args:
            checkpoint_id (int): the checkpoint id to be loaded
        """
        self.optimizer.zero_grad()
        self.chkpt_manager.load(checkpoint_id, self)

    def print_checkpoints_tree(self):
        """Display the checkpoints tree."""
        print(self.chkpt_manager.id_to_path)

    # ========================================================================
    # ========================================================================
    # Data functions
    def reset_data_iterators(self, **kwargs: dict) -> None:
        """
            Reset the data iterators. This is necessary when anything related
            to datasets or dataloaders changes.
        """

        # Train
        dataset = self.train_dataset_tracker
        self.train_dataset_loader = th.utils.data.DataLoader(
            dataset,
            batch_size=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/batch_size',
                kwargs.get('batch_size'),
                1
            ),
            shuffle=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/train_shuffle',
                kwargs.get('train_shuffle'),
                True
            ),
            num_workers=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/num_workers',
                kwargs.get('num_workers'),
                8
            ),
            drop_last=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/drop_last',
                kwargs.get('drop_last'),
                True
            ),
            pin_memory=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/pin_memory',
                kwargs.get('pin_memory'),
                False
            ),
            **self.get_global_hyperparam(
                f'/data/{dataset.__name__}/kwargs',
                kwargs.get('kwargs'),
                {}
            ),
        )
        self.train_dataset_iterator = iter(self.train_dataset_loader)

        # Eval
        dataset = self.test_dataset_tracker
        self.test_dataset_loader = th.utils.data.DataLoader(
            dataset,
            batch_size=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/batch_size',
                kwargs.get('batch_size'),
                1
            ),
            shuffle=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/train_shuffle',
                kwargs.get('train_shuffle'),
                True
            ),
            num_workers=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/num_workers',
                kwargs.get('num_workers'),
                8
            ),
            drop_last=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/drop_last',
                kwargs.get('drop_last'),
                True
            ),
            pin_memory=self.get_global_hyperparam(
                f'/data/{dataset.__name__}/pin_memory',
                kwargs.get('pin_memory'),
                False
            ),
            **self.get_global_hyperparam(
                f'/data/{dataset.__name__}/kwargs',
                kwargs.get('kwargs'),
                {}
            ),
        )
        self.test_dataset_iterator = iter(self.test_dataset_loader)

    def get_train_records(self):
        """"Get all the train samples are records."""
        with self.lock:
            return self.train_dataset_loader.dataset.as_records()

    def get_eval_records(self):
        """"Get all the train samples are records."""
        with self.lock:
            return self.test_dataset_loader.dataset.as_records()

    # ========================================================================
    # ========================================================================
    # Hyperparameters functions
    def set_parameter(
            self,
            parameter_name,
            parameter_value,
            fct2call: Callable = None,
            related_functions: List[Callable] = []
    ):
        """Set a parameter value in the global hyperparameters.

        Args:
            parameter_name (str): the name of the parameter
            parameter_value (any): the new value of the parameter
            fct2call (Callable): a function that will be called when the
            parameter is set.
            related_functions (List[Callable]): a list of functions that will be
            called when the parameter is set.
        """
        with self.lock:
            self.global_hyper_parameters[parameter_name] = HyperParam(
                data_type=type(parameter_value),
                value=parameter_value,
                default_value=parameter_value,
                fct2call=fct2call,
                related_functions=related_functions
            )

        # 2. Dynamic Method Generation
        def getter(instance: 'Experiment'):
            """Dynamically generated getter for {parameter_name}."""
            with instance.lock:
                return instance.global_hyper_parameters[parameter_name].value

        # Define the Setter function template
        def setter(instance: 'Experiment', new_value: Any):
            """Dynamically generated setter for {parameter_name}."""
            with instance.lock:
                # Update the value attribute of the HyperParam object
                instance.global_hyper_parameters[parameter_name].value = new_value
                # Update related object parameters with the new value, e.g.,
                # optimizer and learning_rate.
                setattr(instance.global_hyper_parameters[parameter_name].related_functions, parameter_name, new_value)
                instance.global_hyper_parameters[parameter_name].fct2call()

        # Assign the functions as new methods to the instance using setattr()
        setattr(self, f"get_{parameter_name}", getter)
        setattr(self, f"set_{parameter_name}", setter)

    # ========================================================================
    # ========================================================================
    # Trainer worker function
    def apply_architecture_op(self, op_type, **kwargs):
        # Operate
        with self.architecture_guard, self.model as model:
            model.operate(
                layer_id=kwargs['layer_id'],
                neuron_indices=kwargs['neuron_indices'],
                neuron_operation=op_type
            )
