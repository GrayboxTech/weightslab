import hashlib
import collections
import torch as th

from torch import nn
from enum import Enum
from typing import List, Set, Optional, Callable, Tuple

from weightslab.utils.logs import print
from weightslab.utils.tools import normalize_dicts, reversing_indices
from weightslab.components.tracking import Tracker
from weightslab.components.tracking import TrackingMode
from weightslab.utils.modules_dependencies import DepType
from weightslab.components.tracking import TriggersTracker
from weightslab.modules.neuron_ops import NeuronWiseOperations
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType
from weightslab.components.tracking import copy_forward_tracked_attrs


class LayerWiseOperations(NeuronWiseOperations):
    """
        Base class for the complementary operations needed in order to
        implement the neuron wise operations correctly.
    """

    def __init__(
            self,
            in_neurons: int,
            out_neurons: int,
            device: str = 'cpu',
            module_name: str = "module",
            super_in_name: str = "in_features",
            super_out_name: str = "out_features"
    ) -> None:

        # Variables
        # # Layer variables
        self.super_in_name = super_in_name
        self.in_neurons = in_neurons
        self.super_out_name = super_out_name
        self.out_neurons = out_neurons
        self.module_name = module_name
        self.device = device
        self.tracking_mode = TrackingMode.DISABLED

        # IN/OUT neurons indexing & mapping dictionary
        self.src_to_dst_mapping_tnsrs = {}
        self.dst_to_src_mapping_tnsrs = {}
        self.related_src_to_dst_mapping_tnsrs = {}
        self.related_dst_to_src_mapping_tnsrs = {}

        # Find and save learnable tensors that can be operated
        if hasattr(self, 'named_parameters'):
            # Get every learnable tensors name
            self.learnable_tensors_name = [
                name for name, param in self.named_parameters()
                if param.requires_grad
            ]

            # Initialize masking directory, i.e., for IN/OUT neurons
            # lr to 0 to freeze specific neurons
            self.neuron_2_lr = {
                tensor_name:
                    collections.defaultdict(lambda: 1.0) for tensor_name in
                    self.learnable_tensors_name
            }
            # TODO (GP): weight is the only incoming tensor to be learned ?
            self.incoming_neuron_2_lr = {
                tensor_name:
                    collections.defaultdict(lambda: 1.0) for tensor_name in
                    self.learnable_tensors_name if tensor_name == 'weight'
            }

        # Naming
        self.assign_id()  # assign ids

        # Tracking
        self.register_trackers()

        # Register hooks
        self.register_grad_hook()

    # ===================
    # Getters and Setters
    # ===================
    def set_name(self, n: str):
        if isinstance(n, str):
            self.module_name = n

    def get_name(self) -> str:
        return self.module_name

    def get_neurons_value(self, v: int | List[int]) -> int:
        """
        Method to handle the specific case when IN/OUT updates are not
        neurons information but shapes, e.g., unflatten layers.
        Initial assumption: v is the shape of the tensor, i.e., (B, C, H, W)
        or (C, H, W) only.
        """
        value = v \
            if isinstance(v, int) else \
            (
                v[1] if len(v) == 4 else
                v[0]
            )
        return value

    def get_neurons(self, attr_name: str) -> int:
        """Getter: Returns the value of in_neurons."""
        if not hasattr(self, attr_name):
            raise AttributeError(
                "Accessing 'in_neurons' before calling" +
                "_initialize_neuron_attributes."
            )
        return self.get_neurons_value(getattr(self, attr_name))

    def set_neurons(
            self,
            attr_name: str,
            new_value: int | List[int] | th.Size | tuple
    ) -> dict:
        """
        Setter (The Hook): This method runs whenever 'in_neurons' is assigned
        a value.
        """
        attr_value = getattr(self, attr_name)
        if isinstance(attr_value, (list, th.Size, tuple)):
            if isinstance(attr_value, tuple):
                attr_value = list(attr_value)
            if len(attr_value) == 4:
                if isinstance(new_value, (int, float)):
                    attr_value[1] = new_value
                else:
                    attr_value = new_value
            elif len(attr_value) == 3:
                if isinstance(new_value, (int, float)):
                    attr_value[0] = new_value
                else:
                    attr_value = new_value
        else:
            attr_value = new_value
        # Update attribute
        return self._update_attr(
            attr_name,
            attr_value
        )

    # =================
    # Magic. Operations
    # =================
    def __eq__(self, other: Callable) -> bool:
        _weight = th.allclose(self.weight.data, other.weight.data)
        _bias = th.allclose(self.bias.data, other.bias.data)
        _train_tracker = self.train_dataset_tracker == \
            other.train_dataset_tracker
        _eval_tracker = self.eval_dataset_tracker == \
            other.eval_dataset_tracker
        return _weight and _bias and _train_tracker and _eval_tracker

    def __hash__(self) -> int:
        """
            Get all related learnable instance attributes,
            e.g.,self.in_features, self.out_features, and self.bias
        """
        params = (
            self.__dict__
        )
        return int(hashlib.sha256(str(params).encode()).hexdigest(), 16)

    # ==================
    # Trackers Functions
    # ==================
    def register_trackers(self):
        is_disabled = bool(getattr(self, "wl_same_flag", False))
        self.register_module('train_dataset_tracker', TriggersTracker(
            self.get_neurons('out_neurons'), device=self.device, disabled=is_disabled))
        self.register_module('eval_dataset_tracker', TriggersTracker(
            self.get_neurons('out_neurons'), device=self.device, disabled=is_disabled))

    def set_tracking_mode(self, tracking_mode: TrackingMode):
        """ Set what samples are the stats related to (train/eval/etc). """
        self.tracking_mode = tracking_mode

    def get_tracker(self) -> Tracker:
        if self.tracking_mode == TrackingMode.TRAIN:
            return self.train_dataset_tracker
        elif self.tracking_mode == TrackingMode.EVAL:
            return self.eval_dataset_tracker
        else:
            return None

    def get_trackers(self) -> List[Tracker]:
        return [self.eval_dataset_tracker, self.train_dataset_tracker]

    # ---------------
    # Utils Functions
    def get_operation(
            self,
            op_type: ArchitectureNeuronsOpType | int,
            **_
    ) -> Callable:
        """
            Get the operation function based on the op_type.

            Args:
                op_type (ArchitectureNeuronsOpType | int): The operation type.

            Returns:
                Callable: The operation function.
        """
        if callable(op_type):
            return op_type  # if already got, just return the fct
        elif op_type == ArchitectureNeuronsOpType.ADD or \
                op_type == ArchitectureNeuronsOpType.ADD.value:
            return self._add_neurons
        elif op_type == ArchitectureNeuronsOpType.PRUNE or \
                op_type == ArchitectureNeuronsOpType.PRUNE.value:
            return self._prune_neurons
        elif op_type == ArchitectureNeuronsOpType.FREEZE or \
                op_type == ArchitectureNeuronsOpType.FREEZE.value:
            return self._freeze_neurons
        elif op_type == ArchitectureNeuronsOpType.RESET or \
                op_type == ArchitectureNeuronsOpType.RESET.value:
            return self._reset_neurons

    def get_per_neuron_learning_rate(
            self,
            neurons_id: int,
            is_incoming: bool = 'False',
            tensor_name: str = 'weight'
    ) -> float:
        """
            Get the learning rate for a specific neuron.

            Args:
                neuron_id (int): The neuron id to get the learning rate for.

            Returns:
                float: The learning rate for the specific neuron.
        """
        neuron_2_lr = self.neuron_2_lr if not is_incoming else \
            self.incoming_neuron_2_lr

        if not neuron_2_lr or \
                tensor_name not in neuron_2_lr:
            return [1.0] * len(neurons_id)
        return neuron_2_lr[tensor_name][neurons_id]

    def set_per_neuron_learning_rate(
            self,
            neurons_id: Set[int],
            neurons_lr: Set[float],
            tensor_name: str = 'weight',
            is_incoming: bool = False,
    ):
        """
            Set per neuron learning rates.

            Args:
                neurons_id (Set[int]): The set of neurons to set the learning rate
                lr (float): The value of the learning rate. Can be between [0, 1]
        """
        if isinstance(neurons_id, (list, set)):
            if isinstance(neurons_id, set):
                neurons_id = list(neurons_id)
                if isinstance(neurons_id[0], int):
                    neurons_id = [i for i in neurons_id]
                else:
                    neurons_id = set(neurons_id)

        # Manage incoming module
        in_out_neurons = self.get_neurons(attr_name='out_neurons') if not is_incoming else \
            self.get_neurons(attr_name='in_neurons')
        neuron_2_lr = self.neuron_2_lr if not is_incoming else \
            self.incoming_neuron_2_lr

        # Sanity Check
        invalid_ids = (set(neurons_id) - set(range(in_out_neurons)))
        if invalid_ids:
            raise ValueError(
                f'Layer={self.get_name()}[id={self.module_id}]:'
                f'Cannot set learning rate for neurons {invalid_ids} as they '
                f'are outside the set of existent neurons {in_out_neurons}.'
            )

        # Update neuron lr
        for neuron_id in neurons_id:
            lr = neurons_lr[neuron_id]
            if lr < 0 or lr > 1.0:
                raise ValueError(
                    'Cannot set learning rate outside [0, 1] range'
                )
            if neuron_id in neuron_2_lr[tensor_name] and lr == 1:
                del neuron_2_lr[tensor_name][neuron_id]
            else:
                neuron_2_lr[tensor_name][neuron_id] = lr

    def register_grad_hook(self):
        """
            Register a hook for the gradient of the weights and biases.
            This hook will be called when the gradients are computed and
            constraints the gradients based on the neuron learning rate.
        """
        def create_tensor_grad_hook(tensor_name: str, oneD: bool):
            def weight_grad_hook(weight_grad):
                if tensor_name in self.neuron_2_lr:
                    for neuron_id, neuron_lr in \
                            self.neuron_2_lr[tensor_name].items():
                        if neuron_id >= weight_grad.shape[0]:
                            continue
                        neuron_grad = weight_grad[neuron_id]
                        neuron_grad *= neuron_lr
                        weight_grad[neuron_id] = neuron_grad
                if tensor_name in self.incoming_neuron_2_lr and weight_grad.ndim > 1:
                    for in_neuron_id, neuron_lr in \
                            self.incoming_neuron_2_lr[tensor_name].items():
                        if in_neuron_id >= weight_grad.shape[1]:
                            continue
                        in_neuron_grad = weight_grad[:, in_neuron_id]
                        in_neuron_grad *= neuron_lr
                        weight_grad[:, in_neuron_id] = in_neuron_grad
                return weight_grad

            def oneD_grad_hook(bias_grad):
                for neuron_id, neuron_lr in \
                        self.neuron_2_lr[tensor_name].items():
                    if neuron_id >= bias_grad.shape[0]:
                        continue
                    neuron_grad = bias_grad[neuron_id]
                    neuron_grad *= neuron_lr
                    bias_grad[neuron_id] = neuron_grad
                return bias_grad

            return weight_grad_hook if not oneD else oneD_grad_hook

        # Attribute hooks to corresponding learnable tensors
        for tensor_name in self.learnable_tensors_name:
            tensor = getattr(self, tensor_name)
            if tensor is not None:
                hook_fct = create_tensor_grad_hook(
                    tensor_name,
                    oneD=len(tensor) == 1
                )
                self.weight.register_hook(hook_fct)

    def _update_attr(self, attribute_name: str, value: int) -> int:
        """
            Update the attribute value.

            Args:
                attribute_name (str): The name of the attribute to update.
                value (int): The new value to set.

            Returns:
                int: The updated value.
        """
        # Get the current value using getattr()
        if not hasattr(self, attribute_name):
            return

        # Set the new value back using setattr()
        setattr(self, attribute_name, value)

        return value

    def _find_value_for_key_pattern(
            self,
            key_pattern: str,
            state_dict: dict
    ) -> object:
        """
            Find the value for a key pattern in the state_dict.

            Args:
                key_pattern (str): The pattern to search for in the keys.
                state_dict (dict): The state dictionary.

            Returns:
                object: The value corresponding to the key pattern, or None
                if not found.
        """
        for key, value in state_dict.items():
            if key_pattern in key:
                return value

    def _process_neurons_indices(
            self,
            neuron_indices: Set[int] | int,
            is_incoming: bool = False,
            current_child_name: str = None,
            current_parent_name: str = None,
            **_
    ) -> Tuple[List[int], List[int]]:
        """
        Intelligently processes high-level logical indices (like channels
        or neurons) into the flat, absolute tensor indices required for
        pruning.

        This function handles:
        1. 1-to-1 Mappings (Linear -> Linear, Conv -> Conv)
        2. N-to-1 Mappings (Conv(N) -> Flatten -> Linear(N*H*W))
        3. 1-to-N Mappings (Linear(N*H*W) -> Unflatten -> Conv(N))

        Args:
            neuron_indices (Set[int] | int): The neuron indices to process.
            is_incoming (bool): Whether the indices are incoming.
            current_child_name (str): The name of the current child.
            current_parent_name (str): The name of the current parent.

        Returns:
            Tuple[List[int], List[int]]: The absolute tensor indices and
            the original indices.
        """
        # --- Normalize neuron_indices to a non-empty set ---
        if neuron_indices is None:
            neuron_indices = {-1}
        elif not isinstance(neuron_indices, set):
            # If it's a single int, wrap; if it's iterable, cast to set
            try:
                neuron_indices = set(neuron_indices)  # type: ignore[arg-type]
            except TypeError:
                neuron_indices = {neuron_indices}

        if not neuron_indices:
            neuron_indices = {-1}

        # --- Select the appropriate mapping dict (if any) ---

        mapped_indices_dict = None

        # INCOMING: use dst_to_src_mapping_tnsrs[parent] if available
        if (
            is_incoming
            and current_parent_name is not None
            and current_parent_name in self.dst_to_src_mapping_tnsrs
        ):
            mapped_indices_dict = self.dst_to_src_mapping_tnsrs[current_parent_name]

        # OUTGOING or fallback: use src_to_dst_mapping_tnsrs[child] if available
        elif len(self.src_to_dst_mapping_tnsrs):
            # Prefer an explicit child name if it's present
            key = None
            if current_child_name is not None and current_child_name in self.src_to_dst_mapping_tnsrs:
                key = current_child_name
            else:
                # Fallback: just take the first available mapping
                key = next(iter(self.src_to_dst_mapping_tnsrs.keys()))

            mapped_indices_dict = self.src_to_dst_mapping_tnsrs.get(key, None)

        # If we found a mapping dict, normalize it
        if mapped_indices_dict is not None:
            mapped_indexs = normalize_dicts(
                {"mapped": mapped_indices_dict}
            )["mapped"]  # TODO (GP): Improve this function
        else:
            # No mapping tensors available: fall back to identity mapping
            n_neurons = self.get_neurons(
                attr_name="in_neurons" if is_incoming else "out_neurons"
            )
            mapped_indexs = {i: [i] for i in range(n_neurons)}

        # --- Map logical indices to original indices (reversed order) ---

        original_indexs = reversing_indices(
            len(mapped_indexs),
            neuron_indices
        )

        # --- Compute flat indices, taking into account possible offset ---

        # If there's a bypass / shifted index base, we need to adjust
        offset = min(mapped_indexs.keys()) if len(mapped_indexs) else 0

        flat_indexs: List[List[int]] = []
        for i in original_indexs:
            key = len(mapped_indexs) + i + offset
            # Be robust to missing keys: fall back to identity when needed
            if key in mapped_indexs:
                flat_indexs.append(mapped_indexs[key][::-1])
            else:
                # Identity fallback for this index
                flat_indexs.append([key])

        return flat_indexs, original_indexs

    # ---------------
    # Torch Functions
    def load_from_state_dict(
            self,
            state_dict: dict,
            prefix: str,
            local_metadata: dict,
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            error_msgs: str
    ):
        """
        Load the model from a state dictionary.

        Args:
            state_dict (dict): The state dictionary to load from.
            prefix (str): The prefix to use when searching for keys in the state_dict.
            local_metadata (dict): The local metadata to use when searching for keys in the state_dict.
            strict (bool): Whether to strictly enforce that the keys in the state_dict match
                the keys in this module.
            missing_keys (List[str]): List of keys that are missing in the state_dict.
            unexpected_keys (List[str]): List of keys that are present in the state_dict but not in this module.
            error_msgs (str): Error messages to return.
        """
        tnsr = self._find_value_for_key_pattern('weight', state_dict)
        if tnsr is not None:
            in_size, out_size = tnsr.shape[1], tnsr.shape[0]
            with th.no_grad():
                wshape = (out_size, in_size)
                self.weight.data = nn.Parameter(
                    th.ones(wshape)).to(self.device)
                if self.bias is not None:
                    self.bias.data = nn.Parameter(
                        th.ones(out_size)).to(self.device)

            # Update neurons information
            self.set_neurons(
                attr_name='in_neurons',
                new_value=self._update_attr(self.super_in_name, in_size)
            )
            self.set_neurons(
                attr_name='out_neurons',
                new_value=self._update_attr(self.super_out_name, out_size)
            )
        # Update state dict from torch.nn.Module class
        self._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def to(self, *args, **kwargs):
        """
        Move all tensors and modules to a specified device.

        Args:
            args: Positional arguments passed to the `to` method of the
            underlying tensors.
            kwargs: Keyword arguments passed to the `to` method of the
            underlying tensors.
        """
        self.device = args[0]
        for tracker in self.get_trackers():
            tracker.to(*args, **kwargs)

    def register(
            self,
            activation_map: th.Tensor
    ):
        """
        Register the activation map to the layer.

        Args:
            activation_map (th.Tensor): The activation map for the layer.
        """
        tracker = self.get_tracker()
        if tracker is None or activation_map is None or input is None:
            return
        activation_map = (activation_map > 0).long()  # bool to int
        processed_activation_map = th.sum(activation_map, dim=(-2, -1))
        copy_forward_tracked_attrs(processed_activation_map, activation_map)
        tracker.update(processed_activation_map)

    def perform_layer_op(
        self,
        activation_map: th.Tensor,
        data: th.Tensor,
        skip_register: bool = False,
        intermediary: dict | None = None
    ) -> th.Tensor:
        # Update tensor information
        copy_forward_tracked_attrs(activation_map, data)

        # Sanity check
        if not skip_register:
            self.register(activation_map)
        
        # Handle intermediary layers
        if intermediary is not None and self.get_module_id() in intermediary:
            try:
                intermediary[self.get_module_id()] = activation_map
            except Exception as e:
                print(
                    f"Error {e} occurred while updating intermediary outputs",
                    self.get_module_id(), str(activation_map)[:50],
                    level='ERROR')

        return activation_map

    def get_canal_length(self, incoming=False, parent_name=None, child_name=None):
        if incoming:
            dst_to_src_keys = list(self.dst_to_src_mapping_tnsrs.keys())
            if not len(dst_to_src_keys):
                return 1
            parent_name = dst_to_src_keys[0] if parent_name is None else parent_name
            last_k = list(self.dst_to_src_mapping_tnsrs[parent_name].keys())[-1]
            last_channel = self.dst_to_src_mapping_tnsrs[parent_name][last_k]
            return len(last_channel) if isinstance(last_channel, list) else 1
        else:
            src_to_dst_keys = list(self.src_to_dst_mapping_tnsrs.keys())
            if not len(src_to_dst_keys):
                return 1
            child_name = src_to_dst_keys[0] if child_name is None else child_name
            last_k = list(self.src_to_dst_mapping_tnsrs[child_name].keys())[-1]
            last_channel = self.src_to_dst_mapping_tnsrs[child_name][last_k]
            return len(last_channel) if isinstance(last_channel, list) else 1

    def operate(
        self,
        neuron_indices: int | Set[int],
        is_incoming: bool = False,
        skip_initialization: bool = False,
        neuron_operation: Enum = ArchitectureNeuronsOpType.ADD,
        **kwargs
    ):
        """
        Operate on the neurons.

        TODO (GP): Global Improvements with
        ----------------------------
            a- Some Hardcoded layers types; i.e., with tensor_name_learnable,
        we should be able to update tensors just with its shape, unregarding
        of its name or type based on mapping objects like
        src_to_dst_mapping_tnsrs.
            b- Indexing approach updates can be improve again and refactorize
        as a single function that will update the tensors.

        Args:
            neuron_indices (int | Set[int]): The indices of the neurons to operate on.
            is_incoming (bool): Whether the operation is on the incoming side.
            skip_initialization (bool): Whether to skip the initialization step.
            neuron_operation (Enum): The operation to perform on the neurons.
            kwargs: Additional keyword arguments for the operation.
        """

        # Get Operation
        neuron_operation = ArchitectureNeuronsOpType(neuron_operation)
        op = self.get_operation(neuron_operation)

        # Ensure set of neurons index
        if not isinstance(neuron_indices, set) and \
                isinstance(neuron_indices, int) and \
                neuron_operation != ArchitectureNeuronsOpType.ADD:
            neuron_indices = {neuron_indices}

        # Get Neurons Indexs Formatted for Pruning, Reset, or Frozen only.
        # Except on ADD because we don't look at the index, topped the
        # neurons.
        # Both neuron indices and original exists because sometime with bypass
        # flag, both can be different.
        if isinstance(neuron_indices, int) or isinstance(neuron_indices, set) \
                and len(neuron_indices) > 0:
            neuron_indices, original_neuron_indices = \
                self._process_neurons_indices(
                    neuron_indices,
                    is_incoming=is_incoming,
                    **kwargs
                )
        else:
            original_neuron_indices = neuron_indices

        # Sanity check if neuron_indices is correctly defined
        if not len(neuron_indices) and \
                (
                    neuron_operation == ArchitectureNeuronsOpType.ADD or
                    neuron_operation == ArchitectureNeuronsOpType.PRUNE
                ):
            raise IndexError(
                "[LayerWiseOperations.operate] Neurons index were not found.")
        if not len(neuron_indices):
            neuron_indices, original_neuron_indices = [None], [None]

        # Operate on neurons
        for neuron_indices_, original_neuron_indices_ in zip(
            neuron_indices,
            original_neuron_indices
        ):
            # Process the neuron_indices to match the layer's constraints
            # e.g., Conv's groups parameter
            op(
                original_neuron_indices=original_neuron_indices_,
                neuron_indices=neuron_indices_,
                is_incoming=is_incoming,
                skip_initialization=skip_initialization,
                **kwargs
            )

    # ------------------
    # Neurons Operations
    def _process_input_neurons_index(
            self,
            neuron_indices: int | Set[int]
    ) -> int | Set[int]:
        """
            Sanity function to process neuron indices.

            Args:
                neuron_indices (int | Set[int]): Neuron indices to process.

            Returns:
                int | Set[int]: Processed neuron indices.
        """
        if neuron_indices is None:
            return neuron_indices

        if not isinstance(neuron_indices, set):
            if isinstance(neuron_indices, list):
                neuron_indices = set(neuron_indices)
            else:
                neuron_indices = set(neuron_indices) if not \
                    isinstance(neuron_indices, int) else set([neuron_indices])
        return neuron_indices

    def _add_neurons(
        self,
        neuron_indices: Set[int] | int = -1,
        is_incoming: bool = False,
        skip_initialization: bool = False,
        original_neuron_indices: Optional[List[int]] = None,
        dependency: Optional[Callable] = None,
        **kwargs
    ):
        """
            Add neurons to the layer.

            Args:
                neuron_indices (Set[int] | int): Neuron indices to add.
                is_incoming (bool): Whether the operation is incoming or out.
                skip_initialization (bool): Whether to skip the initialization
                    of the neurons.
                original_neuron_indices (List[int]): Original neuron indices.
                dependency (Callable): Dependency function to call before
                    adding the neurons.
                kwargs: Additional keyword arguments.        
        """

        print(
            f"{self.get_name()}[{self.get_module_id()}].add {neuron_indices}",
            level='DEBUG'
        )

        # Process neuron indices
        neuron_indices = self._process_input_neurons_index(neuron_indices)
        original_neuron_indices = self._process_input_neurons_index(
            original_neuron_indices
        )

        # Incoming operation or out operation; chose the right neurons
        # # TODO (GP): fix hardcoding transpose and norm
        # # TODO (GP): maybe with a one way upgrade from learnable tensors,
        # # TODO (GP): depending on the weight shape (1d, 2d, 3d, ..etc).
        norm = False
        transposed = int('transpose' in self.get_name().lower())
        in_out_neurons = self.get_neurons(attr_name='out_neurons') \
            if is_incoming else self.get_neurons(attr_name='in_neurons')
        # We consider in the weight operation on incoming neurons only,
        # the cluster_size (e.g., groups parameter for convolutions).
        # Here, for a group of 4, we add 1 neuron to the weight matrix and
        # we increase also the number of incoming neurons in the layer.
        group_size = self.groups if hasattr(self, 'groups') \
            else 1
        nb_neurons = self.get_canal_length(
            is_incoming,
            parent_name=kwargs.get('current_parent_name', None),
            child_name=kwargs.get('current_child_name', None)
        ) if len(neuron_indices) == 1 else len(neuron_indices)
        if is_incoming == transposed:
            tensors = (nb_neurons, in_out_neurons // group_size)
        else:
            tensors = (in_out_neurons, nb_neurons // group_size)

        # TODO (GP): fix hardcoding layers op. 1d vs 2d
        # TODO (GP): maybe with a one way upgrade from learnable tensors.
        # Weights
        if hasattr(self, "weight") and self.weight is not None:
            # # Handle n-dims kernels like with conv{n}d
            if hasattr(self, "kernel_size") and self.kernel_size:
                added_weights = th.zeros(
                    tensors + (*self.kernel_size,)
                ).to(self.device)

            # # Handle 1-dims cases like batchnorm without in out mapping
            elif len(self.weight.data.shape) == 1:
                norm = True
                added_weights = th.ones(tensors[0], ).to(self.device)

            # # Handle 1-dims cases like linear, where we have a in out mapping
            # # (similar to conv1d wo. kernel)
            else:
                added_weights = th.zeros(
                    tensors
                ).to(self.device)

            # Biases
            if not is_incoming:
                added_bias = th.zeros(nb_neurons).to(self.device)

            if not norm:
                # Initialization
                if not skip_initialization:
                    nn.init.xavier_uniform_(added_weights,
                                            gain=nn.init.calculate_gain('relu'))

                # Update
                with th.no_grad():
                    # TODO (GP): fix hardcoding transpose approach ?
                    self.weight.data = nn.Parameter(
                        th.cat(
                            (
                                self.weight.data.to(self.device),
                                added_weights
                            ),
                            dim=(transposed ^ is_incoming) & int(
                                len(
                                    self.weight.data.flatten()
                                ) > 1
                            )
                        )
                    )

                    if hasattr(self, 'bias') and self.bias is not None and \
                            not is_incoming:
                        self.bias.data = nn.Parameter(
                            th.cat((self.bias.data.to(self.device), added_bias))
                        ).to(self.device)
            else:
                # Update
                with th.no_grad():
                    self.weight.data = nn.Parameter(
                        th.cat((self.weight.data.to(self.device), added_weights))
                    )
                    if hasattr(self, 'bias') and self.bias is not None:
                        self.bias.data = nn.Parameter(
                            th.cat((self.bias.data.to(self.device), added_bias))
                        )

                    if hasattr(self, 'running_mean') and \
                            self.running_mean is not None:
                        self.running_mean = th.cat((
                            self.running_mean.to(self.device),
                            th.zeros(nb_neurons).to(self.device)))
                    if hasattr(self, 'running_var') and \
                            self.running_var is not None:
                        self.running_var = th.cat((
                            self.running_var.to(self.device),
                            th.ones(nb_neurons).to(self.device))
                        )

        # ----------------------------------
        # ----- Neurons Mapping Update -----
        # ----------------------------------
        # Outcoming neurons
        if not is_incoming:
            # Update neurons count
            self.set_neurons(
                attr_name='out_neurons',
                new_value=self.set_neurons(
                    self.super_out_name,
                    self.get_neurons(self.super_out_name) + nb_neurons
                )
            )

            # Update the src2dst mapping dictionary
            # # Get dependencies
            deps_names = list(self.src_to_dst_mapping_tnsrs.keys())
            if len(deps_names) > 0:
                # Update mapping dictionary
                # We get the length of the first dependency to know the length
                # of every src2dst mapping tensor, as src is equal to dst,
                # except for Linear layers for instance.
                # TODO (GP): Not true when output to multi input (e.g., Conv2d, Linear, Conv2d)
                # TODO (GP): Modulo should not be used but indice - sum(previous lengths)
                length = len(self.src_to_dst_mapping_tnsrs[deps_names[0]]) \
                    if len(deps_names) > 0 else None
                for deps_name in deps_names:
                    for neuron_indice in neuron_indices:
                        # Get the corresponding dst indexs (e.g., Linear)
                        neuron_indice = list(
                            self.src_to_dst_mapping_tnsrs[
                                deps_name
                            ].keys()
                        )[neuron_indice % length]

                        # Update the mapping tensor wiht 1 neurons or
                        # range(x) neurons if its a Linear layer, i.e.,
                        # we add several neurons for a new convolutional
                        # channel.
                        self.src_to_dst_mapping_tnsrs[
                            deps_name
                        ].update(
                            {
                                neuron_indice + 1: [
                                    i + max(
                                        self.src_to_dst_mapping_tnsrs[
                                            deps_name
                                        ][neuron_indice]
                                    ) for i in range(1, nb_neurons + 1)
                                ]
                            }
                        )
                # Normalize mapping dictionary
                # Normalize the mapping tensors to ensure continuouty between
                # mapping and neurons, i.e., {'a': [1, 2, 3], 'b': [4, 5, 6]}.
                self.src_to_dst_mapping_tnsrs = normalize_dicts(
                    self.src_to_dst_mapping_tnsrs
                )

            # # Update the dst2src mapping dictionary from child module
            current_name = self.get_name_wi_id()
            if current_name in self.related_dst_to_src_mapping_tnsrs:
                # Update mapping dictionary
                length = len(
                    self.related_dst_to_src_mapping_tnsrs[current_name]
                )
                if length > 0:
                    # Determine the channel size, e.g., 1 for conv2d (1 <> 1), but
                    # x for linear layers (1 <> x) with linear in
                    # neuron x times out neurons.
                    channel_size = len(
                        list(
                            self.related_dst_to_src_mapping_tnsrs[
                                current_name
                            ].values()
                        )[-1]
                    )
                    for neuron_indice in neuron_indices:
                        mapped_neuron_indice = list(
                            self.related_dst_to_src_mapping_tnsrs[
                                current_name
                            ].keys()
                        )[neuron_indice % length]  # get new index

                        # Update the mapping tensor with 1 or range(x) neurons
                        self.related_dst_to_src_mapping_tnsrs[
                            current_name
                        ].update(
                            {
                                mapped_neuron_indice + 1: [
                                    neuron_index + max(
                                        self.related_dst_to_src_mapping_tnsrs[
                                            current_name
                                        ][mapped_neuron_indice]
                                    ) for neuron_index, _ in enumerate(
                                        range(
                                            (mapped_neuron_indice - 1) *
                                            channel_size,
                                            mapped_neuron_indice * channel_size
                                        )
                                    )  # in range of x neurons
                                ]
                            }
                        )

                    # Normalize mapping dictionary
                    self.related_dst_to_src_mapping_tnsrs = normalize_dicts(
                        self.related_dst_to_src_mapping_tnsrs
                    )

            # Update the trackers with new neurons
            for tracker in self.get_trackers():
                tracker.add_neurons(nb_neurons)

            # Verbose
            print(f'Add one neuron to layer {self}', level='DEBUG')

        # Incoming neurons, e.g., in conv2d for instance, or in norm
        if is_incoming or dependency == DepType.SAME:
            if dependency != DepType.SAME:
                # We don't need to update here if already done before
                # for same flag layers (e.g., batchnorm).
                self.set_neurons(
                    attr_name='in_neurons',
                    new_value=self.set_neurons(
                        self.super_in_name,
                        self.get_neurons(self.super_in_name) + nb_neurons
                    )
                )  # Update neurons count

            # By default get deps name from current relation
            deps_names = list(self.dst_to_src_mapping_tnsrs.keys())
            if len(deps_names) > 0:
                # Update mapping dictionary
                # TODO (GP): Same comment as before src2dst
                length = len(self.dst_to_src_mapping_tnsrs[deps_names[0]])
                for deps_name in deps_names:
                    neuron_indice = list(
                        self.dst_to_src_mapping_tnsrs[deps_name].keys()
                    )[-1]
                    mapped_neuron_indice = list(
                        self.dst_to_src_mapping_tnsrs[
                            deps_name
                        ].keys()
                    )[neuron_indice % length]
                    self.dst_to_src_mapping_tnsrs[
                        deps_name
                    ].update(
                        {
                            mapped_neuron_indice + 1 + j: [
                                i + 1 + max(
                                    self.dst_to_src_mapping_tnsrs[
                                        deps_name
                                    ][mapped_neuron_indice]
                                ) for i in range(0, nb_neurons)  # neurons
                            ] for j in range(0, nb_neurons)  # neurons | chan.
                        }
                    )

                # Normalize mapping dictionary
                self.dst_to_src_mapping_tnsrs = normalize_dicts(
                    self.dst_to_src_mapping_tnsrs
                )

            # # Related neurons
            current_name = self.get_name_wi_id()
            nb_original_neurons_channels = len([original_neuron_indices])
            if current_name in self.related_src_to_dst_mapping_tnsrs:
                # Update mapping dictionary
                length = len(
                    self.related_src_to_dst_mapping_tnsrs[
                        current_name
                    ]
                )
                if length > 0:
                    indexs = list(
                        self.related_src_to_dst_mapping_tnsrs[
                            current_name
                        ].keys()
                    )
                    neuron_indice = indexs[neuron_indice % length]
                    self.related_src_to_dst_mapping_tnsrs[
                        current_name
                    ].update(
                        {
                            neuron_indice + 1: [
                                i + 1 + max(
                                    self.related_src_to_dst_mapping_tnsrs[
                                        current_name
                                    ][neuron_indice]
                                ) for i in range(
                                    0,
                                    nb_original_neurons_channels
                                )
                            ]
                        }
                    )

                    # Normalize mapping dictionary
                    self.related_src_to_dst_mapping_tnsrs = normalize_dicts(
                        self.related_src_to_dst_mapping_tnsrs
                    )

            # Verbose
            print(
                f'New {"INCOMING" if dependency != DepType.SAME else "SAME"}' + 
                f'layer is {self}', level='DEBUG'
            )

    def _prune_neurons(
        self,
        original_neuron_indices: Set[int],
        neuron_indices: List | Set[int],
        is_incoming: bool = False,
        dependency: Optional[Callable] = None,
        **kwargs
    ):
        """
        Prune neurons from the layer based on the provided indices.

        Args:
            original_neuron_indices (Set[int]): Indices of neurons that should remain after pruning.
            neuron_indices (List | Set[int]): Indices of neurons to be pruned.
            is_incoming (bool): Indicates if the pruning operation is for an incoming layer.
            dependency (Optional[Callable]): Dependency callback function.
            **kwargs: Additional keyword arguments.
        """

        print(
            f"{self.get_name()}[{self.get_module_id()}].prune {neuron_indices}",
            level='DEBUG'
        )

        # Process neuron indices
        neuron_indices = self._process_input_neurons_index(neuron_indices)
        original_neuron_indices = self._process_input_neurons_index(
            original_neuron_indices
        )

        # Check if it's a transposed layer
        transposed = int('transpose' in self.get_name().lower())

        # Get the number of corresponding layer weights
        in_out_neurons = self.get_neurons(attr_name='out_neurons') if not is_incoming else \
            self.get_neurons(attr_name='in_neurons')

        # Get current weights indexs & group size
        neurons = set(range(in_out_neurons))
        group_size = self.groups if hasattr(self, 'groups') and is_incoming \
            else 1

        # Sanity check
        # # Overlapping neurons index and neurons available
        if not set(neuron_indices) & neurons:
            print(
                f"{self.get_name()}.prune indices and neurons set do not "
                f"overlapp: {neuron_indices} & {neurons} => \
                    {neuron_indices & neurons}",
                level='WARNING'
            )
            return  # Do not change

        # # Enough neurons to operate
        if len(neurons) <= 1:
            print(f'Not enough neurons to operate (currently {neurons})')
            return

        # Tensor indices to keep
        idx_tokeep = neurons - neuron_indices
        idx_tnsr = th.unique(
            th.Tensor(list(idx_tokeep)).long() // group_size
        ).to(self.device)

        # TODO (GP): fix hardcoding layers op. 1d vs 2d
        # TODO (GP): maybe with a one way upgrade from learnable tensors.
        # Operate on learnable tensor
        if hasattr(self, 'weight') and self.weight is not None:
            # Tensors modification
            with th.no_grad():
                self.weight.data = nn.Parameter(
                    th.index_select(
                        self.weight.data,
                        dim=(transposed ^ is_incoming) if self.weight.data.ndim > 1 else 0,
                        index=idx_tnsr
                    )).to(self.device)

                if hasattr(self, 'bias') and self.bias is not None:
                    self.bias.data = nn.Parameter(
                        th.index_select(
                            self.bias.data,
                            dim=0,
                            index=idx_tnsr
                        )).to(self.device) if (not is_incoming or self.weight.data.ndim == 1) else \
                            self.bias.data
                if hasattr(self, 'running_mean'):
                    print(f"[DEBUG] {self.get_name()} updating running_mean. Old shape: {self.running_mean.shape}")
                    self.running_mean = th.index_select(
                        self.running_mean, dim=0, index=idx_tnsr).to(
                            self.device
                        )
                    print(f"[DEBUG] {self.get_name()} updated running_mean. New shape: {self.running_mean.shape}")
                if hasattr(self, 'running_var'):
                    print(f"[DEBUG] {self.get_name()} updating running_var. Old shape: {self.running_var.shape}")
                    self.running_var = th.index_select(
                        self.running_var, dim=0, index=idx_tnsr).to(
                            self.device
                        )
                    print(f"[DEBUG] {self.get_name()} updated running_var. New shape: {self.running_var.shape}")
                
        # Sort indices to prune from last to first to maintain
        # the original order
        neuron_indices = sorted(neuron_indices)[::-1]
        if not is_incoming:
            # Update neurons count
            self.set_neurons(
                attr_name='out_neurons',
                new_value=self.set_neurons(
                    self.super_out_name,
                    len(idx_tokeep)
                )
            )

            # Update the src2dst mapping dictionary
            # # Get dependencies
            deps_names = list(self.src_to_dst_mapping_tnsrs.keys())
            if len(deps_names) > 0:
                # Update mapping dictionary
                # We get the length of the first dependency to know the length
                # of every src2dst mapping tensor, as src is equal to dst,
                # except for Linear layers for instance.
                # TODO (GP): Not true when output to multi input (e.g., Conv2d, Linear, Conv2d)
                # TODO (GP): Modulo should not be used but indice - sum(previous lengths)
                length = len(self.src_to_dst_mapping_tnsrs[deps_names[0]]) \
                    if len(deps_names) > 0 else None
                for deps_name in deps_names:
                    for neuron_indice in neuron_indices:
                        # Get the corresponding dst indexs (e.g., Linear)
                        neuron_indice = neuron_indice % length
                        index_map = list(
                            self.src_to_dst_mapping_tnsrs[
                                deps_name
                            ].keys()
                        )

                        # Remove from the mapping tensor the prune neurons
                        # or range(x) neurons if its a Linear layer, i.e.,
                        # we prune several neurons for a new corresponding
                        # convolutional channel.
                        try:
                            self.src_to_dst_mapping_tnsrs[
                                deps_name
                            ].pop(
                                index_map[neuron_indice]
                            )
                        except IndexError as e:
                            # TODO (GP): Fix specific error with these models
                            # Condition because it raise IndexError in utests for
                            # models:
                            # TwoLayerUnflattenNet & ToyAvgPoolNet
                            print(f'IndexError: {deps_name}; Error: {str(e)}', level='ERROR')
                # Normalize
                self.src_to_dst_mapping_tnsrs = normalize_dicts(
                    self.src_to_dst_mapping_tnsrs
                )

            # TODO (GP): Not sure how relevant try except are now TODELETE
            # try:
            for rel_name in self.related_dst_to_src_mapping_tnsrs:
                length = len(
                    self.related_dst_to_src_mapping_tnsrs[rel_name]
                )
                if length > 0:
                    indexs = list(self.related_dst_to_src_mapping_tnsrs[
                        rel_name
                    ].keys())
                    for neuron_indice in neuron_indices:
                        if neuron_indice >= length:
                            continue
                        neuron_indice = neuron_indice % length

                        # Remove indexs from all childs as its a src
                        self.related_dst_to_src_mapping_tnsrs[
                                rel_name
                            ].pop(neuron_indice)

            # Normalize mapping dictionary
            self.related_dst_to_src_mapping_tnsrs = normalize_dicts(
                self.related_dst_to_src_mapping_tnsrs
            )

            # except KeyError as e:  TODELETE
            #     print(f'IndexError: {rel_name}; Error: {str(e)}')
            # Tracker
            for tracker in self.get_trackers():
                tracker.prune(neuron_indices)

            # Verbose
            print(f'Prune neurons from the layer: {self}', level='DEBUG')

        # Incoming neurons, e.g., in conv2d for instance, or in norm
        if is_incoming or dependency == DepType.SAME:
            # We don't need to update here if already done before
            # for same flag layers (e.g., batchnorm).
            if dependency != DepType.SAME:
                self.set_neurons(
                    attr_name='in_neurons',
                    new_value=self.set_neurons(
                        self.super_in_name,
                        len(idx_tokeep)
                    )
                )

            # Update index maps
            current_parent_name = kwargs.get('current_parent_name', [])
            # By default get deps name from current relation
            deps_name = kwargs.get(
                'current_child_name',
                kwargs.get('current_parent_name')
            )
            deps_names = self.dst_to_src_mapping_tnsrs.keys() \
                if deps_name not in self.dst_to_src_mapping_tnsrs \
                else [deps_name]
            if len(deps_names) > 0:
                for dep_name in deps_names:
                    length = len(self.dst_to_src_mapping_tnsrs[dep_name])
                    if length > 0 and not hasattr(self, 'bypass') or \
                            (hasattr(self, 'bypass')
                             and dep_name in
                             current_parent_name):
                        indexs = list(self.dst_to_src_mapping_tnsrs[
                            dep_name
                        ].keys())
                        for neuron_indice in sorted(neuron_indices)[::-1]:
                            # try: TODELETE
                            # Prune corresponding neurons
                            self.dst_to_src_mapping_tnsrs[
                                dep_name
                            ].pop(indexs[neuron_indice % length])
                            # except IndexError:
                            #     # pass for specific batch linear case, where you have only one neuron [32: [32, 32, 34, 35]], shoud have create 3 others but only one required
                            #     pass

                # Normalize mapping dictionary
                self.dst_to_src_mapping_tnsrs = normalize_dicts(self.dst_to_src_mapping_tnsrs)

            # # Related neurons
            # try: TODELETE
            for rel_name in self.related_src_to_dst_mapping_tnsrs:
                # Update mapping dictionary
                length = len(
                    self.related_src_to_dst_mapping_tnsrs[
                        rel_name
                    ]
                )
                if length > 0:
                    indexs = list(self.related_src_to_dst_mapping_tnsrs[
                        rel_name
                    ].keys())
                    if len(indexs) == 0:
                        neuron_indices = []
                    for neuron_indice in neuron_indices:
                        if neuron_indice >= length:
                            continue
                        self.related_src_to_dst_mapping_tnsrs[
                            rel_name
                        ].pop(indexs[neuron_indice % length])
                # Normalize mapping dictionary
                self.related_src_to_dst_mapping_tnsrs = normalize_dicts(
                    self.related_src_to_dst_mapping_tnsrs
                )

            # except KeyError as e: TODELETE
            #     print(f'IndexError: {rel_name}; Error: {str(e)}')

            # Verbose
            print(
                f'New {"INCOMING" if dependency != DepType.SAME else "SAME"}' +
                f'layer is {self}', level='DEBUG'
            )

    def _freeze_neurons(
        self,
        neuron_indices: int | Set[int],
        is_incoming: bool = False,
        **_
    ):
        """
            Freeze specified neurons.

            Args:
                neuron_indices: Neuron indices to freeze.
                is_incoming: Whether to freeze incoming neurons.
        """

        print(
            f"{self.get_name()}[{self.get_module_id()}].freeze {neuron_indices}",
            level='DEBUG'
        )

        # Process neuron indices
        neuron_indices = self._process_input_neurons_index(neuron_indices)

        # If layer is not learnable - no need to freeze
        if not hasattr(self, 'weight') or self.weight is None:
            return

        # Neurons not specified - freeze everything
        if neuron_indices is None or not len(neuron_indices):
            neuron_indices = list(range(self.get_neurons(attr_name='out_neurons')))

        # Get group size & reformat neuron indices
        group_size = self.groups if hasattr(self, 'groups') and is_incoming \
            else 1
        neuron_indices = [i//group_size for i in neuron_indices]

        # Work on the output
        tensors_name = self.learnable_tensors_name if not is_incoming \
            else ['weight']  # Weight is the only learnable tensor input
        for tensor_name in tensors_name:
            neurons_lr = {
                neuron_indices[n]:
                    1.0 - self.get_per_neuron_learning_rate(
                        neuron_indice,
                        is_incoming=is_incoming,
                        tensor_name=tensor_name
                    ) for n, neuron_indice in enumerate(neuron_indices)
            }
            self.set_per_neuron_learning_rate(
                neurons_id=set(neuron_indices),
                neurons_lr=neurons_lr,
                is_incoming=is_incoming,
                tensor_name=tensor_name
            )

    def _reset_neurons(
        self,
        neuron_indices: int | Set[int],
        is_incoming: bool = False,
        skip_initialization: bool = False,
        perturbation_ratio: float | None = None,
        **_
    ):
        """
            Reset neurons in the layer.

            Args:
                neuron_indices: Neuron indices to reset.
                is_incoming: Whether the layer is an incoming layer.
                skip_initialization: Whether to skip the initialization.
                perturbation_ratio: Perturbation ratio for neuron initialization.
        """

        print(
            f"{self.get_name()}[{self.get_module_id()}].reset {neuron_indices}",
            level='DEBUG'
        )

        # Process neuron indices
        neuron_indices = self._process_input_neurons_index(neuron_indices)

        # Manage specific usecases
        # # Get group size
        group_size = self.groups if hasattr(self, 'groups') else 1
        # # Incoming Layer
        in_out_neurons = self.get_neurons(attr_name='out_neurons') if not is_incoming else \
            self.get_neurons(attr_name='in_neurons') // group_size
        out_in_neurons = self.get_neurons(attr_name='out_neurons') if is_incoming else \
            self.get_neurons(attr_name='in_neurons') // group_size
        # # Transposed Layer
        transposed = int('transpose' in self.get_name().lower())
        # # Reset everything
        if neuron_indices is None or not len(neuron_indices):
            neuron_indices = set(range(in_out_neurons))

        # If layer is not learnable - no need to reset anything
        if not hasattr(self, 'weight') or self.weight is None:
            return

        # Skip initialization is only to be able to test the function.
        neurons = set(
            range(
                self.get_neurons(attr_name='out_neurons') if
                not is_incoming else self.get_neurons(attr_name='in_neurons')
            )
        )
        if not set(neuron_indices) & neurons:
            raise ValueError(
                f"{self.get_name()}.reset neuron_indices and neurons set dont"
                f"overlapp: {neuron_indices} & {neurons} => "
                f"{set(neuron_indices) & neurons}")

        # Perturbation ratio check
        if perturbation_ratio is not None and (
                0.0 >= perturbation_ratio or perturbation_ratio >= 1.0):
            raise ValueError(
                f"{self.get_name()}.reset perturbation "
                f"{perturbation_ratio} outside of [0.0, 1.0]")

        norm = False
        with th.no_grad():
            for neuron_indice in neuron_indices:
                # Process neuron indice
                neuron_indice = neuron_indice // group_size
                # Weights
                # # Handle n-dims kernels like with conv{n}d
                if hasattr(self, "kernel_size") and self.kernel_size:
                    tensors = (out_in_neurons,)
                    neuron_weights = th.zeros(
                        tensors + (*self.kernel_size,)
                    ).to(self.device)

                # # Handle 1-dims cases like batchnorm without in out mapping
                elif len(self.weight.data.shape) == 1:
                    if hasattr(self, 'running_var') and \
                            hasattr(self, 'running_mean'):
                        norm = True
                    tensors = (out_in_neurons,)
                    neuron_weights = th.ones(tensors).to(self.device) if not \
                        norm else 0.

                # # Handle 1-dims cases like linear, where we have a in out
                # # mapping (similar to conv1d wo. kernel)
                else:
                    tensors = (out_in_neurons,)
                    neuron_weights = th.zeros(
                        tensors
                    ).to(self.device)

                neuron_bias = 0.0
                if not norm and not skip_initialization:
                    nn.init.xavier_uniform_(
                        neuron_weights.unsqueeze(0),
                        gain=nn.init.calculate_gain('relu')
                    )
                    if perturbation_ratio is not None:
                        # weights
                        neuron_weights = self.weight[neuron_indice] if not \
                            is_incoming else self.weight[:, neuron_indice]
                        weights_perturbation = \
                            perturbation_ratio * neuron_weights * \
                            th.randint_like(neuron_weights, -1, 2).float()
                        neuron_weights += weights_perturbation

                        # bias
                        if not is_incoming and hasattr(self, 'bias') and \
                                self.bias is not None:
                            neuron_bias = self.bias[neuron_indice]
                            bias_perturbation = \
                                perturbation_ratio * neuron_bias * \
                                th.randint(-1, 2, (1, )).float().item()
                            neuron_bias += bias_perturbation

                # Harcoded case layers vs normed layers
                # TODO (GP): Refactor these appraoch with learnable tensor name
                if not norm:
                    if not transposed and not is_incoming:
                        self.weight[neuron_indice] = neuron_weights
                        if hasattr(self, 'bias') and self.bias is not None and not is_incoming:
                            self.bias[neuron_indice] = neuron_bias
                    elif transposed and not is_incoming:
                        self.weight[:, neuron_indice] = neuron_weights
                        if hasattr(self, 'bias') and self.bias is not None and not is_incoming:
                            self.bias[neuron_indice] = neuron_bias
                    elif not transposed and is_incoming:
                        self.weight[:, neuron_indice] = neuron_weights
                    else:
                        self.weight[neuron_indice] = neuron_weights
                        if hasattr(self, 'bias') and self.bias is not None and not is_incoming:
                            self.bias[neuron_indice] = neuron_bias
                else:
                    self.running_mean[neuron_indice] = neuron_weights
                    self.running_var[neuron_indice] = 1 - neuron_weights
                    self.weight[neuron_indice] = neuron_weights
                    if hasattr(self, 'bias') and self.bias is not None:
                        self.bias[neuron_indice] = neuron_weights
        
        # Update trackers
        if not is_incoming:
            for tracker in self.get_trackers():
                tracker.reset(neuron_indices)


if __name__ == "__main__":
    from weightslab.backend.watcher_editor import WatcherEditor

    # Define the model & the input
    model = Model()
    dummy_input = th.randn(model.input_shape)

    # Run the forward pass
    output = model(dummy_input)

    # Watcher
    model = WatcherEditor(model, dummy_input=dummy_input, print_graph=False)
    model(dummy_input)
    print(model)
    nn_l = len(model.layers)-1

    # Neurons Operation
    # FREEZE
    with model as m:
        m.operate(0, 2, neuron_operation=ArchitectureNeuronsOpType.FREEZE)
        m(dummy_input)
    with model as m:
        m.operate(
            layer_id=3,
            neuron_operation=ArchitectureNeuronsOpType.FREEZE
        )
        m(dummy_input)
    # - To test on The TinyUnet3p example
    # ADD
    # - layer_id = Base layer (Conv_out); same layer (eg batchnorm);
    # multi-inputs layers (eg. tinyUnet3p); recursive layers (eg. tinyUnet3p))
    # - neuron_indices = {-1, -2, -7, -19, -12} on a layer with 4 neurons - eq.
    with model as m:
        m.operate(
            layer_id=3,
            neuron_operation=ArchitectureNeuronsOpType.FREEZE
        )
        m(dummy_input)
    with model as m:
        m.operate(3, 4, neuron_operation=ArchitectureNeuronsOpType.FREEZE)
        m(dummy_input)
    # ADD 2 neurons
    with model as m:
        m.operate(1, {-1, -2}, neuron_operation=1)
        m(dummy_input)
    # PRUNE 1 neurons
    with model as m:
        m.operate(1, 2, neuron_operation=2)
        m(dummy_input)
    # PRUNE 3 neurons
    with model as m:
        m.operate(3, {-1, -2, 1}, neuron_operation=2)
        m(dummy_input)
