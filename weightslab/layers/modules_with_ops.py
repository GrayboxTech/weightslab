import hashlib
import collections
import torch as th

from torch import nn
from enum import Enum
from copy import deepcopy
from typing import List, Set

from weightslab.utils.logs import print
from weightslab.components.tracking import Tracker
from weightslab.components.tracking import TrackingMode
from weightslab.components.tracking import TriggersTracker
from weightslab.layers.neuron_ops import NeuronWiseOperations
from weightslab.utils.tools import \
    reindex_and_compress_blocks, \
    what_layer_type, \
    get_shape_attribute_from_module
from weightslab.components.tracking import copy_forward_tracked_attrs
from weightslab.models.model_with_ops import ArchitectureNeuronsOpType


class LayerWiseOperations(NeuronWiseOperations):
    """
        Base class for the complementary operations needed in order to
        implement the neuron wise operations correctly.
    """
    object_counter: int = 0

    def __init__(
            self,
            in_neurons: int,
            out_neurons: int,
            device: str = 'cpu',
            module_name: str = "module",
            super_in_name: str = "in_features",
            super_out_name: str = "out_features"
    ) -> None:

        # Init module ids
        self.assign_id()  # assign ids

        # Variables
        self.id = LayerWiseOperations.object_counter
        LayerWiseOperations.object_counter += 1
        self.out_neurons = out_neurons
        self.in_neurons = in_neurons
        self.device = device
        self.tracking_mode = TrackingMode.DISABLED
        self.module_name = module_name
        self.super_in_name = super_in_name
        self.super_out_name = super_out_name
        self.src_to_dst_mapping_tnsrs = {}
        self.dst_to_src_mapping_tnsrs = {}
        self.parents_src_to_dst_mapping_tnsrs = {}
        if hasattr(self, 'named_parameters'):
            self.learnable_tensors_name = [
                name for name, param in self.named_parameters()
                if param.requires_grad
            ]  # Get every learnable tensors name
            self.neuron_2_lr = {
                tensor_name:
                    collections.defaultdict(lambda: 1.0) for tensor_name in
                    self.learnable_tensors_name}
            self.incoming_neuron_2_lr = {
                'weight': collections.defaultdict(lambda: 1.0)
            }

        # Tracking
        self.regiser_trackers()

        # Register hooks
        self.register_grad_hook()

    def get_name(self):
        return self.module_name

    def set_name(self, n):
        if isinstance(n, str):
            self.module_name = n

    # =======================================================
    # Getter and Setter for in_neurons and out_neurons
    # =======================================================
    def get_neurons_value(self, v):
        value = v \
            if isinstance(v, int) else \
            (
                v[1] if len(v) == 4 else
                v[0]
            )
        return value

    def get_neurons(self, attr_name: str):
        """Getter: Returns the value of in_neurons."""
        if not hasattr(self, attr_name):
            raise AttributeError(
                "Accessing 'in_neurons' before calling" +
                "_initialize_neuron_attributes."
            )
        return self.get_neurons_value(getattr(self, attr_name))

    def set_neurons(self, attr_name: str, new_value: int | List[int] | th.Size | tuple):
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
        return self.update_attr(
            attr_name,
            attr_value
        )

    # ------------------------------
    # Magic OPS
    def __eq__(self, other) -> bool:
        _weight = th.allclose(self.weight.data, other.weight.data)
        _bias = th.allclose(self.bias.data, other.bias.data)
        _train_tracker = self.train_dataset_tracker == \
            other.train_dataset_tracker
        _eval_tracker = self.eval_dataset_tracker == \
            other.eval_dataset_tracker
        return _weight and _bias and _train_tracker and _eval_tracker

    def __hash__(self):
        # get all related learnable instance attributes,
        # e.g.,self.in_features, self.out_features, and self.bias
        params = (
            self.__dict__
        )
        return int(hashlib.sha256(str(params).encode()).hexdigest(), 16)

    # ------------------------------
    # Trackers Functions
    def regiser_trackers(self):
        self.register_module('train_dataset_tracker', TriggersTracker(
            self.get_neurons('out_neurons'), device=self.device))
        self.register_module('eval_dataset_tracker', TriggersTracker(
            self.get_neurons('out_neurons'), device=self.device))

    def reset_stats(self):
        """Reset stats for the trackers."""
        self.train_dataset_tracker.reset_stats()
        self.eval_dataset_tracker.reset_stats()

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
    def get_operation(self, op_type, **_):
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
            is_incoming: bool,
            tensor_name: str
    ) -> float:
        """
        Get the learning rate for a specific neuron.

        Args:
            neuron_id (int): The neuron id to get the learning rate for.

        Returns:
            float: The learning rate for the specific neuron.
        """
        if isinstance(neurons_id, set):
            neurons_id = [i for i in list(neurons_id)]

        neuron_2_lr = self.neuron_2_lr if not is_incoming else \
            self.incoming_neuron_2_lr

        if not neuron_2_lr or \
                tensor_name not in neuron_2_lr:
            return [1.0]*len(neurons_id)
        return [
            neuron_2_lr[tensor_name][neuron_id] for neuron_id in
            neurons_id
        ]

    def set_per_neuron_learning_rate(
            self,
            neurons_id: Set[int],
            neurons_lr: Set[float],
            tensor_name: str,
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
                f'Layer={self.get_name()}[id={self.id}]:'
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
        # This is meant to be called in the children classes.
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
                if tensor_name in self.incoming_neuron_2_lr:
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

    def find_value_for_key_pattern_(self, key_pattern, state_dict):
        for key, value in state_dict.items():
            if key_pattern in key:
                return value
        return None

    def get_parameter_count(self):
        """Compute the rough number of parameters in the layer.

        Returns:
            int: The number of parameters in the layer.
        """
        parameters = 1
        for i in range(len(self.weight.shape)):
            parameters *= self.weight.shape[i]
        if self.bias:
            parameters += self.bias.shape[0]
        return parameters

    # ---------------
    # Torch Functions
    def load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs
    ):
        tnsr = self.find_value_for_key_pattern_('weight', state_dict)
        if tnsr is not None:
            in_size, out_size = tnsr.shape[1], tnsr.shape[0]
            with th.no_grad():
                wshape = (out_size, in_size)
                self.weight.data = nn.Parameter(
                    th.ones(wshape)).to(self.device)
                if self.bias is not None:
                    self.bias.data = nn.Parameter(
                        th.ones(out_size)).to(self.device)

            self.update_attr(self.super_in_name, in_size)
            self.set_neurons(attr_name='in_neurons', new_value=in_size)
            self.update_attr(self.super_out_name, out_size)
            self.set_neurons(attr_name='out_neurons', new_value=out_size)
        self._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def to(self, *args, **kwargs):
        self.device = args[0]
        for tracker in self.get_trackers():
            tracker.to(*args, **kwargs)

    def register(
            self,
            activation_map: th.Tensor):
        tracker = self.get_tracker()
        if tracker is None or activation_map is None or input is None:
            return
        activation_map = (activation_map > 0).long()  # bool to int
        processed_activation_map = th.sum(activation_map, dim=(-2, -1))
        copy_forward_tracked_attrs(processed_activation_map, activation_map)
        tracker.update(processed_activation_map)

    def perform_layer_op(self,
                         activation_map: th.Tensor,
                         data: th.Tensor,
                         skip_register: bool = False,
                         intermediary: dict | None = None):
        copy_forward_tracked_attrs(activation_map, data)
        if not skip_register:
            self.register(activation_map)
        if intermediary is not None and self.get_module_id() in intermediary:
            try:
                intermediary[self.get_module_id()] = activation_map
            except Exception as e:
                print(
                    f"Error {e} occurred while updating intermediary outputs",
                    self.get_module_id(), str(activation_map)[:50],
                    level='ERROR')
        return activation_map

    def update_attr(self, attribute_name, value):
        # Get the current value using getattr()
        if not hasattr(self, attribute_name):
            return

        # Set the new value back using setattr()
        setattr(self, attribute_name, value)

        return value

    def is_flatten_layer(self):
        """
        Flatten dimensions are based on the shape of the layer.
        What is a flatten layer ? If every layers concerned have weights, and
        so tensor with shape, they must match (B, C, H, W), or (B, H, W, C).
        The B is important and here everytime. So we base our analysis on the
        tensor shape.
        """
        if hasattr(self, 'weight'):
            shape = self.weight.shape[1:]  # Remove the batch shape
            return len(shape) == 1
        return False

    def _process_neurons_indices(
            self,
            neuron_indices: Set[int] | int,
            is_incoming: bool = False,
            current_child_name: str = None,
            current_parent_name: str = None,
            **_
    ):
        """
        Intelligently processes high-level logical indices (like channels or
        neurons) into the flat, absolute tensor indices required for pruning.

        This function handles:
        1. 1-to-1 Mappings (Linear -> Linear, Conv -> Conv)
        2. N-to-1 Mappings (Conv(N) -> Flatten -> Linear(N*H*W))
        3. 1-to-N Mappings (Linear(N*H*W) -> Unflatten -> Conv(N))
        """
        if not isinstance(neuron_indices, set):
            neuron_indices = {neuron_indices}
        if not len(neuron_indices) or neuron_indices is None:
            neuron_indices = {-1}

        def reversing_indices(n_neurons, indices_set):
            """Your provided function to normalize indices."""
            return sorted(
                {
                    neg_idx for i in indices_set
                    if -n_neurons <= (
                        neg_idx :=
                        (i if i < 0 else -(n_neurons - i))
                    ) <= -1
                }
            )[::-1]

        if current_parent_name is not None and current_parent_name in \
                self.dst_to_src_mapping_tnsrs and is_incoming:
            mapped_indexs = self.dst_to_src_mapping_tnsrs[current_parent_name]
        elif len(list(self.src_to_dst_mapping_tnsrs.keys())):
            mapped_indexs = self.src_to_dst_mapping_tnsrs[
                current_child_name if current_child_name is not None else
                list(self.src_to_dst_mapping_tnsrs.keys())[0]
            ]
        else:
            mapped_indexs = self.get_neurons(attr_name='out_neurons') if not is_incoming else \
                self.get_neurons(attr_name='in_neurons')
            mapped_indexs = {i: [i] for i in range(mapped_indexs)}

        # Reverse index to last first, i.e., -1, -3, -5, ..etc
        reversed_indexs = reversing_indices(
            len(mapped_indexs),
            neuron_indices  # Ensure it's a set
        )
        original_indexs = list(
            len(mapped_indexs)+i for i in reversed_indexs
        )
        if hasattr(self, 'bypass'):
            offset = min(list(mapped_indexs.keys()))
            flat_indexs = list(
                mapped_indexs[len(mapped_indexs)+i+offset][::-1] for i in reversed_indexs
            )
        elif not hasattr(self, 'src_bypass') or is_incoming:
            flat_indexs = list(
                mapped_indexs[len(mapped_indexs)+i][::-1] for i in reversed_indexs
            )
        else:
            flat_indexs = list(
                len(mapped_indexs)+i for i in reversed_indexs
            )  # chose only mapped indexs if bypass flag

        # Return the final set of flat indices, sorted last-to-first as in
        # your original
        return flat_indexs, original_indexs

    def get_canal_length(self, incoming=False):
        if incoming:
            dst_to_src_keys = list(self.dst_to_src_mapping_tnsrs.keys())
            if not len(dst_to_src_keys):
                return 1
            last_channel = self.dst_to_src_mapping_tnsrs[dst_to_src_keys[0]][0]
            return len(last_channel) if isinstance(last_channel, list) else 1
        else:
            src_to_dst_keys = list(self.src_to_dst_mapping_tnsrs.keys())
            if not len(src_to_dst_keys):
                return 1
            last_channel = self.src_to_dst_mapping_tnsrs[src_to_dst_keys[0]][0]
            return len(last_channel) if isinstance(last_channel, list) else 1

    def operate(
        self,
        neuron_indices: int | Set[int],
        is_incoming: bool = False,
        skip_initialization: bool = False,
        neuron_operation: Enum = ArchitectureNeuronsOpType.ADD,
        **kwargs
    ):
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

    # ---------------
    # Neurons Operations
    def _add_neurons(
        self,
        neuron_indices: Set[int] | int = -1,
        is_incoming: bool = False,
        skip_initialization: bool = False,
        **kwargs
    ):
        if not isinstance(neuron_indices, set):
            if isinstance(neuron_indices, list):
                neuron_indices = set(neuron_indices)
            else:
                neuron_indices = set(neuron_indices) if not \
                    isinstance(neuron_indices, int) else set([neuron_indices])
        print(
            f"{self.get_name()}[{self.get_module_id()}].add {neuron_indices}",
            level='DEBUG'
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
        nb_neurons = self.get_canal_length(is_incoming) if \
            len(neuron_indices) == 1 else len(neuron_indices)
        if is_incoming == transposed:
            tensors = (nb_neurons, in_out_neurons // group_size)
        else:
            tensors = (in_out_neurons, nb_neurons // group_size)

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

        # Update neurons count & indexing
        # nb_neurons = nb_neurons * group_size
        if not is_incoming:
            # Update
            new_value = self.set_neurons(
                self.super_out_name,
                self.get_neurons(self.super_out_name) + nb_neurons
            )
            self.set_neurons(
                attr_name='out_neurons',
                new_value=new_value
            )
            # Add indexs from indexs map
            for parent_name in self.src_to_dst_mapping_tnsrs:
                key_index = int(
                    (
                        self.get_neurons(self.super_out_name) - nb_neurons
                    ) / nb_neurons - 1
                )
                self.src_to_dst_mapping_tnsrs[
                    parent_name
                ][key_index + 1] = [
                    i + max(
                        self.src_to_dst_mapping_tnsrs[
                            parent_name
                        ][key_index]
                    ) for i in range(1, nb_neurons + 1)
                ]
            # Tracking
            for tracker in self.get_trackers():
                tracker.add_neurons(nb_neurons)
            print(f'New layer is {self}', level='DEBUG')
        else:
            # Update
            new_value = self.set_neurons(
                self.super_in_name,
                self.get_neurons(self.super_in_name) + nb_neurons
            )
            self.set_neurons(
                attr_name='in_neurons',
                new_value=new_value
            )
            # Add indexs from indexs map
            min_p = 0
            current_parent_name = kwargs.get('current_parent_name', [])
            for parent_name in self.dst_to_src_mapping_tnsrs:
                # if dict has several items that are bypass of the module,
                # we split the new neurons between the two inputs channels
                # from the two input tensors, and so re index every neurons
                if min_p > 0:
                    tmp_ = deepcopy(self.dst_to_src_mapping_tnsrs[parent_name])
                    self.dst_to_src_mapping_tnsrs[parent_name].clear()
                    for k, v in tmp_.items():
                        self.dst_to_src_mapping_tnsrs[parent_name][k+min_p] = v
                if hasattr(self, 'bypass') and parent_name in \
                        current_parent_name:
                    key_index = max(
                        list(
                            self.dst_to_src_mapping_tnsrs[
                                parent_name
                            ].keys()
                        )
                    )
                    self.dst_to_src_mapping_tnsrs[
                        parent_name
                    ][key_index + 1] = [
                        i + 1 + max(
                            self.dst_to_src_mapping_tnsrs[
                                parent_name
                            ][key_index]
                        ) for i in range(0, nb_neurons)
                    ]
                    if hasattr(self, 'bypass'):
                        min_p = nb_neurons
            print(f'New INCOMING layer is {self}', level='DEBUG')

    def _prune_neurons(
        self,
        original_neuron_indices: Set[int],
        neuron_indices: List | Set[int],
        is_incoming: bool = False,
        **kwargs
    ):
        if not isinstance(neuron_indices, set):
            if isinstance(neuron_indices, list):
                neuron_indices = set(neuron_indices)
            else:
                neuron_indices = set([neuron_indices])
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

        # Generate idx to keep
        idx_tokeep = neurons - neuron_indices
        idx_tnsr = th.unique(
            th.Tensor(list(idx_tokeep)).long() // group_size
        ).to(self.device)

        # Operate on learnable tensor
        if hasattr(self, 'weight') and self.weight is not None:
            # Tensors modification
            with th.no_grad():
                self.weight.data = nn.Parameter(
                    th.index_select(
                        self.weight.data,
                        dim=(transposed ^ is_incoming),
                        index=idx_tnsr
                    )).to(self.device)

                if hasattr(self, 'bias') and self.bias is not None:
                    self.bias.data = nn.Parameter(
                        th.index_select(
                            self.bias.data,
                            dim=0,
                            index=idx_tnsr
                        )).to(self.device) if not is_incoming else \
                            self.bias.data
                if hasattr(self, 'running_mean'):
                    self.running_mean = th.index_select(
                        self.running_mean, dim=0, index=idx_tnsr).to(self.device)
                if hasattr(self, 'running_var'):
                    self.running_var = th.index_select(
                        self.running_var, dim=0, index=idx_tnsr).to(self.device)

        # Update neurons count and indexing
        if not is_incoming:
            # Update module attribute
            new_value = self.set_neurons(
                self.super_out_name,
                len(idx_tokeep)
            )
            self.set_neurons(
                attr_name='out_neurons',
                new_value=new_value
            )
            # Remove indexs from indexs map
            for child_name in self.src_to_dst_mapping_tnsrs:
                channel_size = 1  # Identical throught tensor
                for index in list(neuron_indices)[::-1]:
                    channel_size = len(self.src_to_dst_mapping_tnsrs[
                        child_name
                    ].pop(index))  # Remove indexs from all childs as its a src
                # Re-index every neurons
                self.src_to_dst_mapping_tnsrs[child_name] = \
                    reindex_and_compress_blocks(
                        self.src_to_dst_mapping_tnsrs[
                            child_name
                        ],
                        channel_size
                    )
            # Tracker
            for tracker in self.get_trackers():
                tracker.prune(neuron_indices)
            print(f'New layer is {self}', level='DEBUG')
        else:
            # Update module attribute
            new_value = self.set_neurons(
                self.super_in_name,
                len(idx_tokeep)
            )
            self.set_neurons(
                attr_name='in_neurons',
                new_value=new_value
            )
            # Remove indexs from indexs map
            min_p = 0
            current_parent_name = kwargs.get('current_parent_name', [])
            for parent_name in self.dst_to_src_mapping_tnsrs:
                # if dict has several items that are bypass of the module,
                # we split the new neurons between the two inputs channels
                # from the two input tensors, and so re index every neurons
                if min_p > 0:
                    tmp_ = deepcopy(self.dst_to_src_mapping_tnsrs[parent_name])
                    self.dst_to_src_mapping_tnsrs[parent_name].clear()
                    for k, v in tmp_.items():
                        self.dst_to_src_mapping_tnsrs[parent_name][k-min_p] = v

                if not hasattr(self, 'bypass') or (hasattr(self, 'bypass')
                                                   and parent_name in
                                                   current_parent_name):
                    channel_size = 1  # Identical throught tensor
                    for index in neuron_indices:  # [original_neuron_indices]:
                        channel_size = len(self.dst_to_src_mapping_tnsrs[
                            parent_name
                        ].pop(index))
                    # Re-index every neurons - no use to keep active pointer
                    # here as it's dst2src
                    indexs = self.dst_to_src_mapping_tnsrs[
                        parent_name
                    ]
                    offset_index = min(
                        indexs.keys()
                    ) if len(indexs) else 0
                    self.dst_to_src_mapping_tnsrs[parent_name] = \
                        reindex_and_compress_blocks(
                            self.dst_to_src_mapping_tnsrs[parent_name],
                            channel_size,
                            offset_index=offset_index
                        )
                    if hasattr(self, 'bypass'):
                        min_p = channel_size
            print(f'New INCOMING layer is {self}', level='DEBUG')

    def _freeze_neurons(
        self,
        neuron_indices: int | Set[int],
        is_incoming: bool = False,
        **_
    ):
        if isinstance(neuron_indices, int):
            neuron_indices = [neuron_indices]
        if isinstance(neuron_indices, set):
            neuron_indices = list(neuron_indices)
            if isinstance(neuron_indices[0], list):
                neuron_indices = [i for j in neuron_indices for i in j]

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
                neuron_indices[neuron_indice]:
                    1.0 - neuron_lr for neuron_indice, neuron_lr in enumerate(
                        self.get_per_neuron_learning_rate(
                            neuron_indices,
                            is_incoming=is_incoming,
                            tensor_name=tensor_name
                        )
                    )
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
        if isinstance(neuron_indices, int):
            neuron_indices = set([neuron_indices])
        if isinstance(neuron_indices, (list, set)):
            if isinstance(neuron_indices, set):
                neuron_indices = list(neuron_indices)
                if isinstance(neuron_indices[0], int):
                    neuron_indices = [i for i in neuron_indices]
                else:
                    neuron_indices = set(neuron_indices)

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
        neurons = set(range(self.get_neurons(attr_name='out_neurons') if not is_incoming else \
            self.get_neurons(attr_name='in_neurons')))
        if not set(neuron_indices) & neurons:
            raise ValueError(
                f"{self.get_name()}.reset neuron_indices and neurons set dont"
                f"overlapp: {neuron_indices} & {neurons} => "
                f"{set(neuron_indices) & neurons}")
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
                    tensors = (out_in_neurons,)  # if (not is_incoming and transposed) or (is_incoming and not transposed) or (not is_incoming and not transposed) else
                            #    in_out_neurons,)
                    neuron_weights = th.zeros(
                        tensors + (*self.kernel_size,)
                    ).to(self.device)

                # # Handle 1-dims cases like batchnorm without in out mapping
                elif len(self.weight.data.shape) == 1:
                    if hasattr(self, 'running_var') and \
                            hasattr(self, 'running_mean'):
                        norm = True
                    tensors = (out_in_neurons,)  # if (not is_incoming and transposed) or (is_incoming and not transposed) or (not is_incoming and not transposed) else
                            #    in_out_neurons,)
                    neuron_weights = th.ones(tensors).to(self.device) if not \
                        norm else 0.

                # # Handle 1-dims cases like linear, where we have a in out
                # # mapping (similar to conv1d wo. kernel)
                else:
                    tensors = (out_in_neurons,)  # if (not is_incoming and transposed) or (is_incoming and not transposed)  or (not is_incoming and not transposed) else
                            #    in_out_neurons,)
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
        if not is_incoming:
            for tracker in self.get_trackers():
                tracker.reset(neuron_indices)


if __name__ == "__main__":
    from weightslab.backend.watcher_editor import WatcherEditor
    from weightslab.tests.torch_models import FashionCNN as Model

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
