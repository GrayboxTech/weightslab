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
from weightslab.utils.tools import reindex_and_compress_blocks
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
            device,
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
        self.neuron_2_learning_rate = collections.defaultdict(lambda: 1.0)
        self.incoming_neuron_2_lr = collections.defaultdict(lambda: 1.0)
        self.module_name = module_name
        self.super_in_name = super_in_name
        self.super_out_name = super_out_name
        self.src_to_dst_mapping_tnsrs = {}
        self.dst_to_src_mapping_tnsrs = {}
        self.parents_src_to_dst_mapping_tnsrs = {}
        self.learnable_tensors_name = [
            name for name, param in self.named_parameters()
            if param.requires_grad
        ]  # Get every learnable tensors name

        # Tracking
        self.regiser_trackers()

        # Register hooks
        self.register_grad_hook()

    def get_name(self):
        return self.module_name

    def set_name(self, n):
        if isinstance(n, str):
            self.module_name = n

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
            self.out_neurons, device=self.device))
        self.register_module('eval_dataset_tracker', TriggersTracker(
            self.out_neurons, device=self.device))

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

    def get_per_neuron_learning_rate(self, neurons_id: int) -> float:
        """
        Get the learning rate for a specific neuron.

        Args:
            neuron_id (int): The neuron id to get the learning rate for.

        Returns:
            float: The learning rate for the specific neuron.
        """
        if isinstance(neurons_id, (list, set)) and isinstance(neurons_id[0], (list, set)):
            neurons_id = [i for i in neurons_id]

        if not self.neuron_2_learning_rate:
            return [1.0]*len(neurons_id)
        return [self.neuron_2_learning_rate[neuron_id] for neuron_id in neurons_id]

    def set_per_neuron_learning_rate(self, neurons_id: Set[int], neurons_lr: Set[float], incoming_neurons: bool = False):
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
        in_out_neurons = self.out_neurons if not incoming_neurons else \
            self.in_neurons
        neurons_2_lr = self.neuron_2_learning_rate if not incoming_neurons else self.incoming_neuron_2_lr

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
                raise ValueError('Cannot set learning rate outside [0, 1] range')
            neurons_2_lr[neuron_id] = lr

    def register_grad_hook(self):
        # This is meant to be called in the children classes.
        def weight_grad_hook(weight_grad):
            for neuron_id, neuron_lr in self.neuron_2_learning_rate.items():
                if neuron_id >= weight_grad.shape[0]:
                    continue
                neuron_grad = weight_grad[neuron_id]
                neuron_grad *= neuron_lr
                weight_grad[neuron_id] = neuron_grad

            for in_neuron_id, neuron_lr in self.incoming_neuron_2_lr.items():
                if in_neuron_id >= weight_grad.shape[1]:
                    continue
                in_neuron_grad = weight_grad[:, in_neuron_id]
                in_neuron_grad *= neuron_lr
                weight_grad[:, in_neuron_id] = in_neuron_grad
            return weight_grad

        def oneD_grad_hook(bias_grad):
            for neuron_id, neuron_lr in self.neuron_2_learning_rate.items():
                if neuron_id >= bias_grad.shape[0]:
                    continue
                neuron_grad = bias_grad[neuron_id]
                neuron_grad *= neuron_lr
                bias_grad[neuron_id] = neuron_grad
            return bias_grad

        for tensor_name in self.learnable_tensors_name:
            if tensor_name == 'weight' and \
                    hasattr(self, 'weight') and \
                    self.weight is not None:
                self.weight.register_hook(weight_grad_hook)
            else:
                if hasattr(self, 'tensor_name') and getattr(self, tensor_name) is not None:
                    getattr(self, tensor_name).register_hook(oneD_grad_hook)

    def _find_value_for_key_pattern(self, key_pattern, state_dict):
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
    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
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

            self.update_attr(self.super_in_name, in_size)
            self.in_neurons = in_size
            self.update_attr(self.super_out_name, out_size)
            self.out_neurons = out_size
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

    def _process_neurons_index(
            self,
            index_neurons: set,
            incoming_neurons: bool = False,
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

        def reversing_index(n_neurons, indices_set):
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
                self.dst_to_src_mapping_tnsrs and incoming_neurons:
            mapped_indexs = self.dst_to_src_mapping_tnsrs[current_parent_name]
        elif current_child_name is not None and current_child_name in \
                self.src_to_dst_mapping_tnsrs and not incoming_neurons:
            mapped_indexs = self.src_to_dst_mapping_tnsrs[current_child_name]
        else:
            mapped_indexs = self.out_neurons if not incoming_neurons else \
                self.in_neurons
            mapped_indexs = {i: [i] for i in range(mapped_indexs)}

        # Reverse index to last first, i.e., -1, -3, -5, ..etc
        reversed_indexs = reversing_index(
            len(mapped_indexs),
            index_neurons  # Ensure it's a set
        )
        original_indexs = list(
            len(mapped_indexs)+i for i in reversed_indexs
        )
        flat_indexs = list(
            mapped_indexs[len(mapped_indexs)+i][::-1] for i in reversed_indexs
        ) if hasattr(self, 'bypass') else list(
            len(mapped_indexs)+i for i in reversed_indexs
        )

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
        index_neurons: int | Set[int],
        incoming_neurons: bool = False,
        skip_initialization: bool = False,
        neurons_operation: Enum = ArchitectureNeuronsOpType.ADD,
        **kwargs
    ):
        # Get Operation
        op = self.get_operation(neurons_operation)

        # Get Neurons Indexs Formatted
        index_neurons, original_index_neurons = self._process_neurons_index(
            index_neurons,
            incoming_neurons=incoming_neurons,
            **kwargs
        )

        # Operate
        for ind in range(len(index_neurons)):
            op(
                original_index_neurons=original_index_neurons[ind],
                index_neurons=index_neurons[ind],
                incoming_neurons=incoming_neurons,
                skip_initialization=skip_initialization,
                **kwargs
            )

    # ---------------
    # Neurons Operations
    def _prune_neurons(
        self,
        original_index_neurons: Set[int],
        index_neurons: List | Set[int],
        incoming_neurons: bool = False,
        **kwargs
    ):
        if not isinstance(index_neurons, set):
            if isinstance(index_neurons, list):
                index_neurons = set(index_neurons)
            else:
                index_neurons = set([index_neurons])
        # Check if it's a transposed layer
        transposed = int('transpose' in self.get_name().lower())

        # Get the number of corresponding layer weights
        in_out_neurons = self.out_neurons if not incoming_neurons else \
            self.in_neurons

        # Get current weights indexs
        neurons = set(range(in_out_neurons))

        # Sanity check
        if not set(index_neurons) & neurons:
            raise ValueError(
                f"{self.get_name()}.prune indices and neurons set do not "
                f"overlapp: {index_neurons} & {neurons} => \
                    {index_neurons & neurons}")

        # Generate idx to keep
        idx_tnsr = th.tensor(
            list(neurons - index_neurons)
        ).to(self.device)

        with th.no_grad():
            self.weight.data = nn.Parameter(
                th.index_select(
                    self.weight.data,
                    dim=(transposed ^ incoming_neurons)
                    & int(len(self.weight.data) > 1),
                    index=idx_tnsr
                )).to(self.device)
            if hasattr(self, 'bias'):
                self.bias.data = nn.Parameter(
                    th.index_select(
                        self.bias.data,
                        dim=0,
                        index=idx_tnsr
                    )).to(self.device) if not incoming_neurons else \
                        self.bias.data
            if hasattr(self, 'running_mean'):
                self.running_mean = th.index_select(
                    self.running_mean, dim=0, index=idx_tnsr).to(self.device)
            if hasattr(self, 'running_var'):
                self.running_var = th.index_select(
                    self.running_var, dim=0, index=idx_tnsr).to(self.device)

        # Update
        if not incoming_neurons:
            # Remove indexs from indexs map
            for child_name in self.src_to_dst_mapping_tnsrs:
                channel_size = 1  # Identical throught tensor
                for index in list(index_neurons)[::-1]:
                    channel_size = len(self.src_to_dst_mapping_tnsrs[
                        child_name
                    ].pop(index))  # Remove indexs from all childs as its a src
                # Re-index every neurons
                self.src_to_dst_mapping_tnsrs[child_name] = reindex_and_compress_blocks(self.src_to_dst_mapping_tnsrs[child_name], channel_size)
            # Update module attribute
            self.out_neurons = self.update_attr(
                self.super_out_name,
                len(idx_tnsr)
            )
            # Tracker
            for tracker in self.get_trackers():
                tracker.prune(index_neurons)
            print(f'New layer is {self}', level='DEBUG')
        else:
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

                if not hasattr(self, 'bypass') or (hasattr(self, 'bypass') and parent_name in current_parent_name):
                    channel_size = 1  # Identical throught tensor
                    for index in [original_index_neurons]:
                        min_index = min(self.dst_to_src_mapping_tnsrs[
                                parent_name
                            ].keys()
                        )
                        channel_size = len(self.dst_to_src_mapping_tnsrs[
                            parent_name
                        ].pop(min_index + index))
                    # Re-index every neurons - no use to keep active pointer here as it's dst2src
                    self.dst_to_src_mapping_tnsrs[parent_name] = reindex_and_compress_blocks(self.dst_to_src_mapping_tnsrs[parent_name], channel_size)
                    if hasattr(self, 'bypass'):
                        min_p = channel_size
            # Update module attribute
            self.in_neurons = self.update_attr(
                self.super_in_name,
                len(idx_tnsr)
            )
            print(f'New INCOMING layer is {self}', level='DEBUG')

    def _reset_neurons(
        self,
        index_neurons: int | Set[int],
        incoming_neurons: bool = False,
        skip_initialization: bool = False,
        perturbation_ratio: float | None = None,
        **_
    ):
        if isinstance(index_neurons, int):
            index_neurons = set([index_neurons])
        if isinstance(index_neurons, (list, set)):
            if isinstance(index_neurons, set):
                index_neurons = list(index_neurons)
                if isinstance(index_neurons[0], int):
                    index_neurons = [i for i in index_neurons]
                else:
                    index_neurons = set(index_neurons)

        # Manage specific usecases
        # # Incoming Layer
        in_out_neurons = self.out_neurons if not incoming_neurons else \
            self.in_neurons
        out_in_neurons = self.out_neurons if incoming_neurons else \
            self.in_neurons
        # # Transposed Layer
        transposed = int('transpose' in self.get_name().lower())
        # # Reset everything
        if index_neurons is None or not len(index_neurons):
            index_neurons = set(range(in_out_neurons))

        # Skip initialization is only to be able to test the function.
        neurons = set(range(in_out_neurons))
        if not set(index_neurons) & neurons:
            raise ValueError(
                f"{self.get_name()}.reset index_neurons and neurons set do not "
                f"overlapp: {index_neurons} & {neurons} => "
                f"{index_neurons & neurons}")
        if perturbation_ratio is not None and (
                0.0 >= perturbation_ratio or perturbation_ratio >= 1.0):
            raise ValueError(
                f"{self.get_name()}.reset perturbation "
                f"{perturbation_ratio} outside of [0.0, 1.0]")

        norm = False
        with th.no_grad():
            for index_neuron in index_neurons:
                tensors = (out_in_neurons,) if \
                    not incoming_neurons ^ transposed else \
                    (in_out_neurons,)

                # Weights
                # neuron_weights = th.zeros(self.in_channels, *self.kernel_size).to(
                #     self.device)
                # # Handle n-dims kernels like with conv{n}d
                if hasattr(self, "kernel_size") and self.kernel_size:
                    neuron_weights = th.zeros(
                        tensors + (*self.kernel_size,)
                    ).to(self.device)

                # # Handle 1-dims cases like batchnorm without in out mapping
                elif len(self.weight.data.shape) == 1:
                    if hasattr(self, 'running_var') and \
                            hasattr(self, 'running_mean'):
                        norm = True
                    neuron_weights = th.ones(tensors).to(self.device) if not \
                        norm else 0.

                # # Handle 1-dims cases like linear, where we have a in out
                # # mapping (similar to conv1d wo. kernel)
                else:
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
                        neuron_weights = self.weight[index_neuron] if not \
                            incoming_neurons else self.weight[:, index_neuron]
                        weights_perturbation = \
                            perturbation_ratio * neuron_weights * \
                            th.randint_like(neuron_weights, -1, 2).float()
                        neuron_weights += weights_perturbation

                        # bias
                        if not incoming_neurons:
                            neuron_bias = self.bias[index_neuron]
                            bias_perturbation = \
                                perturbation_ratio * neuron_bias * \
                                th.randint(-1, 2, (1, )).float().item()
                            neuron_bias += bias_perturbation
                if not norm:
                    if not incoming_neurons:
                        self.weight[index_neuron] = neuron_weights
                        self.bias[index_neuron] = neuron_bias
                    else:
                        self.weight[:, index_neuron] = neuron_weights
                else:
                    self.running_mean[index_neuron] = neuron_weights
                    self.running_var[index_neuron] = 1 - neuron_weights
                    self.weight[index_neuron] = neuron_weights
                    self.bias[index_neuron] = neuron_weights
        if not incoming_neurons:
            for tracker in self.get_trackers():
                tracker.reset(index_neurons)

    def _freeze_neurons(
        self,
        index_neurons: int | Set[int],
        incoming_neurons: bool = False,
        **_
    ):
        if isinstance(index_neurons, int):
            index_neurons = set([index_neurons])
        if isinstance(index_neurons, (list, set)):
            if isinstance(index_neurons, set):
                index_neurons = list(index_neurons)
                if isinstance(index_neurons[0], int):
                    index_neurons = [i for i in index_neurons]
                else:
                    index_neurons = set(index_neurons)

        # Neurons not specified - freeze everything
        if index_neurons is None or not len(index_neurons):
            index_neurons = set(range(self.out_neurons))

        neurons_lr = {index_neurons[index_neuron]: 1.0 - neuron_lr for index_neuron, neuron_lr in enumerate(self.get_per_neuron_learning_rate(index_neurons))}
        self.set_per_neuron_learning_rate(
            neurons_id=set(index_neurons), neurons_lr=neurons_lr, incoming_neurons=incoming_neurons
        )

    def _add_neurons(
        self,
        index_neurons: Set[int] = -1,
        incoming_neurons: bool = False,
        skip_initialization: bool = False,
        **kwargs
    ):
        if not isinstance(index_neurons, set):
            if isinstance(index_neurons, list):
                index_neurons = set(index_neurons)
            else:
                index_neurons = set([index_neurons])
        print(
            f"{self.get_name()}[{self.get_module_id()}].add {index_neurons}",
            level='DEBUG'
        )
        nb_neurons = self.get_canal_length(incoming_neurons) if len(index_neurons) == 1 else len(index_neurons)
        # Incoming operation or out operation; chose the right neurons
        norm = False
        # # TODO (GP): fix hardcoding transpose
        transposed = int('transpose' in self.get_name().lower())
        in_out_neurons = self.out_neurons if incoming_neurons else \
            self.in_neurons  # tuple (in_r, out_r)
        tensors = (nb_neurons, in_out_neurons) if \
            not incoming_neurons ^ transposed else \
            (in_out_neurons, nb_neurons)

        # Weights
        # # Handle n-dims kernels like with conv{n}d
        if hasattr(self, "kernel_size") and self.kernel_size:
            added_weights = th.zeros(
                tensors + (*self.kernel_size,)
            ).to(self.device)

        # # Handle 1-dims cases like batchnorm without in out mapping
        elif len(self.weight.data.shape) == 1:
            norm = True
            added_weights = th.ones(nb_neurons, ).to(self.device)

        # # Handle 1-dims cases like linear, where we have a in out mapping
        # # (similar to conv1d wo. kernel)
        else:
            added_weights = th.zeros(
                tensors
            ).to(self.device)

        # Biases
        if not incoming_neurons:
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
                        (self.weight.data, added_weights),
                        dim=(transposed ^ incoming_neurons) & int(len(self.weight.data.flatten()) > 1)
                    )
                ).to(self.device)

                if self.bias is not None and not incoming_neurons:
                    self.bias.data = nn.Parameter(
                        th.cat((self.bias.data, added_bias))
                    ).to(self.device)
        else:
            # Update
            with th.no_grad():
                self.weight.data = nn.Parameter(
                    th.cat((self.weight.data, added_weights))
                )
                if self.bias is not None:
                    self.bias.data = nn.Parameter(
                        th.cat((self.bias.data, added_bias))
                    )

                if hasattr(self, 'running_mean') and \
                        self.running_mean is not None:
                    self.running_mean = th.cat((
                        self.running_mean,
                        th.zeros(nb_neurons).to(self.device)))
                if hasattr(self, 'running_var') and \
                        self.running_var is not None:
                    self.running_var = th.cat((
                        self.running_var,
                        th.ones(nb_neurons).to(self.device))
                    )

        # Update
        if not incoming_neurons:
            # Update
            self.out_neurons = self.update_attr(
                self.super_out_name,
                getattr(self, self.super_out_name) + len(index_neurons)
            )
            print(f'New layer is {self}', level='DEBUG')
            # Add indexs from indexs map
            for parent_name in self.src_to_dst_mapping_tnsrs:
                key_index = int(
                    (
                        getattr(self, self.super_out_name) - nb_neurons
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
        else:
            # Update
            self.in_neurons = self.update_attr(
                self.super_in_name,
                getattr(self, self.super_in_name) + len(index_neurons)
            )
            # Add indexs from indexs map
            min_p = 0
            current_parent_name = kwargs.get('current_parent_name', [])
            for parent_name in self.dst_to_src_mapping_tnsrs:
                # key_index = int(
                #     (
                #         getattr(self, self.super_in_name) - nb_neurons
                #     ) / nb_neurons - 1
                # )
                # self.dst_to_src_mapping_tnsrs[
                #     parent_name
                # ][key_index + 1] = [
                #     i + max(
                #         self.dst_to_src_mapping_tnsrs[
                #             parent_name
                #         ][key_index]
                #     ) for i in range(1, nb_neurons+1)
                # ]
                #
                # if dict has several items that are bypass of the module,
                # we split the new neurons between the two inputs channels
                # from the two input tensors, and so re index every neurons
                if min_p > 0:
                    tmp_ = deepcopy(self.dst_to_src_mapping_tnsrs[parent_name])
                    self.dst_to_src_mapping_tnsrs[parent_name].clear()
                    for k, v in tmp_.items():
                        self.dst_to_src_mapping_tnsrs[parent_name][k+min_p] = v
                if hasattr(self, 'bypass') and parent_name in current_parent_name:
                    key_index = max(list(self.dst_to_src_mapping_tnsrs[parent_name].keys()))
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


if __name__ == "__main__":
    import torch.nn.functional as F
    from weightslab.backend.watcher_editor import WatcherEditor

    class UnusualModelTransposed(nn.Module):
        """
        Implements a hybrid model using Conv2d for downsampling (Encoder) and 
        ConvTranspose2d for upsampling (Decoder), replacing the final Linear/Conv2d.
        Input size assumed: 3 x 32 x 32.
        """

        # Encoder Constants (Output of conv2)
        ENCODER_CHANNELS = 32
        ENCODER_H = 16
        ENCODER_W = 16
        FLATTENED_SIZE = ENCODER_CHANNELS * ENCODER_H * ENCODER_W # 8192

        LATENT_SIZE = 256 # Size of the compact feature vector after linear1

        # Decoder Constants (Starting point for ConvTranspose)
        # The latent vector (256) is expanded to this size to form a small feature map
        DECODER_START_CHANNELS = 64
        DECODER_START_H = 4
        DECODER_START_W = 4
        DECODER_START_SIZE = DECODER_START_CHANNELS * DECODER_START_H * DECODER_START_W # 1024

        UNFLATTEN_SHAPE = (DECODER_START_CHANNELS, DECODER_START_H, DECODER_START_W)

        def __init__(self):
            super().__init__()

            # --- ENCODER PATH (Downsampling) ---
            # 1. Conv2d (3 -> 16 channels, 32x32 -> 32x32)
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            self.b1 = nn.BatchNorm2d(16)

            # 2. Conv2d (16 -> 32 channels, 32x32 -> 16x16, using stride for downsampling)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=self.ENCODER_CHANNELS, 
                                kernel_size=3, stride=2, padding=1)

            # 3. Flatten (N, 32, 16, 16) -> (N, 8192)
            self.flatten = nn.Flatten()

            # 4. Linear (Latent Layer)
            self.linear1 = nn.Linear(in_features=self.FLATTENED_SIZE, out_features=self.LATENT_SIZE)

            # --- DECODER PATH (Upsampling using ConvTranspose2d) ---

            # 5. Linear Upsample: Expand latent vector to match the 4D input size of the ConvTranspose layers
            self.linear_to_4d = nn.Linear(in_features=self.LATENT_SIZE, 
                                        out_features=self.DECODER_START_SIZE)
            self.unflatten = nn.Unflatten(dim=1, unflattened_size=self.UNFLATTEN_SHAPE)

            # 6. ConvTranspose2d 1: Upsample 4x4 -> 8x8
            # Output: 32 channels @ 8x8
            self.convt1 = nn.ConvTranspose2d(in_channels=self.DECODER_START_CHANNELS, 
                                            out_channels=32, kernel_size=4, stride=2, padding=1)
            self.b_t1 = nn.BatchNorm2d(32)

            # 7. ConvTranspose2d 2: Upsample 8x8 -> 16x16
            # Output: 16 channels @ 16x16
            self.convt2 = nn.ConvTranspose2d(in_channels=32,
                                            out_channels=16, kernel_size=4, stride=2, padding=1)
            self.b_t2 = nn.BatchNorm2d(16)

            # 8. ConvTranspose2d 3 (Final Layer): Upsample 16x16 -> 32x32
            # Output: 3 channels (original image depth) @ 32x32
            self.convt3 = nn.ConvTranspose2d(in_channels=16, 
                                            out_channels=3, kernel_size=4, stride=2, padding=1)

        def forward(self, x):
            N = x.size(0)  # Batch size

            # --- ENCODER FORWARD PASS ---

            # 1. Conv1 (32x32)
            x = F.relu(self.b1(self.conv1(x)))

            # 2. Conv2 (16x16)
            x = F.relu(self.conv2(x))

            # 3. Flatten (4D -> 1D vector)
            x = self.flatten(x)

            # 4. Linear1 (Down to Latent Space)
            x = F.relu(self.linear1(x))

            # --- DECODER FORWARD PASS ---

            # 5. Linear Upsample (Latent -> Decoder Start Size)
            x = F.relu(self.linear_to_4d(x))

            # 6. Reshape/Unflatten (1D vector -> 4D feature map)
            # (N, 1024) -> (N, 64, 4, 4)
            x = self.unflatten(x)

            # 7. ConvTranspose 1 (4x4 -> 8x8)
            x = F.relu(self.b_t1(self.convt1(x)))

            # 8. ConvTranspose 2 (8x8 -> 16x16)
            x = F.relu(self.b_t2(self.convt2(x)))

            # 9. ConvTranspose 3 (16x16 -> 32x32)
            # Use tanh or sigmoid for final image-like output, or ReLU/None
            # for features
            x = th.sigmoid(self.convt3(x))

            # Final output shape: (N, 3, 32, 32)
            return x

    # model = UnusualModelTransposed()
    from weightslab.tests.test_utils import TinyUNet_Straightforward
    model = TinyUNet_Straightforward()

    # 1. Input: 1 image, 3 channels, 32x32 spatial size
    dummy_input = th.randn(model.input_shape)

    # Run the forward pass
    output = model(dummy_input)

    # Watcher
    model = WatcherEditor(model, dummy_input=dummy_input, print_graph=False)
    model(dummy_input)
    print(model)
    nn_l = len(model.layers)-1
    # Neurons Operation
    # # Adding
    # # # # Case A.1: C0out to Bin -> Channel adding
    # with model as m:
    #     m.operate(nn_l, {-1}, neurons_operation=1)
    #     m(dummy_input)
    # # # # Case A.2: Bout to Cin -> Channel adding
    # with model as m:
    #     m.operate(1, {-3, -1}, neurons_operation=1)
    #     m(dummy_input)
    # # # Case B: C1out to Lin -> Channel adding & corr. Lin. neurons
    # with model as m:
    #     m.operate(2, {-2, -1}, neurons_operation=1)
    #     m(dummy_input)
    # # # # Case C: Lout to Lin -> Corresp. Neurons to add both side
    # with model as m:
    #     m.operate(3, {-145, -1}, neurons_operation=1)
    #     m(dummy_input)
    # # # # Pruning
    # # # # Case A.1: C0out to Bin -> Channel pruning
    # with model as m:
    #     m.operate(6, {-1}, neurons_operation=2)
    #     m(dummy_input)
    # # # # Case A.2: Bout to Cin -> Channel pruning
    # with model as m:
    #     m.operate(1, {-3, -1}, neurons_operation=2)
    #     m(dummy_input)
    # # # Case B: C1out to Lin -> Channel pruning & corr. Lin. neurons
    # with model as m:
    #     m.operate(2, {-2, -1}, neurons_operation=2)
    #     m(dummy_input)
    # # # # Case C: Lout to Lin -> Corresp. Neurons to prune
    # with model as m:
    #     m.operate(3, {-145, -1}, neurons_operation=2)
    #     m(dummy_input)
    # # # # # Freezing
    # # # # Case A.1: C0out to Bin -> Channel freezing
    # with model as m:
    #     m.operate(0, {-3, -1}, neurons_operation=3)
    #     m(dummy_input)
    # # # # Case A.2: Bout to Cin -> Channel freezing
    # with model as m:
    #     m.operate(1, {-3, -1}, neurons_operation=3)
    #     m(dummy_input)
    # # # Case B: C1out to Lin -> Channel freezing & corr. Lin. neurons
    # with model as m:
    #     m.operate(2, {-2, -1}, neurons_operation=3)
    #     m(dummy_input)
    # # # # Case C: Lout to Lin -> Corresp. Neurons to freeze
    # with model as m:
    #     m.operate(3, {-145, -1}, neurons_operation=3)
    #     m(dummy_input)
    # # # Reset ids
    # # # Case A.1: C0out to Bin to Cin -> Channel reseting
    with model as m:
        m.operate(0, {-3, -5}, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    # # # Case A.2: Bout to Cin -> Channel reseting
    with model as m:
        m.operate(1, {-3, -1}, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    # # Case B: C1out to Lin -> Channel reseting & corr. Lin. neurons
    with model as m:
        m.operate(2, {-2, -1}, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    # # # Case C: Lout to Lin -> Corresp. Neurons to reset
    with model as m:
        m.operate(3, {-145, -1}, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    # # # Reset every neurons
    # # # Case A.1: C0out to Bin to Cin -> Channel reseting
    with model as m:
        m.operate(0, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    # # # Case A.2: Bout to Cin -> Channel reseting
    with model as m:
        m.operate(1, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    # # Case B: C1out to Lin -> Channel reseting & corr. Lin. neurons
    with model as m:
        m.operate(2, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    # # # Case C: Lout to Lin -> Corresp. Neurons to reset
    with model as m:
        m.operate(3, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)

    # Transposed part
    # # # # Case A.1: C0out to Bin -> Channel adding
    with model as m:
        m.operate(5, {-1}, neurons_operation=1)
        m(dummy_input)
    # # # Case A.1: C0out to Bin -> Channel pruning
    with model as m:
        m.operate(5, {-3, -1}, neurons_operation=2)
        m(dummy_input)
    # # # Case A.1: C0out to Bin -> Channel freezing
    with model as m:
        m.operate(5, {-3, -1}, neurons_operation=3)
        m.operate(5, neurons_operation=3)
        m(dummy_input)
    # # # Case A.1: C0out to Bin to Cin -> Channel reseting
    with model as m:
        m.operate(5, neurons_operation=4)
        m.operate(5, perturbation_ratio=0.5, neurons_operation=4)
        m.operate(5, {-2}, neurons_operation=4)
        m.operate(5, {-2}, perturbation_ratio=0.5, neurons_operation=4)
        m(dummy_input)
    print('Bye World')
