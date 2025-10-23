import hashlib
import collections
import torch as th

from torch import nn
from typing import List, Set

from weightslab.utils.logs import print
from weightslab.components.tracking import TrackingMode
from weightslab.components.tracking import Tracker
from weightslab.components.tracking import TriggersTracker
from weightslab.components.tracking import copy_forward_tracked_attrs
from weightslab.layers.neuron_ops import NeuronWiseOperations


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

    def get_per_neuron_learning_rate(self, neuron_id: int) -> float:
        """
        Get the learning rate for a specific neuron.

        Args:
            neuron_id (int): The neuron id to get the learning rate for.

        Returns:
            float: The learning rate for the specific neuron.
        """
        if not self.neuron_2_learning_rate:
            return 1.0
        return self.neuron_2_learning_rate[neuron_id]

    def set_per_incoming_neuron_learning_rate(
            self, neuron_ids: Set[int], lr: float):
        """
        Set learning rate per incoming neuron.

        Args:
            neuron_ids (Set[int]):
                The set of incoming neurons to set the learning rate
            lr (float): The value of the learning rate. Can be between [0, 1]
        """

        if lr < 0 or lr > 1.0:
            raise ValueError('Cannot set learning rate outside [0, 1] range')

        invalid_ids = (neuron_ids - set(range(self.in_neurons)))
        if invalid_ids:
            raise ValueError(
                f'Layer[id={self.id}]:'
                f'Cannot set learning rate for neurons {invalid_ids} as they '
                f'are outside the set of existent neurons '
                f'{self.in_neurons}.'
            )

        for neuron_id in neuron_ids:
            self.incoming_neuron_2_lr[neuron_id] = lr

    def zerofy_connections_from(self, from_neuron_ids: Set[int], to_neuron_ids: Set[int]):
        if self.weight.ndim != 2:
            # Child classes (e.g. Conv2d) should override.
            return
        in_max = self.weight.shape[1]
        out_max = self.weight.shape[0]
        with th.no_grad():
            for to_id in to_neuron_ids:
                if to_id < 0 or to_id >= out_max:
                    continue
                for from_id in from_neuron_ids:
                    if from_id < 0 or from_id >= in_max:
                        continue
                    self.weight[to_id, from_id] = 0.0

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

        def bias_grad_hook(bias_grad):
            for neuron_id, neuron_lr in self.neuron_2_learning_rate.items():
                if neuron_id >= bias_grad.shape[0]:
                    continue
                neuron_grad = bias_grad[neuron_id]
                neuron_grad *= neuron_lr
                bias_grad[neuron_id] = neuron_grad
            return bias_grad

        if hasattr(self, 'weight') and self.weight is not None:
            self.weight.register_hook(weight_grad_hook)
        if hasattr(self, 'bias') and self.bias is not None:
            self.bias.register_hook(bias_grad_hook)

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
        # 1. Get the current value using getattr()
        if not hasattr(self, attribute_name):
            return

        # This will raise an AttributeError if the attribute_name is not found.
        current_value = getattr(self, attribute_name)

        # 2. Increment the value
        new_value = current_value + value

        # 3. Set the new value back using setattr()
        setattr(self, attribute_name, new_value)

    def add_neurons(self,
                    out_neurons: int,
                    skip_initialization: bool = False):
        print(
            f"{self.get_name()}[{self.get_module_id()}].add {out_neurons}",
            level='DEBUG'
        )
        device = self.weight.device  # get current device on which operate
        batchnorm = False

        # Weights
        # # Handle n-dims kernels like with conv{n}d
        if hasattr(self, "kernel_size") and self.kernel_size:
            added_weights = th.zeros(
                (out_neurons, self.in_neurons, *self.kernel_size)
            ).to(self.device)
        # # Handle 1-dims cases like batchnorm without in out mapping
        elif len(self.weight.data.shape) == 1:
            batchnorm = True
            added_weights = th.ones(out_neurons, ).to(device)

        # # Handle 1-dims cases like linear, where we have a in out mapping
        # # (similar to conv1d wo. kernel)
        else:
            added_weights = th.zeros(out_neurons, self.in_neurons).to(
                self.device)
        added_bias = th.zeros(out_neurons).to(device)

        if not batchnorm:
            # Initialization
            if not skip_initialization:
                nn.init.xavier_uniform_(added_weights,
                                        gain=nn.init.calculate_gain('relu'))

            # Update
            with th.no_grad():
                self.weight.data = nn.Parameter(
                    th.cat((self.weight.data, added_weights))).to(device)
                if self.bias is not None:
                    self.bias.data = nn.Parameter(
                        th.cat((self.bias.data, added_bias))).to(device)
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

                self.running_mean = th.cat((
                    self.running_mean,
                    th.zeros(out_neurons).to(self.running_mean.device)))
                self.running_var = th.cat((
                    self.running_var,
                    th.ones(out_neurons).to(self.running_var.device))
                )

        # Update
        self.update_attr(self.super_out_name, out_neurons)
        self.out_neurons = getattr(self, self.super_out_name)

        # Tracking
        for tracker in self.get_trackers():
            tracker.add_neurons(out_neurons)
        print(f'New layer is {self}', level='DEBUG')

    def add_incoming_neurons(
            self,
            out_neurons: int,
            skip_initialization: bool = True):
        print(f"{self.get_name()}[{self.get_module_id()}].add_incoming \
               {out_neurons}", level='DEBUG')
        device = self.weight.device  # get current device on which operate

        # Weights
        if hasattr(self, "kernel_size") and self.kernel_size:
            added_weights = th.zeros(
                (self.out_neurons, out_neurons, *self.kernel_size)
            ).to(device)
        else:
            added_weights = th.zeros(
                (self.out_neurons, out_neurons)
            ).to(device)

        # Initialization
        if not skip_initialization:
            nn.init.xavier_uniform_(added_weights,
                                    gain=nn.init.calculate_gain('relu'))

        # Update
        with th.no_grad():
            self.weight.data = nn.Parameter(
                th.cat((self.weight.data, added_weights), dim=1)
            ).to(device)

        self.update_attr(self.super_in_name, out_neurons)
        self.in_neurons = getattr(self, self.super_in_name)
        print(f'New INCOMING layer is {self}', level='DEBUG')
