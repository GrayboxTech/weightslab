""" Classes related to network architecture operations and internals. """
from typing import List, Set, Optional, Tuple
from torch import nn

import numpy as np
import torch as th

from weightslab.components.tracking import TrackingMode
from weightslab.utils.tools import get_children
from weightslab.utils.modules_dependencies import _ModulesDependencyManager, \
    DepType


class NetworkWithOps(nn.Module):
    def __init__(self):
        super(NetworkWithOps, self).__init__()
        self.seen_samples = 0
        self.tracking_mode = TrackingMode.DISABLED
        self._architecture_change_hook_fns = []
        self._dep_manager = _ModulesDependencyManager()
        self.linearized_layers = []
        self.suppress_rec_ids = {}
        self.visited_nodes = set()  # Keep in memory in the backward / forward graph path exploration nodes visited

    def register_dependencies(self, dependencies_list: List):
        """Register the dependencies between children modules.

        Args:
            dependencies_dict (Dict): a dictionary in which the key is a
                pair of modules and the value is the type of the dependency
                between them.
        """
        for child_module in self.layers:
            self._dep_manager.register_module(
                child_module.get_module_id(), child_module)

        for module1, module2, value in dependencies_list:
            id1, id2 = module1.get_module_id(), module2.get_module_id()
            if value == DepType.INCOMING:
                self._dep_manager.register_incoming_dependency(id1, id2)
            elif value == DepType.SAME:
                self._dep_manager.register_same_dependency(id1, id2)
            elif value == DepType.REC:
                self._dep_manager.register_rec_dependency(id1, id2)

    @property
    def layers(self):
        if not self.linearized_layers:
            self.linearized_layers = get_children(self)
        return self.linearized_layers

    def get_layer_by_id(self, layer_id: int):
        return self._dep_manager.id_2_layer[layer_id]

    def reset_all_stats(self):
        for layer in self.layers:
            if hasattr(layer, "reset_stats"):
                layer.reset_stats()

    def reset_stats_by_layer_id(self, layer_id: int):
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.prune] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        module.reset_stats()

    def get_parameter_count(self):
        count = 0
        for layer in self.parameters():
            count += np.prod(layer.shape)
        return count

    def register_hook_fn_for_architecture_change(self, fn):
        self._architecture_change_hook_fns.append(fn)

    def __hash__(self):
        return hash(self.seen_samples) + \
            hash(self.tracking_mode) + \
            hash(self._dep_manager)

    def set_tracking_mode(self, mode: TrackingMode):
        self.tracking_mode = mode
        for layer in self.layers:
            layer.tracking_mode = mode

    def to(self, device, dtype=None, non_blocking=False, **kwargs):
        self.device = device
        super().to(device, dtype, non_blocking, **kwargs)
        for layer in self.layers:
            layer.to(device, dtype, non_blocking, **kwargs)

    def maybe_update_age(self, tracked_input: th.Tensor):
        if self.tracking_mode != TrackingMode.TRAIN:
            return
        if not hasattr(tracked_input, 'batch_size'):
            setattr(tracked_input, 'batch_size', tracked_input.shape[0])
        self.seen_samples += tracked_input.batch_size

    def get_age(self):
        return self.seen_samples

    def freeze(self, layer_id: int, neuron_ids: Set[int] | None = None):
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.freeze] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        if neuron_ids is None:
            neuron_ids = set(range(module.out_neurons))

        for neuron_id in neuron_ids:
            neuron_lr = module.get_per_neuron_learning_rate(neuron_id)
            module.set_per_neuron_learning_rate(
                neuron_ids={neuron_id}, lr=1.0 - neuron_lr)

    def reinit_neurons(
            self,
            layer_id: int,
            neuron_indices: Set[int],
            perturbation_ratio: float | None = None):
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.prune] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        module.reset(neuron_indices, perturbation_ratio=perturbation_ratio)

        # If the dependent layer is of type "SAME", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            self.reinit_neurons(
                same_dep_id, neuron_indices, perturbation_ratio)

        # If the next layer is of type "INCOMING", say after a conv we have 
        # either a conv or a linear, then we add to incoming neurons.
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            incoming_module.reset_incoming_neurons(
                neuron_indices,
                skip_initialization=True,
                perturbation_ratio=perturbation_ratio)

        # TODO(rotaru): Deal with through_flatten case.

    def _conv_neuron_to_linear_neurons_through_flatten(
            self, conv_layer, linear_layer):
        conv_neurons = conv_layer.weight.shape[0]
        linear_neurons = linear_layer.weight.shape[1]
        linear_neurons_per_conv_neuron = linear_neurons // conv_neurons
        return linear_neurons_per_conv_neuron

    def _same_ancestors(self, node_id: int) -> Set[int]:
        visited = set()
        frontier = {node_id}
        while frontier:
            next_frontier = set()
            for nid in frontier:
                visited.add(nid)
                same_parents = set(self._dep_manager.get_parent_ids(nid, DepType.SAME))
                same_parents -= visited
                next_frontier |= same_parents
            if not next_frontier:
                return frontier
            frontier = next_frontier
        return {node_id}

    def _mask_and_zerofy_new_neurons(self, producer_id: int, new_start: int, new_count: int):
        if new_count <= 0:
            return

        new_ids = set(range(new_start, new_start + new_count))

        self.freeze(producer_id, neuron_ids=new_ids)
        incoming_children = self._dep_manager.get_dependent_ids(producer_id, DepType.INCOMING)
        for child_id in incoming_children:
            child = self._dep_manager.id_2_layer[child_id]
            out_max = getattr(child, "neuron_count", None)
            if out_max is None:
                continue
            all_out = set(range(int(out_max)))
            if hasattr(child, "zerofy_connections_from"):
                try:
                    child.zerofy_connections_from(from_neuron_ids=new_ids, to_neuron_ids=all_out)
                except Exception:
                    pass

        same_children = self._dep_manager.get_dependent_ids(producer_id, DepType.SAME)
        for same_id in same_children:
            try:
                self.freeze(same_id, neuron_ids=new_ids)
            except Exception:
                pass

    def prune(
            self,
            layer_id: int,
            neuron_indices: Tuple[int, Set[int]],
            through_flatten: bool = False):

        # Sanity check
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.prune] No module with id {layer_id}")
        if not isinstance(neuron_indices, (set, list)):
            neuron_indices = set([neuron_indices])  # set the int

        module = self._dep_manager.id_2_layer[layer_id]
        through_flatten = False
        if hasattr(self, "flatten_conv_id"):
            through_flatten = (self.flatten_conv_id == layer_id)

        # If the dependent layer is of type "SAME", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            self.prune(
                same_dep_id, neuron_indices, through_flatten=through_flatten)

        # If the next layer is of type "INCOMING", say after a conv we have
        # either a conv or a linear, then we add to incoming neurons.
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]

            if through_flatten:
                incoming_neurons_per_outgoing_neuron = \
                    incoming_module.in_neurons // module.out_neurons
                incoming_prune_indices = []
                for index in neuron_indices:
                    incoming_prune_indices.extend(list(range(
                        incoming_neurons_per_outgoing_neuron * index,
                        incoming_neurons_per_outgoing_neuron * (index + 1))))
                incoming_module.prune_incoming_neurons(incoming_prune_indices)
            else:
                incoming_module.prune_incoming_neurons(neuron_indices)

        module.prune(neuron_indices)

        for hook_fn in self._architecture_change_hook_fns:
            hook_fn(self)

    def is_flatten_layer(self, module):
        """
        Flatten dimensions are based on the shape of the layer. What is a flatten layer ? If every layers concerned have weights, and so tensor with shape, they must match (B, C, H, W), or (B, H, W, C). The B is important and here everytime. So we base our analysis on the tensor shape.
        """
        if hasattr(module, 'weight'):
            shape = module.weight.shape[1:]  # Remove the batch shape
            return len(shape) == 1
        return False

    def add_neurons(self,
                    layer_id: int,
                    neuron_count: int,
                    skip_initialization: bool = False,
                    _suppress_incoming_ids: Optional[Set[int]] = set(),
                    _suppress_rec_ids: Optional[Set[int]] = set(),
                    _suppress_same_ids: Optional[Set[int]] = set()):
        """
        Basicly this function will be a recursive function operating on the model graph, regarding each path and its label.

        :param layer_id: [description]
        :type layer_id: int
        :param neuron_count: [description]
        :type neuron_count: int
        :param skip_initialization: [description], defaults to False
        :type skip_initialization: bool, optional
        :param _suppress_incoming_ids: [description], defaults to None
        :type _suppress_incoming_ids: Optional[Set[int]], optional
        :param _suppress_same_ids: [description], defaults to None
        :type _suppress_same_ids: Optional[Set[int]], optional
        :raises ValueError: [description]
        """

        # Sanity check to see if layer exists
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.add_neurons] No module with id {layer_id}")

        # Get the current module
        module = self._dep_manager.id_2_layer[layer_id]

        # Sanity check to avoid redundancy
        bypass = hasattr(module, "bypass")
        if layer_id in self.visited_nodes and not bypass:  # Be sure that nodes are updated only one time by pass
            return None

        # ------------------------------------------------------------------- #
        # ------------------------- REC ------------------------------------- #
        # If the dependent layer is of type "REC", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        # Go through parents
        for rec_dep_id in self._dep_manager.get_parent_ids(
                layer_id, DepType.REC):
            if not _suppress_rec_ids:
                _suppress_rec_ids = set()
            if rec_dep_id in _suppress_rec_ids or rec_dep_id == layer_id:
                continue
            _suppress_rec_ids.add(layer_id)
            self.add_neurons(
                rec_dep_id,
                neuron_count, skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids
            )
        # Go through childs
        for rec_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.REC):
            if not _suppress_rec_ids:
                _suppress_rec_ids = set()
            if rec_dep_id in _suppress_rec_ids or rec_dep_id == layer_id:
                continue
            _suppress_rec_ids.add(layer_id)
            self.add_neurons(
                rec_dep_id,
                neuron_count, skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_same_ids=_suppress_same_ids,
                _suppress_rec_ids=_suppress_rec_ids
            )

        # ------------------------------------------------------------------- #
        # ------------------------ SAME ------------------------------------- #
        # If the dependent layer is of type "REC", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_parent_ids(
                layer_id, DepType.SAME):
            if not _suppress_same_ids:
                _suppress_same_ids = set()
            if same_dep_id in _suppress_same_ids or same_dep_id == layer_id:
                continue
            _suppress_same_ids.add(layer_id)
            self.add_neurons(
                same_dep_id,
                neuron_count, skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids
            )
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            if not _suppress_same_ids:
                _suppress_same_ids = set()
            if same_dep_id in _suppress_same_ids or same_dep_id == layer_id:
                continue
            _suppress_same_ids.add(layer_id)
            self.add_neurons(
                same_dep_id,
                neuron_count, skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids
            )

        # ---------------------------------------------------------------- #
        # ------------------------ INCOMING ------------------------------ #
        # If the next layer is of type "INCOMING", say after a conv we have
        # either a conv or a linear, then we add to incoming neurons.
        # Go through childs
        updated_incoming_children: List[int] = []
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            bypass = hasattr(incoming_module, "bypass")
            # to avoid double expansion
            if incoming_id in self.visited_nodes and not bypass:
                continue
            if _suppress_incoming_ids and incoming_id in _suppress_incoming_ids:
                continue

            incoming_skip_initialization = False
            if incoming_id == self.layers[-1].get_module_id():
                incoming_skip_initialization = False

            through_flatten = self.is_flatten_layer(incoming_module)
            if through_flatten:
                incoming_neurons_per_outgoing_neuron = \
                    incoming_module.in_neurons // module.out_neurons
                in_neurons = \
                    neuron_count * incoming_neurons_per_outgoing_neuron
                incoming_module.add_incoming_neurons(
                    in_neurons, incoming_skip_initialization)
            else:
                incoming_module.add_incoming_neurons(
                    neuron_count, incoming_skip_initialization)
            self.visited_nodes.add(incoming_id)

            updated_incoming_children.append(incoming_id)
        module.add_neurons(
                neuron_count, skip_initialization=skip_initialization) if layer_id not in self.visited_nodes else None
        self.visited_nodes.add(layer_id)

        current_parent_out = module.out_neurons
        for child_id in updated_incoming_children:
            sibling_incoming_parents = self._dep_manager.get_parent_ids(child_id, DepType.INCOMING)
            for sib_parent_id in sibling_incoming_parents:
                if sib_parent_id == layer_id:
                    continue
                # to get the producer id
                producer_ids = self._same_ancestors(sib_parent_id)  # e.g., {conv1} instead of {bn1}

                for producer_id in producer_ids:
                    sib_prod_module = self._dep_manager.id_2_layer[producer_id]
                    delta = current_parent_out - sib_prod_module.out_neurons

                    # use bypass to not increase the delta if it is generated from a recursive layers,
                    # i.e., delta came from cat function.
                    if bypass or delta <= 0:
                        continue

                    old_nc = int(sib_prod_module.out_neurons)
                    self.add_neurons(
                        producer_id,
                        neuron_count=delta,
                        skip_initialization=True,
                        _suppress_incoming_ids={child_id}
                    )
                    try:
                        self._mask_and_zerofy_new_neurons(producer_id, new_start=old_nc, new_count=delta)
                    except:
                        pass
        for hook_fn in self._architecture_change_hook_fns:
            hook_fn(self)

    def reorder(self,
                layer_id: int,
                indices: List[int],
                through_flatten: bool = False):

        if layer_id not in self._dep_manager.id_2_layer:
            id_and_type = []
            for id in self._dep_manager.id_2_layer:
                id_and_type.append(
                    (id, type(self._dep_manager.id_2_layer[id])))
            raise ValueError(
                f"[NetworkWithOps.reorder] No module with id {layer_id}"
                f" in {str(id_and_type)}")

        module = self._dep_manager.id_2_layer[layer_id]
        module.reorder(indices)

        through_flatten = False
        if hasattr(self, "flatten_conv_id"):
            through_flatten = (self.flatten_conv_id == layer_id)

        # If the dependent layer is of type "SAME", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            self.reorder(same_dep_id, indices, through_flatten=through_flatten)

        # If the next layer is of type "INCOMING", say after a conv we have 
        # either a conv or a linear, then we add to incoming neurons.
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            if through_flatten:
                incoming_neurons_per_outgoing_neuron = \
                    incoming_module.in_neurons // module.out_neurons
                incoming_reorder_indices = []
                for index in indices:
                    incoming_reorder_indices.extend(
                        list(range(incoming_neurons_per_outgoing_neuron * index,
                                   incoming_neurons_per_outgoing_neuron * (index + 1))))
                incoming_module.reorder_incoming_neurons(incoming_reorder_indices)
            else:
                incoming_module.reorder_incoming_neurons(indices)

        # TODO(rotaru): Deal with through_flatten case.

    def reorder_neurons_by_trigger_rate(self, layer_id: int):
        if layer_id not in self._dep_manager.id_2_layer:
            id_and_type = []
            for id in self._dep_manager.id_2_layer:
                id_and_type.append(
                    (id, type(self._dep_manager.id_2_layer[id])))
            raise ValueError(
                f"[NetworkWithOps.reorder_by] No module with id {layer_id}"
                f" in {str(id_and_type)}")

        module = self._dep_manager.id_2_layer[layer_id]
        if not hasattr(module, 'train_dataset_tracker'):
            raise ValueError(
                f"[NetworkWithOps.reorder_by] Module with id {layer_id} "
                f"has not trackers")
        tracker = module.train_dataset_tracker

        ids_and_rates = []
        for neuron_id in range(tracker.number_of_neurons):
            frq_curr = tracker.get_neuron_stats(neuron_id)
            ids_and_rates.append((neuron_id, frq_curr))
        ids_and_rates.sort(key=lambda x: x[1], reverse=True)
        indices = [idx_and_frq[0] for idx_and_frq in ids_and_rates]

        self.reorder(layer_id=layer_id, indices=indices)

    def model_summary_str(self):
        repr = "Model|"
        for layer in self.layers:
            repr += layer.summary_repr() + "|"
        return repr

    def __eq__(self, other: "NetworkWithOps") -> bool:
        return self.seen_samples == other.seen_samples and \
            self.tracking_mode == other.tracking_mode and \
            self.layers == other.layers

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        state[prefix + 'seen_samples'] = self.seen_samples
        state[prefix + 'tracking_mode'] = self.tracking_mode
        return state

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        self.seen_samples = state_dict[prefix + 'seen_samples']
        self.tracking_mode = state_dict[prefix + 'tracking_mode']
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return super().__repr__() + f" age=({self.seen_samples})"

    def forward(self,
                tensor: th.Tensor,
                intermediary_outputs: List[int] = []):
        x = tensor
        intermediaries = {}

        for layer in self.layers:
            x = layer(x)

            if layer.get_module_id() in intermediary_outputs:
                intermediaries[layer.get_module_id()] = x.detach().cpu()

        if intermediary_outputs:
            return x, intermediaries
        return x
