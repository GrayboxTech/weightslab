import numpy as np
import torch as th

from torch import nn
from enum import Enum, auto
from typing import List, Set, Optional

from weightslab.components.tracking import TrackingMode
from weightslab.utils.tools import get_children
from weightslab.utils.modules_dependencies import _ModulesDependencyManager, \
    DepType


class ArchitectureNeuronsOpType(Enum):
    """
        Different types of operation.
    """
    ADD = auto()
    PRUNE = auto()
    FREEZE = auto()
    RESET = auto()


class NetworkWithOps(nn.Module):
    def __init__(self):
        super(NetworkWithOps, self).__init__()

        self.seen_samples = 0
        self.tracking_mode = TrackingMode.DISABLED
        self._architecture_change_hook_fns = []
        self._dep_manager = _ModulesDependencyManager()
        self.linearized_layers = []
        self.suppress_rec_ids = {}
        self.visited_nodes = set()  # Memory trace of explored nodes
        self.name = self._get_name()

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

    def get_name(self):
        return self.name

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
                same_parents = set(
                    self._dep_manager.get_parent_ids(
                        nid,
                        DepType.SAME
                    )
                )
                same_parents -= visited
                next_frontier |= same_parents
            if not next_frontier:
                return frontier
            frontier = next_frontier
        return {node_id}

    def _mask_and_zerofy_new_neurons(
            self,
            producer_id: int,
            new_start: int,
            new_count: int
    ):
        if new_count <= 0:
            return

        new_ids = set(range(new_start, new_start + new_count))

        self.freeze(producer_id, neuron_ids=new_ids)
        incoming_children = self._dep_manager.get_dependent_ids(
            producer_id,
            DepType.INCOMING
        )
        for child_id in incoming_children:
            child = self._dep_manager.id_2_layer[child_id]
            out_max = getattr(child, "neuron_count", None)
            if out_max is None:
                continue
            all_out = set(range(int(out_max)))
            if hasattr(child, "zerofy_connections_from"):
                try:
                    child.zerofy_connections_from(
                        from_neuron_ids=new_ids,
                        to_neuron_ids=all_out
                    )
                except Exception:
                    pass

        same_children = self._dep_manager.get_dependent_ids(
            producer_id,
            DepType.SAME
        )
        for same_id in same_children:
            try:
                self.freeze(same_id, neuron_ids=new_ids)
            except Exception:
                pass

    def operate(
        self,
        layer_id: int,
        neuron_indices: Set[int] | int = {},
        neuron_operation: Enum = ArchitectureNeuronsOpType.ADD,
        skip_initialization: bool = False,
        _suppress_incoming_ids: Optional[Set[int]] = set(),
        _suppress_rec_ids: Optional[Set[int]] = set(),
        _suppress_same_ids: Optional[Set[int]] = set(),
        **kwargs
    ):
        """
        Basicly this function will be a recursive function operating on the
        model graph, regarding each path and its label.

        :param layer_id: [description]
        :type layer_id: int
        :param neuron_indices: [description]
        :type neuron_indices: int
        :type neuron_operation: Operation TAG
        :param skip_initialization: [description], defaults to False
        :type skip_initialization: bool, optional
        :param _suppress_incoming_ids: [description], defaults to None
        :type _suppress_incoming_ids: Optional[Set[int]], optional
        :param _suppress_same_ids: [description], defaults to None
        :type _suppress_same_ids: Optional[Set[int]], optional
        :raises ValueError: [description]
        """

        # Sanity check to see if layer exists
        if not isinstance(layer_id, int):
            raise ValueError(
                f"[NetworkWithOps.operate] Layer_id ({layer_id}) is not int.")
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.operate] No module with id {layer_id}")

        # Get the current module
        module = self._dep_manager.id_2_layer[layer_id]
        current_parent_out = module.out_neurons

        # Sanity check to avoid redundancy
        # To be sure that nodes are updated only one time by pass
        bypass = hasattr(module, "bypass")
        if layer_id in self.visited_nodes and not bypass:
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
            self.operate(
                rec_dep_id,
                neuron_indices,
                neuron_operation=neuron_operation,
                skip_initialization=skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids,
                **kwargs
            )
        # Go through childs
        for rec_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.REC):
            if not _suppress_rec_ids:
                _suppress_rec_ids = set()
            if rec_dep_id in _suppress_rec_ids or rec_dep_id == layer_id:
                continue
            _suppress_rec_ids.add(layer_id)
            self.operate(
                rec_dep_id,
                neuron_indices,
                neuron_operation=neuron_operation,
                skip_initialization=skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_same_ids=_suppress_same_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                **kwargs
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
            self.operate(
                same_dep_id,
                neuron_indices,
                neuron_operation=neuron_operation,
                skip_initialization=skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids,
                **kwargs
            )
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            if not _suppress_same_ids:
                _suppress_same_ids = set()
            if same_dep_id in _suppress_same_ids or same_dep_id == layer_id:
                continue
            _suppress_same_ids.add(layer_id)
            self.operate(
                same_dep_id,
                neuron_indices,
                neuron_operation=neuron_operation,
                skip_initialization=skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids,
                **kwargs
            )

        # ---------------------------------------------------------------- #
        # ------------------------ INCOMING ------------------------------ #
        # If the next layer is of type "INCOMING", say after a conv we have
        # either a conv or a linear, then we add to incoming neurons.
        # Go through childs
        incoming_module = None
        updated_incoming_children: List[int] = []
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            bypass = hasattr(incoming_module, "bypass")

            # to avoid double expansion
            if incoming_id in self.visited_nodes and not bypass:
                continue
            if _suppress_incoming_ids and incoming_id \
                    in _suppress_incoming_ids:
                continue

            # # Operate on module incoming neurons
            incoming_module.operate(
                neuron_indices=neuron_indices,
                is_incoming=True,
                neuron_operation=neuron_operation,
                skip_initialization=False,
                current_parent_name=module.get_name_wi_id(),
                **kwargs
            )
            # Keep visited node in mem. if bypass flag,
            # i.e., it's the output of a cat layer.
            self.visited_nodes.add(incoming_id) if not bypass else None

            # Save incoming children from layer_id
            updated_incoming_children.append(incoming_id)

        # Operate in module out neurons
        module.operate(
                neuron_indices,
                neuron_operation=neuron_operation,
                skip_initialization=skip_initialization,
                current_child_name=incoming_module.get_name_wi_id()
                if incoming_module is not None else None,
                **kwargs
        ) if layer_id not in self.visited_nodes else None
        self.visited_nodes.add(layer_id)  # Update visited node

        # Iterate over incoming childs
        for child_id in updated_incoming_children:
            # Iterate over my siblings, generated from my parents
            for sib_parent_id in self._dep_manager.get_parent_ids(
                child_id,
                DepType.INCOMING
            ):
                if sib_parent_id == layer_id:
                    continue

                # Get the producer id - e.g., conv1 from batchnorm1
                for producer_id in self._same_ancestors(sib_parent_id):
                    sib_prod_module = self._dep_manager.id_2_layer[producer_id]
                    delta = current_parent_out - sib_prod_module.out_neurons

                    # use bypass to not increase the delta if it is generated
                    # from a recursive layers,
                    # i.e., delta came from cat function.
                    # Not bypass for e.g. __add__ operation
                    if bypass or delta <= 0:
                        continue

                    old_nc = int(sib_prod_module.out_neurons)
                    self.operate(
                        producer_id,
                        neuron_indices=delta,
                        skip_initialization=True,
                        neuron_operation=neuron_operation,
                        _suppress_incoming_ids={child_id},
                        **kwargs
                    )
                    try:
                        self._mask_and_zerofy_new_neurons(
                            producer_id,
                            new_start=old_nc,
                            new_count=delta
                        )
                    except Exception:
                        # TODO (GP): Check why sometime it raises errors
                        pass

        # Hooking
        for hook_fn in self._architecture_change_hook_fns:
            hook_fn(self)

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
