import time
import logging
import numpy as np
import torch as th

from torch import nn
from enum import Enum
from typing import List, Set, Optional, Callable

from weightslab.components.tracking import TrackingMode
from weightslab.utils.tools import get_children
from weightslab.utils.modules_dependencies import _ModulesDependencyManager, \
    DepType


# Global logger
logger = logging.getLogger(__name__)


class NetworkWithOps(nn.Module):
    def __init__(self):
        super(NetworkWithOps, self).__init__()

        # Initialize variables
        self.seen_samples = 0
        self.seen_batched_samples = 0
        self.visited_nodes = set()  # Memory trace of explored nodes
        self.name = self._get_name()  # Name of the model
        self.linearized_layers = []
        self._architecture_change_hook_fns = []
        self.tracking_mode = TrackingMode.DISABLED
        self._dep_manager = _ModulesDependencyManager()

    @property
    def layers(self):
        if not self.linearized_layers:
            self.linearized_layers = get_children(self)
        return self.linearized_layers

    def __eq__(self, other: "NetworkWithOps") -> bool:
        return self.seen_samples == other.seen_samples and \
            self.tracking_mode == other.tracking_mode and \
            self.layers == other.layers

    def __hash__(self):
        return hash(self.seen_samples) + \
            hash(self.tracking_mode) + \
            hash(self._dep_manager)

    def __repr__(self):
        return super().__repr__() + f" age=({self.seen_samples})"
    
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

    def _reverse_indexing(self, layer_id: int, nb_layers: int) -> List[int]:
        """
            Returns the reverse indexing of the layer based on the input shape.
        """
        return (nb_layers + layer_id) if layer_id < 0 else layer_id

    def set_tracking_mode(self, mode: TrackingMode):
        self.tracking_mode = mode
        for layer in self.layers:
            layer.tracking_mode = mode

    def get_age(self):
        return self.seen_samples

    def get_batched_age(self):
        return self.seen_batched_samples
     
    def get_name(self):
        return self.name

    def get_layer_by_id(self, layer_id: int):
        return self._dep_manager.id_2_layer[layer_id]

    def get_parameter_count(self):
        count = 0
        for layer in self.parameters():
            count += np.prod(layer.shape)
        return count

    def register_dependencies(
        self,
        dependencies_list: List
    ):
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

    def register_hook_fn_for_architecture_change(self, fn):
        self._architecture_change_hook_fns.append(fn)

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
        self.seen_batched_samples += 1
        
        # If an instance provides an auto-dump hook (e.g., ModelInterface), call it.
        try:
            hook = getattr(self, '_maybe_auto_dump', None)
            if callable(hook):
                try:
                    hook()
                except Exception:
                    pass
        except Exception:
            pass
    
    def operate(
        self,
        layer_id: int,
        neuron_indices: Set[int] | int = {},
        op_type: Enum = None,
        current_child_name: Optional[str] = None,
        skip_initialization: bool = False,
        _suppress_incoming_ids: Optional[Set[int]] = set(),
        _suppress_rec_ids: Optional[Set[int]] = set(),
        _suppress_same_ids: Optional[Set[int]] = set(),
        dependency: Optional[Callable] = None,
        **kwargs
    ):
        """
        Wrapper function for _operate to reset visited nodes memory.

        :param layer_id: [description]
        :type layer_id: int
        :param neuron_indices: [description]
        :type neuron_indices: int
        :type op_type: Operation TAG
        :param skip_initialization: [description], defaults to False
        :type skip_initialization: bool, optional
        :param _suppress_incoming_ids: [description], defaults to None
        :type _suppress_incoming_ids: Optional[Set[int]], optional
        :param _suppress_same_ids: [description], defaults to None
        :type _suppress_same_ids: Optional[Set[int]], optional
        :param current_child_name: [description], defaults to None
        :type current_child_name: Optional[str], optional
        :param dependency: The type of in/out dependency, defaults to None
        :type dependency: Optional[Callable], optional
        :raises ValueError: [description]
        """
        # Reset visited nodes memory
        self.visited_nodes = set()

        # Call the recursive function
        self._operate(
            layer_id,
            neuron_indices,
            op_type,
            current_child_name,
            skip_initialization,
            _suppress_incoming_ids,
            _suppress_rec_ids,
            _suppress_same_ids,
            dependency,
            **kwargs
        )

        # Final hooking after operation
        for hook_fn in self._architecture_change_hook_fns:
            hook_fn(self)
        
        # wait for all processes to sync - used for bkwd sync
        th.cuda.synchronize() if th.cuda.is_available() else None
        time.sleep(0.5)  # small sleep to ensure all ops are done

    def _operate(
        self,
        layer_id: int,
        neuron_indices: Set[int] | int = {},
        op_type: Enum = None,
        current_child_name: Optional[str] = None,
        skip_initialization: bool = False,
        _suppress_incoming_ids: Optional[Set[int]] = set(),
        _suppress_rec_ids: Optional[Set[int]] = set(),
        _suppress_same_ids: Optional[Set[int]] = set(),
        dependency: Optional[Callable] = None,
        **kwargs
    ):
        """
        Basicly this function will be a recursive function operating on the
        model graph, regarding each path and its label.

        :param layer_id: [description]
        :type layer_id: int
        :param neuron_indices: [description]
        :type neuron_indices: int
        :type op_type: Operation TAG
        :param skip_initialization: [description], defaults to False
        :type skip_initialization: bool, optional
        :param _suppress_incoming_ids: [description], defaults to None
        :type _suppress_incoming_ids: Optional[Set[int]], optional
        :param _suppress_same_ids: [description], defaults to None
        :type _suppress_same_ids: Optional[Set[int]], optional
        :param current_child_name: [description], defaults to None
        :type current_child_name: Optional[str], optional
        :param dependency: The type of in/out dependency, defaults to None
        :type dependency: Optional[Callable], optional
        :raises ValueError: [description]
        """
        logger.debug(f'Operate currently on neurons: {neuron_indices} ' +
              f'of layer id: {layer_id} with op_type: {op_type}')
        # Sanity check to see if layer exists
        if not isinstance(layer_id, int):
            raise ValueError(
                f"[NetworkWithOps.operate] Layer_id ({layer_id}) is not int.")
        if op_type is None:
            raise ValueError(
                f"[NetworkWithOps.operate] Neuron operation " + 
                f"{op_type} has not been defined.")
        
        # Convert to index from back
        print(f"[DEBUG OPERATE] Called with layer_id={layer_id}")
        layer_id = self._reverse_indexing(layer_id, len(self.layers))
        print(f"[DEBUG OPERATE] After _reverse_indexing, layer_id={layer_id}")
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.operate] No module with id {layer_id}")

        # Get the current module
        module = self._dep_manager.id_2_layer[layer_id]
        current_parent_out = module.get_neurons(attr_name='out_neurons')

        # Sanity check to avoid redundancy
        # To be sure that nodes are updated only one time by pass
        bypass = hasattr(module, "bypass")
        if layer_id in self.visited_nodes and not bypass:
            return None

        # ------------------------------------------------------------------- #
        # ------------------------- REC ------------------------------------- #
        # If the dependent layer is of type "REC", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        # # Go through parent nodes
        for rec_dep_id in self._dep_manager.get_parent_ids(
                layer_id, DepType.REC):
            if not _suppress_rec_ids:
                _suppress_rec_ids = set()
            if rec_dep_id in _suppress_rec_ids or \
                    rec_dep_id == layer_id:# or \
                    # kwargs.get('current_child_name', None) == \
                    # self._dep_manager.id_2_layer[rec_dep_id].get_name_wi_id():
                continue
            _suppress_rec_ids.add(layer_id)

            # Operate on the dependent layer
            kwargs['current_child_name'] = module.get_name_wi_id()
            self._operate(
                rec_dep_id,
                neuron_indices,
                op_type=op_type,
                skip_initialization=skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids,
                **kwargs
            )
        # # Go through child nodes
        for rec_dep_id in self._dep_manager.get_child_ids(
                layer_id, DepType.REC):
            print(f"[DEBUG REC] Layer {layer_id} has REC child {rec_dep_id}")
            if not _suppress_rec_ids:
                _suppress_rec_ids = set()
            if rec_dep_id in _suppress_rec_ids or \
                    rec_dep_id == layer_id:# or \
                    # kwargs.get('current_parent_name', None) == \
                    # self._dep_manager.id_2_layer[rec_dep_id].get_name_wi_id():
                continue
            _suppress_rec_ids.add(layer_id)

            # Operate on the dependent layer
            kwargs['current_parent_name'] = module.get_name_wi_id()
            self._operate(
                rec_dep_id,
                neuron_indices,
                op_type=op_type,
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
        # # Go through parents nodes
        for same_dep_id in self._dep_manager.get_parent_ids(
                layer_id, DepType.SAME):
            print(f"[DEBUG SAME PARENT] Layer {layer_id} has SAME parent {same_dep_id}")
            if not _suppress_same_ids:
                _suppress_same_ids = set()
            if same_dep_id in _suppress_same_ids or \
                    same_dep_id == layer_id: #or \
                    # kwargs.get('current_child_name', None) == \
                    # self._dep_manager.id_2_layer[same_dep_id].get_name_wi_id():
                continue
            _suppress_same_ids.add(layer_id)

            # Operate on the dependent layer
            kwargs['current_child_name'] = module.get_name_wi_id()
            self._operate(
                same_dep_id,
                neuron_indices,
                op_type=op_type,
                skip_initialization=skip_initialization,
                _suppress_incoming_ids=_suppress_incoming_ids,
                _suppress_rec_ids=_suppress_rec_ids,
                _suppress_same_ids=_suppress_same_ids,
                **kwargs
            )
        # # Go through child nodes
        for same_dep_id in self._dep_manager.get_child_ids(
                layer_id, DepType.SAME):
            print(f"[DEBUG SAME CHILD] Layer {layer_id} has SAME child {same_dep_id}")
            if not _suppress_same_ids:
                _suppress_same_ids = set()
            if same_dep_id in _suppress_same_ids or \
                    same_dep_id == layer_id: # or \
                    # kwargs.get('current_parent_name', None) == \
                    # self._dep_manager.id_2_layer[same_dep_id].get_name_wi_id():
                continue
            _suppress_same_ids.add(layer_id)

            # Operate on the dependent layer
            kwargs['current_parent_name'] = module.get_name_wi_id()
            self._operate(
                same_dep_id,
                neuron_indices,
                op_type=op_type,
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
        
        incoming_ids = self._dep_manager.get_child_ids(layer_id, DepType.INCOMING)
        if incoming_ids:
            print(f"[DEBUG] {module.get_name_wi_id()} (ID: {layer_id}) has INCOMING children IDs: {incoming_ids}")
            print(f"[DEBUG] Current visited_nodes: {self.visited_nodes}")
            print(f"[DEBUG] Current _suppress_incoming_ids: {_suppress_incoming_ids}")

        for incoming_id in incoming_ids:
            # Get module with id incoming_id
            incoming_module = self._dep_manager.id_2_layer[incoming_id]

            # Check bypass flag
            # # Bypass flag means here that the node is
            # # incoming several time (inc_node = th.cat([...,]))
            bypass = hasattr(incoming_module, "bypass")

            # Avoid double expansion except for bypass nodes
            if incoming_id in self.visited_nodes and not bypass:
                print(f"[DEBUG] Skipping {incoming_module.get_name_wi_id()} (ID: {incoming_id}) because it is in visited_nodes")
                continue
            if _suppress_incoming_ids and incoming_id \
                    in _suppress_incoming_ids:
                print(f"[DEBUG] Skipping {incoming_module.get_name_wi_id()} (ID: {incoming_id}) because it is in _suppress_incoming_ids")
                continue

            # Operate on module incoming neurons
            kwargs['current_parent_name'] = module.get_name_wi_id()
            incoming_module.operate(
                neuron_indices=neuron_indices,
                is_incoming=True,
                op_type=op_type,
                skip_initialization=False,
                dependency=DepType.INCOMING,
                **kwargs
            )

            # Keep visited node in mem. if it's a bypass node,
            # i.e., it's incoming from a cat layer for instance.
            self.visited_nodes.add(incoming_id)

            # Save incoming children from layer_id
            updated_incoming_children.append(incoming_id)

        # ----------------------------------------------------------------- #
        # ------------------------ OUTCOMING ------------------------------ #
        # Operate in module out neurons
        # # Check first the node dependency type
        if dependency is None:
            dependency = DepType.SAME if hasattr(module, 'wl_same_flag') \
                and module.wl_same_flag else None
        # # Define child
        if incoming_module is not None:
            kwargs['current_child_name'] = incoming_module.get_name_wi_id()
        elif current_child_name is not None and current_child_name in module.src_to_dst_mapping_tnsrs:
            kwargs['current_child_name'] = current_child_name
        else:
            kwargs['current_child_name'] = None  # Its child is an Orphan node
        # # Operate
        module.operate(
                neuron_indices,
                op_type=op_type,
                skip_initialization=skip_initialization,
                dependency=dependency,
                **kwargs
        ) if layer_id not in self.visited_nodes else None

        # # Update visited node
        self.visited_nodes.add(layer_id)

        # Iterate over incoming childs of this module
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
                    delta = current_parent_out - \
                        sib_prod_module.get_neurons(attr_name='out_neurons')

                    # use bypass to not increase the delta if it is generated
                    # from a recursive layers,
                    # i.e., delta came from cat function.
                    # Not bypass for e.g. __add__ operation
                    if bypass or delta <= 0:
                        continue

                    # Operate
                    kwargs['current_parent_name'] = module.get_name_wi_id()
                    self._operate(
                        producer_id,
                        neuron_indices=delta,
                        skip_initialization=True,
                        op_type=op_type,
                        _suppress_incoming_ids={child_id},
                        dependency=DepType.INCOMING,
                        **kwargs
                    )


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        state[prefix + 'seen_samples'] = self.seen_samples
        state[prefix + 'tracking_mode'] = self.tracking_mode
        return state

    def load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        self.seen_samples = state_dict[prefix + 'seen_samples']
        self.tracking_mode = state_dict[prefix + 'tracking_mode']
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def load_state_dict(
            self, state_dict, strict, assign=True, **kwargs):
        self.seen_samples = state_dict['seen_samples']
        self.tracking_mode = state_dict['tracking_mode']
        super().load_state_dict(
            state_dict, strict=strict, assign=assign, **kwargs)

    def forward(self,
                tensor: th.Tensor,
                intermediary_outputs: List[int] = []):
        intermediaries = {}
        x = tensor
        for layer in self.layers:
            x = layer(x)

            if layer.get_module_id() in intermediary_outputs:
                intermediaries[layer.get_module_id()] = x.detach().cpu()

        if intermediary_outputs:
            return x, intermediaries
        return x


if __name__ == "__main__":
    print("This is the NetworkWithOps module.")
    from weightslab.baseline_models.pytorch.models import TinyUNet_Straightforward
    from weightslab.backend.model_interface import ModelInterface
    from weightslab.utils.tools import model_op_neurons

    DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

    class TinyUNet_Straightforward(nn.Module):
        """
        Implémentation UNet ultra-minimaliste (1 niveau d'encodage/décodage)
        utilisant l'interpolation pour l'upsampling.

        Architecture:
        Input (H, W) -> Enc1 -> Bottleneck -> Up1 -> Output (H, W)
        """
        def __init__(self, in_channels=1, out_classes=1):
            super().__init__()

            # Set input shape
            self.input_shape = (1, 1, 256, 256)

            # Hyperparamètres (Canaux à chaque étape)
            # c[1]=8 (Encodage/Décodage), c[2]=16 (Bottleneck)
            c = [in_channels, 8, 16]

            # --- A. ENCODER (Down Path) ---
            # 1. ENCODER 1: Conv -> 8 canaux (Génère le skip connection x1)
            self.enc1 = nn.Sequential(
                nn.Conv2d(c[0], c[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(c[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(c[1], c[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(c[1]),
                nn.ReLU(inplace=True)
            )
            self.pool1 = nn.MaxPool2d(2)  # Downsample 1

            # --- B. BOTTLENECK ---
            # 2. BOTTLENECK: Conv -> 16 canaux
            self.bottleneck = nn.Sequential(
                nn.Conv2d(c[1], c[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(c[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(c[2], c[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(c[2]),
                nn.ReLU(inplace=True)
            )

            # --- C. DECODER (Up Path) ---
            # 3. UPSAMPLE 1 (Transition Bottleneck -> Up1)
            # Interpolation du Bottleneck (16 -> 32)
            scale_factor = 2
            self.up_interp1 = nn.Upsample(
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=True
            )
            # Dual conv after cat (In: 16 (bottleneck) + 8 (skip) = 24 -> Out: 8)
            self.up_conv1 = nn.Sequential(
                nn.Conv2d(c[2] + scale_factor * c[1], c[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(c[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(c[1], c[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(c[1]),
                nn.ReLU(inplace=True)
            )

            # --- D. OUTPUT ---
            # 4. OUTPUT: Ramène à out_classes
            self.out_conv = nn.Conv2d(c[1], out_classes, kernel_size=1)

        def forward(self, x):
            # 1. ENCODER
            x1 = self.enc1(x)
            p1 = self.pool1(x1)  # Skip x1

            # 2. BOTTLENECK
            bottleneck = self.bottleneck(p1)

            # 3. DECODER 1: Interp + Concat (x1) + Conv
            up_b = self.up_interp1(bottleneck)
            merged1 = th.cat([x1, self.up_interp1(p1), up_b], dim=1)
            d1 = self.up_conv1(merged1)

            # 4. OUTPUT
            logits = self.out_conv(d1)

            return logits


    def _test_inference(model, dummy_input, op=None):
        import traceback
        # Infer
        try:
            with th.no_grad():
                model(dummy_input)
        except Exception as e:
            print(f"Error during inference: {e}")
            traceback.print_exc()


    # # Initialize model
    model = TinyUNet_Straightforward()
    # # Create dummy input tensor
    dummy_input = th.randn(model.input_shape).to(DEVICE)
    # # Interface the model
    model = ModelInterface(
        model,
        dummy_input=dummy_input,
        print_graph=False
    )
    model.to(DEVICE)
    model.eval()
    layer_id = len(model.layers) // 2  # Middle layer

    # --- Forward Pass Testing ---
    _test_inference(model, dummy_input)

    print('----'*50)
    print('Performing model parameters operations..')
    # model_op_neurons(model, dummy_input=dummy_input, rand=False)

    # if True:
    #     n = 7
    #     print(n)
    #     with model as m:
    #         print('Adding operation - 5 neurons added.')
    #         m.operate(n, {-1, -1}, op_type=2)
    #         m(dummy_input) if dummy_input is not None else None
    if True:
        n = 6
        print(n)
        with model as m:
            print('Adding operation - 5 neurons added.')
            m.operate(n, {-1, -1}, op_type=2)
            m(dummy_input) if dummy_input is not None else None
    len(m.layers[8].dst_to_src_mapping_tnsrs['BatchNorm2d_3']) == m.layers[3].out_neurons == m.layers[3].in_neurons
    len(m.layers[8].dst_to_src_mapping_tnsrs['BatchNorm2d_7']) == m.layers[7].out_neurons == m.layers[7].in_neurons


    # if True:
    #     n = 3
    #     print(n)
    #     with model as m:
    #         print('Adding operation - 5 neurons added.')
    #         m.operate(n, {-1, -1}, op_type=2)
    #         m(dummy_input) if dummy_input is not None else None
    # if True:
    #     n = 2
    #     print(n)
    #     with model as m:
    #         print('Adding operation - 5 neurons added.')
    #         m.operate(n, {-1, -1}, op_type=2)
    #         m(dummy_input) if dummy_input is not None else None
    print()