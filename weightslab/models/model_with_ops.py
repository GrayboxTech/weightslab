import logging
import torch as th

from torch import nn
from enum import Enum
from typing import Iterable, List, Set, Optional, Callable, Dict, Any

from weightslab.components.tracking import TrackingMode
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType
from weightslab.utils.tools import get_children
from weightslab.utils.modules_dependencies import _ModulesDependencyManager, DepType


# Global logger
logger = logging.getLogger(__name__)


def _remap_keys(per_neuron: dict, mapping: Dict[int, int]) -> None:
    """Re-key a {neuron index: value} dict after neurons were removed / inserted."""
    items = [(mapping[k], v) for k, v in list(per_neuron.items()) if k in mapping]
    per_neuron.clear()
    per_neuron.update(items)


def _remap_tracker(tracker, mapping: Dict[int, int], new_size: int) -> None:
    """Move a neuron tracker's per-neuron statistics to the new indexing."""
    old_idx = [o for o in mapping]
    new_idx = [mapping[o] for o in old_idx]
    for attr in ("triggrs_by_neuron", "updates_by_neuron"):
        old = getattr(tracker, attr, None)
        if isinstance(old, th.Tensor):
            new = th.zeros(new_size, dtype=old.dtype, device=old.device)
            keep = [(o, n) for o, n in zip(old_idx, new_idx) if o < len(old)]
            if keep:
                new[[n for _, n in keep]] = old[[o for o, _ in keep]]
            setattr(tracker, attr, new)
    for attr in ("triggers_by_in_id", "triggers_by_class"):
        old = getattr(tracker, attr, None)
        if isinstance(old, list):
            from collections import defaultdict
            fill = set if attr == "triggers_by_in_id" else (lambda: defaultdict(lambda: 0))
            new = [fill() for _ in range(new_size)]
            for o, n in zip(old_idx, new_idx):
                if o < len(old):
                    new[n] = old[o]
            setattr(tracker, attr, new)
    if hasattr(tracker, "number_of_neurons"):
        tracker.number_of_neurons = th.tensor(new_size)


class NetworkWithOps(nn.Module):
    def __init__(self):
        super(NetworkWithOps, self).__init__()

        # Initialize variables
        self.current_step = 0
        self.visited_nodes = set() # Memory trace of explored nodes
        self.visited_incoming_nodes = set() # Memory trace of explored nodes
        self.name = self._get_name() # Name of the model
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
        if not isinstance(other, NetworkWithOps):
            return False
        return other is not None and \
            self.tracking_mode == other.tracking_mode and \
            self.layers == other.layers

    def __hash__(self):
        return hash(self.tracking_mode) + hash(self._dep_manager)

    def __repr__(self):
        return super().__repr__() + f" age=({self.current_step})"

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
        layer_ids = sorted(self._dep_manager.id_2_layer.keys()) if self._dep_manager.id_2_layer else list(range(nb_layers))

        # Negative indexing: map from the end of actual module-id ordering
        if layer_id < 0:
            try:
                return layer_ids[layer_id]
            except Exception:
                return nb_layers + layer_id

        # Positive value can be either a real module_id OR a positional index
        if layer_id in self._dep_manager.id_2_layer:
            return layer_id

        if 0 <= layer_id < len(layer_ids):
            return layer_ids[layer_id]

        return layer_id

    def set_tracking_mode(self, mode: TrackingMode):
        self.tracking_mode = mode
        for layer in self.layers:
            layer.tracking_mode = mode

    def get_age(self):
        return self.current_step

    def increase_age(self):
        self.current_step += 1

    def get_name(self):
        return self.name

    def get_layer_by_id(self, layer_id: int):
        layer = self._dep_manager.id_2_layer.get(layer_id)
        if layer is None:
            available = sorted(self._dep_manager.id_2_layer)
            raise ValueError(
                f"Unknown layer id {layer_id}. Available layer ids: {available}. "
                "Wrap the model with compute_dependencies=True before using "
                "the modelling API."
            )
        return layer

    def _layer_names_by_id(self) -> Dict[int, str]:
        """Return qualified PyTorch names for the registered editable layers."""
        root = getattr(self, "model", None)
        if not isinstance(root, nn.Module):
            root = self

        names = {}
        for qualified_name, module in root.named_modules():
            get_module_id = getattr(module, "get_module_id", None)
            if not callable(get_module_id):
                continue
            try:
                names[get_module_id()] = qualified_name or "<root>"
            except (AttributeError, TypeError):
                continue
        return names

    @staticmethod
    def _neuron_count(layer: nn.Module, attribute: str) -> Optional[int]:
        getter = getattr(layer, "get_neurons", None)
        try:
            value = getter(attribute) if callable(getter) else getattr(layer, attribute, None)
        except (AttributeError, TypeError, ValueError):
            value = getattr(layer, attribute, None)
        return int(value) if isinstance(value, (int, float)) else None

    def get_layer_info(
        self,
        layer_id: int,
        *,
        include_neurons: bool = True,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable description of one editable layer.

        Layer ids are stable for the lifetime of a wrapped model. They are not
        checkpoint identifiers and may change when a new model is wrapped.
        """
        return self._build_layer_info(
            layer_id,
            include_neurons=include_neurons,
            layer_names=self._layer_names_by_id(),
        )

    def _build_layer_info(
        self,
        layer_id: int,
        *,
        include_neurons: bool,
        layer_names: Dict[int, str],
    ) -> Dict[str, Any]:
        layer = self.get_layer_by_id(layer_id)
        input_neurons = self._neuron_count(layer, "in_neurons")
        output_neurons = self._neuron_count(layer, "out_neurons")

        lr_overrides = getattr(layer, "neuron_2_lr", {})
        weight_lrs = lr_overrides.get("weight", {}) if lr_overrides else {}
        neuron_count = output_neurons or 0
        neuron_lrs = [float(weight_lrs.get(index, 1.0)) for index in range(neuron_count)]
        frozen_neurons = {index for index, lr in enumerate(neuron_lrs) if lr == 0.0}

        layer_info = {
            "id": int(layer_id),
            "name": layer_names.get(layer_id, f"layer_{layer_id}"),
            "type": getattr(layer, "module_name", layer.__class__.__name__),
            "input_neurons": input_neurons,
            "output_neurons": output_neurons,
            "parameter_count": sum(
                parameter.numel() for parameter in layer.parameters(recurse=False)
            ),
            "frozen_neuron_count": len(frozen_neurons),
            "operation_counts": dict(getattr(layer, "operation_age", {})),
        }
        if include_neurons:
            layer_info["neurons"] = [
                {
                    "id": index,
                    "learning_rate": neuron_lrs[index],
                    "frozen": index in frozen_neurons,
                }
                for index in range(neuron_count)
            ]
        return layer_info

    def get_model_graph(self, *, include_neurons: bool = False) -> Dict[str, Any]:
        """Return the editable model graph as JSON-serializable primitives.

        The default response is intentionally structural and cheap. Pass
        ``include_neurons=True`` when per-neuron learning-rate/frozen state is
        needed.
        """
        layer_ids = sorted(self._dep_manager.id_2_layer)
        if not layer_ids:
            raise RuntimeError(
                "Model graph information is unavailable. Wrap the model with "
                "wl.watch_or_edit(..., flag='model', compute_dependencies=True)."
            )

        dependency_edges = set()
        for dependency_type, parents in self._dep_manager.dependency_2_id_2_id.items():
            for source_id, target_ids in parents.items():
                dependency_edges.update(
                    (int(source_id), int(target_id), dependency_type.value)
                    for target_id in target_ids
                )
        dependencies = [
            {
                "source_layer_id": source_id,
                "target_layer_id": target_id,
                "type": dependency_type,
            }
            for source_id, target_id, dependency_type in sorted(dependency_edges)
        ]
        layer_names = self._layer_names_by_id()

        return {
            "schema_version": 1,
            "name": self.get_name(),
            "age": self.get_age(),
            "layers": [
                self._build_layer_info(
                    layer_id,
                    include_neurons=include_neurons,
                    layer_names=layer_names,
                )
                for layer_id in layer_ids
            ],
            "dependencies": dependencies,
        }

    def _validated_neuron_indices(
        self,
        layer_id: int,
        neuron_indices: Optional[Iterable[int] | int],
        *,
        allow_empty: bool,
    ) -> Set[int]:
        layer = self.get_layer_by_id(layer_id)
        neuron_count = self._neuron_count(layer, "out_neurons")
        if neuron_count is None:
            raise ValueError(f"Layer {layer_id} does not expose output neurons.")

        if neuron_indices is None:
            indices = set()
        elif isinstance(neuron_indices, int):
            indices = {neuron_indices}
        else:
            indices = set(neuron_indices)
        if not all(isinstance(index, int) for index in indices):
            raise TypeError("neuron_indices must contain integers only.")
        if not indices and not allow_empty:
            raise ValueError("neuron_indices cannot be empty for this operation.")

        invalid = sorted(
            index for index in indices if index < -neuron_count or index >= neuron_count
        )
        if invalid:
            raise ValueError(
                f"Neuron indices {invalid} are outside layer {layer_id}'s "
                f"valid range [-{neuron_count}, {neuron_count - 1}]."
            )
        return indices

    def add_neurons(self, layer_id: int, count: int = 1):
        """Append ``count`` neurons and propagate the change through the graph.

        With the neuron graph (``compute_dependencies=True``) every coupled
        tensor follows -- the next layers' inputs, norms, residual partners,
        heads, tied weights -- and the returned ``EditResult`` says what changed.
        """
        self.get_layer_by_id(layer_id)
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer.")
        if getattr(self, "_neuron_editor", None) is not None:
            return self._edit_neurons(layer_id, "add", count=count)
        self.operate(
            layer_id=layer_id,
            neuron_indices=set(range(count)),
            op_type=ArchitectureNeuronsOpType.ADD,
        )

    def prune_neurons(self, layer_id: int, neuron_indices: Iterable[int] | int):
        """Remove selected neurons and update dependent layers."""
        if getattr(self, "_neuron_editor", None) is not None:
            if isinstance(neuron_indices, int):
                neuron_indices = [neuron_indices]
            indices = list(neuron_indices or [])
            if not indices:
                raise ValueError("neuron_indices cannot be empty for this operation.")
            if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices):
                raise TypeError("neuron_indices must contain integers only.")
            return self._edit_neurons(layer_id, "prune", indices=indices)
        indices = self._validated_neuron_indices(
            layer_id, neuron_indices, allow_empty=False
        )
        self.operate(
            layer_id=layer_id,
            neuron_indices=indices,
            op_type=ArchitectureNeuronsOpType.PRUNE,
        )

    # ------------------------------------------------------------------
    # Neuron graph edits (weightslab.utils.neuron_graph)
    # ------------------------------------------------------------------
    def neuron_edit_blocker(self, layer_id: int) -> Optional[str]:
        """Why neurons of this layer cannot be pruned / added (None: they can)."""
        editor = getattr(self, "_neuron_editor", None)
        if editor is None:
            return None
        return editor.pinned_reason(self.get_layer_by_id(layer_id))

    def _edit_neurons(self, layer_id: int, op: str, indices=(), count: int = 1):
        layer = self.get_layer_by_id(layer_id)
        editor = self._neuron_editor
        nodes = editor.neuron_nodes(layer)
        if nodes is None:
            raise ValueError(f"Layer {layer_id} has no neurons to {op}.")
        if op == "prune":
            invalid = sorted(i for i in indices if not -len(nodes) <= i < len(nodes))
            if invalid:
                raise ValueError(
                    f"Neuron indices {invalid} are outside layer {layer_id}'s "
                    f"valid range [-{len(nodes)}, {len(nodes) - 1}]."
                )
            result = editor.prune(layer, indices)
        else:
            result = editor.add(layer, count)
        self._after_neuron_edit(result, layer, op)
        for hook_fn in self._architecture_change_hook_fns:
            hook_fn(self)
        return result

    def _after_neuron_edit(self, result, target: nn.Module, op: str) -> None:
        """Keep each edited layer's per-neuron state (frozen learning rates,
        trackers, frozen rows, gradient hooks) on the same neurons."""
        from weightslab.utils.neuron_graph import neuron_axis

        per_module: Dict[int, Dict[str, Any]] = {}
        for ch in result.changes:
            entry = per_module.setdefault(id(ch.module), {"module": ch.module, "out": None, "in": None})
            mod = ch.module
            weight = getattr(mod, "weight", None)
            primary = "weight" if isinstance(weight, th.Tensor) and weight.ndim > 0 else None
            if primary is None:
                primary = next((n for n, p in mod.named_parameters(recurse=False) if p.ndim > 0), None)
            out_axis = neuron_axis(mod) if (primary == "weight" and weight.ndim > 1) else 0
            if ch.name == primary and ch.dim == out_axis:
                entry["out"] = ch
            elif ch.name == "weight" and isinstance(weight, th.Tensor) and weight.ndim > 1 \
                    and ch.dim == (1 - out_axis):
                entry["in"] = ch

        for entry in per_module.values():
            mod = entry["module"]
            weight = getattr(mod, "weight", None)
            if entry["out"] is not None:
                ch = entry["out"]
                tensor = getattr(mod, ch.name)
                new_size = int(tensor.shape[ch.dim])
                old_size = new_size + len(ch.positions) - sum(n for _, n in ch.inserts)
                mapping = ch.index_map(old_size)
                for lrs in (getattr(mod, "neuron_2_lr", None) or {}).values():
                    _remap_keys(lrs, mapping)
                for tracker_name in ("train_dataset_tracker", "eval_dataset_tracker"):
                    tracker = getattr(mod, tracker_name, None)
                    if tracker is not None:
                        _remap_tracker(tracker, mapping, new_size)
                frozen = getattr(mod, "_wl_frozen_rows", None)
                if frozen is not None and frozen.rows:
                    axis, idx, values = frozen.rows.get("weight", (0, [], None))
                    kept = sorted(mapping[i] for i in idx if i in mapping)
                    frozen.rows = {"weight": (axis, kept, values)} if kept else {}
                    frozen.recapture(mod)
            if entry["in"] is not None:
                ch = entry["in"]
                new_size = int(weight.shape[ch.dim])
                old_size = new_size + len(ch.positions) - sum(n for _, n in ch.inserts)
                mapping = ch.index_map(old_size)
                for lrs in (getattr(mod, "incoming_neuron_2_lr", None) or {}).values():
                    _remap_keys(lrs, mapping)
            # The weights are new tensors: hooks on the old ones are gone.
            register = getattr(mod, "register_grad_hook", None)
            if callable(register) and isinstance(weight, th.Tensor):
                try:
                    register()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("grad hook re-registration failed: %s", exc)
            if mod is target and hasattr(mod, "operation_age"):
                key = "ADD" if op == "add" else "PRUNE"
                mod.operation_age[key] = mod.operation_age.get(key, 0) + 1

        rebuild = getattr(self, "_rebuild_legacy_dependencies", None)
        if callable(rebuild):
            rebuild()

    def freeze_neurons(
        self,
        layer_id: int,
        neuron_indices: Optional[Iterable[int] | int] = None,
    ) -> None:
        """Freeze selected neurons idempotently (all when omitted)."""
        self._set_neurons_frozen(layer_id, neuron_indices, frozen=True)

    def unfreeze_neurons(
        self,
        layer_id: int,
        neuron_indices: Optional[Iterable[int] | int] = None,
    ) -> None:
        """Unfreeze selected neurons idempotently (all when omitted)."""
        self._set_neurons_frozen(layer_id, neuron_indices, frozen=False)

    def _set_neurons_frozen(
        self,
        layer_id: int,
        neuron_indices: Optional[Iterable[int] | int],
        *,
        frozen: bool,
    ) -> None:
        indices = self._validated_neuron_indices(
            layer_id, neuron_indices, allow_empty=True
        )
        layer = self.get_layer_by_id(layer_id)
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise ValueError(
                f"Layer {layer_id} has no learnable weights to freeze. "
                "Select a learnable layer from get_model_graph()."
            )
        lr_overrides = getattr(layer, "neuron_2_lr", {})
        if not weight.requires_grad or "weight" not in lr_overrides:
            raise ValueError(
                f"Layer {layer_id} does not have a trainable, per-neuron "
                "tracked weight. Freeze or unfreeze it through PyTorch before "
                "wrapping the model."
            )
        neuron_count = self._neuron_count(layer, "out_neurons") or 0
        selected = (
            {index % neuron_count for index in indices}
            if indices
            else set(range(neuron_count))
        )
        weight_lrs = lr_overrides["weight"]
        currently_frozen = {
            index for index in selected if float(weight_lrs.get(index, 1.0)) == 0.0
        }
        indices_to_toggle = (
            selected - currently_frozen if frozen else currently_frozen
        )
        if not indices_to_toggle:
            return
        self.operate(
            layer_id=layer_id,
            neuron_indices=indices_to_toggle,
            op_type=ArchitectureNeuronsOpType.FREEZE,
        )

    def reset_neurons(
        self,
        layer_id: int,
        neuron_indices: Optional[Iterable[int] | int] = None,
    ) -> None:
        """Reinitialize selected neurons (all when omitted)."""
        indices = self._validated_neuron_indices(
            layer_id, neuron_indices, allow_empty=True
        )
        self.operate(
            layer_id=layer_id,
            neuron_indices=indices,
            op_type=ArchitectureNeuronsOpType.RESET,
        )

    def perturb_neurons(
        self,
        layer_id: int,
        neuron_indices: Optional[Iterable[int] | int] = None,
        *,
        ratio: float = 0.1,
    ) -> None:
        """Perturb selected weights by a relative random amount."""
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < ratio < 1:
            raise ValueError("ratio must be strictly between 0 and 1.")
        indices = self._validated_neuron_indices(
            layer_id, neuron_indices, allow_empty=True
        )
        self.operate(
            layer_id=layer_id,
            neuron_indices=indices,
            op_type=ArchitectureNeuronsOpType.RESET,
            perturbation_ratio=float(ratio),
        )

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
        # First clean dependencies
        remove_layers_index = []
        for n, child_module in enumerate(self.layers):
            if not hasattr(child_module, 'module_id'):
                remove_layers_index.append(n)
        for index in sorted(remove_layers_index, reverse=True):
            self.layers.pop(index)

        for child_module in self.layers:
            self._dep_manager.register_module(
                child_module.get_module_id(), child_module)

        # Register dependencies
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
        return self

    def maybe_update_age(self):
        if self.tracking_mode != TrackingMode.TRAIN:
            return

        # Increase model age at each forward pass during training
        self.increase_age()

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
        # Structural edits go through the neuron graph when there is one.
        if getattr(self, "_neuron_editor", None) is not None and op_type is not None:
            op = op_type if isinstance(op_type, ArchitectureNeuronsOpType) \
                else ArchitectureNeuronsOpType(op_type)
            if op in (ArchitectureNeuronsOpType.ADD, ArchitectureNeuronsOpType.PRUNE):
                layer_id = self._reverse_indexing(layer_id, len(self.layers))
                if isinstance(neuron_indices, int):
                    indices = [neuron_indices]
                else:
                    indices = list(neuron_indices or [])
                if op == ArchitectureNeuronsOpType.ADD:
                    return self._edit_neurons(layer_id, "add", count=max(1, len(indices)))
                return self._edit_neurons(layer_id, "prune", indices=indices or [-1])

        # Reset visited nodes memory
        self.visited_nodes = set()
        self.visited_incoming_nodes = set()

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
        logger.debug(f"[DEBUG OPERATE] Called with layer_id={layer_id}")
        layer_id = self._reverse_indexing(layer_id, len(self.layers))
        logger.debug(f"[DEBUG OPERATE] After _reverse_indexing, layer_id={layer_id}")
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.operate] No module with id {layer_id}")

        # Get the current module
        module = self._dep_manager.id_2_layer[layer_id]

        current_parent_out = module.get_neurons(attr_name='out_neurons')

        # Sanity check to avoid redundancy
        # To be sure that nodes are updated only one time by pass
        bypass = hasattr(module, "bypass")

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
                    rec_dep_id == layer_id:
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
            logger.debug(f"[DEBUG REC] Layer {layer_id} has REC child {rec_dep_id}")
            if not _suppress_rec_ids:
                _suppress_rec_ids = set()
            if rec_dep_id in _suppress_rec_ids or \
                    rec_dep_id == layer_id:
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
            logger.debug(f"[DEBUG SAME PARENT] Layer {layer_id} has SAME parent {same_dep_id}")
            if not _suppress_same_ids:
                _suppress_same_ids = set()
            if same_dep_id in _suppress_same_ids or \
                    same_dep_id == layer_id:
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
            logger.debug(f"[DEBUG SAME CHILD] Layer {layer_id} has SAME child {same_dep_id}")
            if not _suppress_same_ids:
                _suppress_same_ids = set()
            if same_dep_id in _suppress_same_ids or \
                    same_dep_id == layer_id:
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
            logger.debug(f"[DEBUG] {module.get_name_wi_id()} (ID: {layer_id}) has INCOMING children IDs: {incoming_ids}")
            logger.debug(f"[DEBUG] Current visited_nodes: {self.visited_nodes}")
            logger.debug(f"[DEBUG] Current _suppress_incoming_ids: {_suppress_incoming_ids}")

        for incoming_id in incoming_ids:
            # Get module with id incoming_id
            incoming_module = self._dep_manager.id_2_layer[incoming_id]

            # Check bypass flag
            # # Bypass flag means here that the node is
            # # incoming several time (inc_node = th.cat([...,]))
            bypass = hasattr(incoming_module, "bypass")

            # Avoid double expansion except for bypass nodes
            if incoming_id in self.visited_incoming_nodes and not bypass:
                logger.debug(f"[DEBUG] Skipping {incoming_module.get_name_wi_id()} (ID: {incoming_id}) because it is in visited_incoming_nodes")
                continue
            if _suppress_incoming_ids and incoming_id \
                    in _suppress_incoming_ids:
                logger.debug(f"[DEBUG] Skipping {incoming_module.get_name_wi_id()} (ID: {incoming_id}) because it is in _suppress_incoming_ids")
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
            self.visited_incoming_nodes.add(incoming_id)

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
            kwargs['current_child_name'] = None # Its child is an Orphan node
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


    def state_dict(self, destination: Optional[Dict[str, Any]] = None, prefix: str = '', keep_vars: bool = False) -> Dict[str, Any]:
        # TODO (GP): Also add jic recursive self.load_state_dict function to manage if self hasattr model, then self.state_dict(not self but self.model)
        if hasattr(self, 'model'):
            state_dict = self.model.state_dict(**{'destination': destination, 'prefix': prefix, 'keep_vars': keep_vars})
        else:
            state_dict = super().state_dict(**{'destination': destination, 'prefix': prefix, 'keep_vars': keep_vars})
        state_dict[prefix + 'current_step'] = self.current_step
        state_dict[prefix + 'tracking_mode'] = self.tracking_mode
        return state_dict

    def load_state_dict(
            self, state_dict, strict=True, assign=True, **kwargs):
        self.current_step = state_dict.pop('current_step', self.current_step)
        self.tracking_mode = state_dict.pop('tracking_mode', self.tracking_mode)

        # Preprocess trackers
        # TODO (GP): better way to handle this? Maybe load the trackers
        state_dict = {
            k: v for k, v in state_dict.items()
            if not '_dataset_tracker' in k
        }
        try:
            if hasattr(self, 'model'):
                self.model.load_state_dict(
                    state_dict, strict=strict, assign=assign, **kwargs
                )
            else:
                super().load_state_dict(
                    state_dict, strict=strict, assign=assign, **kwargs
                )
        except Exception:
            # If the state dict comes from a wrapper (e.g., ModelInterface) it may
            # include a leading "model." prefix. Strip that prefix for matching.
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    remapped[k[len('model.'):]] = v
                else:
                    remapped['model.' + k] = v
            super().load_state_dict(
                remapped, strict=strict, assign=assign, **kwargs)

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
