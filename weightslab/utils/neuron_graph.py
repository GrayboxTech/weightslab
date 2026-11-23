"""Which neurons go together: the coupling graph behind neuron prune / add.

Removing output neuron ``i`` of a layer is only valid if every tensor that
carries that neuron loses the same index: the next layer's input column, a
batch norm's statistics, the other branch of a residual add, the slice of a
concatenation, the matching head position of the key projection, the tied
output embedding ... This module finds all of them for any model, by running
it once on an example input.

How it works
------------
* Every dimension of every parameter and buffer is a *space*: one node per
  index. ``Linear.weight`` (out, in) has an "out" space and an "in" space.
* One forward runs under a ``TorchFunctionMode``. Each tensor carries, per
  dimension, the node of each position (or nothing: batch, sequence and
  spatial dimensions are not resizable). Each op says how its output
  dimensions relate to its inputs' (linear: ``x[..., j]`` meets column ``j``;
  add: same position, same neuron; cat: concatenation; view: split or merge
  ...). Nodes that must go together are joined (union-find).
* Reshapes that split one dimension into several (heads: ``C -> (n_head,
  head_size)``; ``chunk(3)`` of a fused q/k/v projection; grouped convs) or
  merge several into one (``flatten`` of ``C, H, W``) make a *grid*: a cell can
  only go with a whole hyperplane of it -- the same position in every head,
  all ``H*W`` columns of a channel.
* Nodes the model cannot change are *pinned*: positions that meet an
  unresizable dimension, the model outputs (their size is the task: classes,
  vocabulary, image channels), dimensions of ops this module does not know
  (unknown ops pin what they touch: an edit never guesses).

An edit takes the closure of the chosen neurons over couplings and grids,
removes (or inserts) those indices in every parameter and buffer, updates the
modules' size attributes, then re-runs the example input. If anything is off
(an error, a different output shape -- e.g. a forward that hard-codes a size)
the edit is rolled back and another grid layout is tried; if none works the
edit is refused with the reason.

A forward that should support edits derives sizes from its tensors
(``q.view(B, T, self.n_head, -1)``) instead of hard-coding them.
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from torch.overrides import TorchFunctionMode

logger = logging.getLogger(__name__)

Ids = Optional[np.ndarray]          # node id per position of one dim; -1 = fixed position


class NeuronEditError(ValueError):
    """An edit that cannot be done (with the reason why)."""


# =============================================================================
# Graph state
# =============================================================================
@dataclass(eq=False)
class _Space:
    offset: int
    size: int
    kind: str                         # "param" | "buffer" | "virtual"
    tensor: Optional[th.Tensor] = None
    dim: int = 0
    name: str = ""
    role: str = ""                    # "in" for dims contracted by a layer


@dataclass
class _Grid:
    """``cells`` (node ids) laid out on axes; ``axes[j][i]`` is the node of
    hyperplane ``i`` along axis ``j`` (None: that axis cannot shrink)."""
    cells: np.ndarray
    axes: List[Ids]
    why: str = ""


@dataclass
class TensorChange:
    """One dim of one parameter / buffer edited: ``positions`` removed (prune),
    or ``inserts`` = [(at, how many)] in the old indexing (add)."""
    module: nn.Module
    module_name: str
    name: str
    dim: int
    positions: List[int] = field(default_factory=list)
    inserts: List[Tuple[int, int]] = field(default_factory=list)

    def index_map(self, old_size: int) -> Dict[int, int]:
        """Old index -> new index (removed ones absent)."""
        if self.positions:
            gone = set(self.positions)
            kept = [i for i in range(old_size) if i not in gone]
            return {old: new for new, old in enumerate(kept)}
        mapping, shift, ins = {}, 0, sorted(self.inserts)
        k = 0
        for i in range(old_size):
            while k < len(ins) and ins[k][0] <= i:
                shift += ins[k][1]
                k += 1
            mapping[i] = i + shift
        return mapping


@dataclass
class EditResult:
    op: str
    changes: List[TensorChange] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)            # modules whose tensors changed
    neurons: int = 0                                           # neurons of the target layer added / removed
    total: int = 0                                             # neuron indices added / removed overall

    @property
    def message(self) -> str:
        verb = "pruned" if self.op == "prune" else "added"
        others = max(0, len(self.layers) - 1)
        tail = f" (+{others} coupled layer{'s' if others != 1 else ''})" if others else ""
        return f"{verb} {self.neurons} neuron(s){tail}"


class NeuronGraph:
    """Neuron couplings of one model (see the module docstring)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.spaces: List[_Space] = []
        self.parent: List[int] = []
        self.pinned: Dict[int, str] = {}
        self.grids: List[_Grid] = []
        self.unknown_ops: Dict[str, int] = {}
        self.output_shapes: List[Tuple[int, ...]] = []
        self._tensor_space: Dict[Tuple[int, int], int] = {}   # (id(tensor), dim) -> space index
        self._tags: Dict[int, Tuple[Any, List[Ids]]] = {}        # id -> (weakref, tags)
        self._keep: List[th.Tensor] = []
        self.shape_attrs: List[Tuple[nn.Module, str, List[Ids], Tuple[int, ...]]] = []
        self._stack: List[nn.Module] = []
        self.module_out: Dict[int, np.ndarray] = {}     # weightless layer -> the neurons it passes on
        self._finalized = False

    # ------------------------------------------------------------------ nodes
    def _new_space(self, size: int, kind: str, tensor=None, dim=0, name="") -> _Space:
        space = _Space(len(self.parent), int(size), kind, tensor, dim, name)
        self.spaces.append(space)
        self.parent.extend(range(space.offset, space.offset + space.size))
        return space

    def _fresh(self, size: int, name: str = "") -> np.ndarray:
        space = self._new_space(size, "virtual", name=name)
        return np.arange(space.offset, space.offset + space.size, dtype=np.int64)

    def find(self, a: int) -> int:
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def _couple(self, a: Ids, b: Ids, why: str = "") -> None:
        """Same position, same neuron. A position fixed on one side pins the other."""
        if a is None and b is None:
            return
        if a is None or b is None:
            self._pin(b if a is None else a, why or "meets a dimension that cannot change")
            return
        if len(a) != len(b):
            self._pin(a, why or "size mismatch")
            self._pin(b, why or "size mismatch")
            return
        for x, y in zip(a.tolist(), b.tolist()):
            if x >= 0 and y >= 0:
                self._union(x, y)
            elif x >= 0:
                self._pin_node(x, why or "meets a dimension that cannot change")
            elif y >= 0:
                self._pin_node(y, why or "meets a dimension that cannot change")

    def _pin_node(self, node: int, why: str) -> None:
        self.pinned.setdefault(int(node), why)

    def _pin(self, ids: Ids, why: str) -> None:
        if ids is None:
            return
        for node in np.asarray(ids).ravel().tolist():
            if node >= 0:
                self._pin_node(node, why)

    # ------------------------------------------------------------------- tags
    def _tag(self, t: Any) -> Optional[List[Ids]]:
        if not isinstance(t, th.Tensor):
            return None
        entry = self._tags.get(id(t))
        if entry is not None and entry[0]() is t and len(entry[1]) == t.ndim:
            return entry[1]
        return [None] * t.ndim

    def _set(self, t: Any, tags: List[Ids]) -> None:
        if isinstance(t, th.Tensor):
            if len(tags) != t.ndim:
                tags = [None] * t.ndim
            try:
                ref = weakref.ref(t)
            except TypeError:
                self._keep.append(t)
                ref = (lambda t=t: t)
            self._tags[id(t)] = (ref, [None if (x is None or not (np.asarray(x) >= 0).any()) else x
                                       for x in tags])

    def _register_tensor(self, tensor: th.Tensor, kind: str, name: str) -> None:
        if (id(tensor), 0) in self._tensor_space or tensor.ndim == 0:   # tied: one tensor, two names
            return
        tags = []
        for d, size in enumerate(tensor.shape):
            space = self._new_space(size, kind, tensor, d, f"{name}[{d}]")
            self._tensor_space[(id(tensor), d)] = len(self.spaces) - 1
            tags.append(np.arange(space.offset, space.offset + size, dtype=np.int64))
        self._tags[id(tensor)] = ((lambda t=tensor: t), tags)

    def space_of(self, tensor: th.Tensor, dim: int) -> Optional[_Space]:
        index = self._tensor_space.get((id(tensor), dim))
        return self.spaces[index] if index is not None else None

    # ================================================================== trace
    @classmethod
    def trace(cls, model: nn.Module, example: Any) -> "NeuronGraph":
        graph = cls(model)
        for name, param in model.named_parameters(remove_duplicate=True):
            graph._register_tensor(param, "param", name)
        for name, buf in model.named_buffers(remove_duplicate=True):
            if buf is not None and buf.ndim > 0:
                graph._register_tensor(buf, "buffer", name)
        was_training = model.training
        mine = {id(m) for m in model.modules()}
        stack = graph._stack

        def _enter(mod, _args):
            if id(mod) in mine:
                stack.append(mod)

        def _leave(mod, _args, out):
            if stack and stack[-1] is mod:
                stack.pop()
            if id(mod) in mine and id(mod) not in graph.module_out and isinstance(out, th.Tensor) \
                    and out.ndim >= 2 and not any(True for _ in mod.parameters(recurse=False)):
                tags = graph._tag(out)
                dim = out.ndim - 1 if out.ndim == 3 else 1
                if tags[dim] is not None:
                    graph.module_out[id(mod)] = tags[dim].copy()

        hooks = [nn.modules.module.register_module_forward_pre_hook(_enter),
                 nn.modules.module.register_module_forward_hook(_leave)]
        model.eval()
        try:
            with th.no_grad(), _Tracer(graph):
                out = _call(model, example)
        finally:
            for h in hooks:
                h.remove()
            stack.clear()
            model.train(was_training)
        for t in _tensors_in(out):
            graph.output_shapes.append(tuple(t.shape))
            for ids in graph._tag(t) or []:
                graph._pin(ids, "a model output (its size is the task's)")
        graph._finalize()
        return graph

    def _finalize(self) -> None:
        self._tags, self._keep = {}, []      # free the activations of the trace
        self._roots = np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)
        self._members: Dict[int, List[int]] = {}
        for node, root in enumerate(self._roots.tolist()):
            self._members.setdefault(root, []).append(node)
        self._pinned_roots: Dict[int, str] = {}
        for node, why in self.pinned.items():
            self._pinned_roots.setdefault(int(self._roots[node]), why)
        self._cell_of: Dict[int, List[Tuple[int, int]]] = {}
        self._axis_of: Dict[int, List[Tuple[int, int, int]]] = {}
        for g_index, grid in enumerate(self.grids):
            for pos, node in enumerate(grid.cells.ravel().tolist()):
                if node >= 0:
                    self._cell_of.setdefault(node, []).append((g_index, pos))
            for j, axis in enumerate(grid.axes):
                if axis is None:
                    continue
                for i, node in enumerate(axis.tolist()):
                    if node >= 0:
                        self._axis_of.setdefault(node, []).append((g_index, j, i))
        self._finalized = True

    # ============================================================== op rules
    def on_op(self, func: Callable, args: tuple, kwargs: dict, out: Any) -> None:
        name = _op_name(func)
        rule = _RULES.get(name)
        outs = [t for t in _tensors_in(out)]
        if not outs and name not in ("__setitem__",):
            return
        if rule is not None:
            rule(self, name, args, kwargs, out)
            return
        self._generic(name, args, kwargs, out)

    def _generic(self, name: str, args, kwargs, out) -> None:
        ins = [t for t in _tensors_in((args, kwargs))]
        outs = list(_tensors_in(out))
        if len(outs) == 1 and ins:
            o = outs[0]
            if len(ins) == 1 and tuple(ins[0].shape) == tuple(o.shape):
                self._set(o, list(self._tag(ins[0])))        # same-shape op: positions kept
                return
            try:
                shape = th.broadcast_shapes(*[t.shape for t in ins])
            except Exception:
                shape = None
            if shape is not None and tuple(shape) == tuple(o.shape):
                self._set(o, self._elementwise(o.shape, ins, name))
                return
        self.unknown_ops[name] = self.unknown_ops.get(name, 0) + 1
        for t in ins:
            for ids in self._tag(t):
                self._pin(ids, f"goes through `{name}`, which neuron edits do not handle")

    def _elementwise(self, shape, inputs: Sequence[th.Tensor], name: str = "") -> List[Ids]:
        nd = len(shape)
        tags = [None] * nd
        for d in range(nd):
            size = int(shape[d])
            found: List[np.ndarray] = []
            fixed = False
            for t in inputs:
                k = d - (nd - t.ndim)
                if k < 0 or int(t.shape[k]) != size:
                    continue
                ids = self._tag(t)[k]
                if ids is None:
                    fixed = fixed or size > 1
                else:
                    found.append(ids)
            if found:
                for other in found[1:]:
                    self._couple(found[0], other, f"`{name}` of two tensors")
                if fixed:
                    self._pin(found[0], f"`{name}` with a tensor whose size cannot change")
                tags[d] = found[0]
        return tags

    # --- grids ---------------------------------------------------------------
    def _split(self, ids: Ids, sizes: Sequence[int], why: str) -> List[Ids]:
        """One dim of ``ids`` becomes dims of ``sizes`` (row-major)."""
        if ids is None:
            return [None] * len(sizes)
        # Undo of a merge (unflatten after flatten): give the original dims back.
        for grid in self.grids:
            if grid.cells.ndim == len(sizes) and tuple(grid.cells.shape) == tuple(sizes) \
                    and grid.why == "merge" and np.array_equal(grid.cells.ravel(), ids):
                return [None if a is None else a for a in grid.axes]
        cells = ids.reshape(tuple(sizes))
        axes = [self._fresh(s, why) if s > 1 else None for s in sizes]
        self.grids.append(_Grid(cells, axes, why))
        return [a for a in axes]

    def _merge(self, dims: Sequence[Ids], sizes: Sequence[int], why: str = "merge") -> Ids:
        """Several dims become one (row-major)."""
        live = [(ids, s) for ids, s in zip(dims, sizes) if s > 1 or ids is not None]
        if not live:
            return None
        if len(live) == 1:
            return live[0][0]
        if all(ids is None for ids, _ in live):
            return None
        # Undo of a split (heads merged back): the original ids.
        for grid in self.grids:
            if grid.why == "merge" or len(grid.axes) != len(live):
                continue
            if all(a is not None and ids is not None and (a is ids or np.array_equal(a, ids))
                   for a, (ids, _) in zip(grid.axes, live)):
                return grid.cells.ravel()
        shape = tuple(s for _, s in live)
        cells = self._fresh(int(np.prod(shape)), why).reshape(shape)
        self.grids.append(_Grid(cells, [ids for ids, _ in live], "merge"))
        return cells.ravel()

    def _grid_on(self, ids: Ids, n_outer: int, why: str) -> None:
        """``ids`` laid out as ``n_outer`` equal blocks that must stay equal
        (groups, chunks): a position goes with the same position in every block."""
        if ids is None or n_outer <= 1 or len(ids) % n_outer:
            return
        cells = ids.reshape(n_outer, len(ids) // n_outer)
        inner = self._fresh(cells.shape[1], why) if cells.shape[1] > 1 else None
        if inner is None:
            self._pin(ids, f"{why}: one position per block")
            return
        self.grids.append(_Grid(cells, [None, inner], why))

    def _reshape(self, tags: List[Ids], in_shape, out_shape, name: str) -> List[Ids]:
        """view / reshape / flatten / unflatten / (un)squeeze: match groups of
        input dims to groups of output dims with the same number of elements.
        Only dims that matter take part: size > 1, or holding a neuron (a
        1-channel layer's channel dim is real even at size 1)."""
        in_shape, out_shape = list(map(int, in_shape)), list(map(int, out_shape))
        out: List[Ids] = [None] * len(out_shape)
        I = [k for k in range(len(in_shape)) if in_shape[k] > 1 or tags[k] is not None]
        J = [k for k in range(len(out_shape)) if out_shape[k] > 1]
        i = j = 0
        while i < len(I) or j < len(J):
            gi, gj = [], []
            pi = pj = 1
            if i < len(I):
                gi.append(I[i]); pi *= in_shape[I[i]]; i += 1
            if j < len(J):
                gj.append(J[j]); pj *= out_shape[J[j]]; j += 1
            while pi != pj:
                if pi < pj and i < len(I):
                    gi.append(I[i]); pi *= in_shape[I[i]]; i += 1
                elif pj < pi and j < len(J):
                    gj.append(J[j]); pj *= out_shape[J[j]]; j += 1
                else:
                    break
            if pi != pj:
                for k in gi + I[i:]:
                    self._pin(tags[k], f"`{name}` that this module cannot follow")
                break
            if not gj:
                for k in gi:
                    self._pin(tags[k], f"`{name}` squeezes this dimension away")
                continue
            if not gi:
                continue
            if len(gi) == 1 and len(gj) == 1 and in_shape[gi[0]] == out_shape[gj[0]]:
                out[gj[0]] = tags[gi[0]]
                continue
            merged = self._merge([tags[k] for k in gi], [in_shape[k] for k in gi])
            if len(gj) == 1:
                out[gj[0]] = merged
            else:
                parts = self._split(merged, [out_shape[k] for k in gj], f"`{name}` split")
                for k, ids in zip(gj, parts):
                    out[k] = ids
        return out


# =============================================================================
# The trace mode
# =============================================================================
class _Tracer(TorchFunctionMode):
    def __init__(self, graph: NeuronGraph):
        super().__init__()
        self.graph = graph

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            self.graph.on_op(func, args, kwargs, out)
        except Exception as exc:      # a rule must never break the model's forward
            name = _op_name(func)
            logger.debug("neuron graph: rule for %s failed: %s", name, exc)
            g = self.graph
            g.unknown_ops[name] = g.unknown_ops.get(name, 0) + 1
            for t in _tensors_in((args, kwargs)):
                for ids in g._tag(t) or []:
                    g._pin(ids, f"`{name}` (could not be analysed)")
        return out


def _op_name(func) -> str:
    name = getattr(func, "__name__", "") or str(func)
    if name == "__get__":
        owner = getattr(func, "__self__", None)
        return "get:" + str(getattr(owner, "__name__", ""))
    return name


def _tensors_in(obj, _depth: int = 0):
    if _depth > 6:
        return
    if isinstance(obj, th.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _tensors_in(v, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _tensors_in(v, _depth + 1)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (nn.Module, type)) and _depth < 3 \
            and type(obj).__module__ not in ("builtins",):
        for v in vars(obj).values():
            yield from _tensors_in(v, _depth + 1)


def _signature(model: nn.Module) -> Tuple:
    return tuple((id(t), tuple(t.shape)) for t in model.parameters()) +         tuple((id(t), tuple(t.shape)) for t in model.buffers())


def _device_of(model: nn.Module):
    for t in model.parameters():
        return t.device
    for t in model.buffers():
        return t.device
    return th.device("cpu")


def _to_device(example, device):
    if isinstance(example, th.Tensor):
        return example.to(device)
    if isinstance(example, dict):
        return {k: _to_device(v, device) for k, v in example.items()}
    if isinstance(example, (list, tuple)):
        return type(example)(_to_device(v, device) for v in example)
    return example


def _call(model, example):
    if isinstance(example, dict):
        return model(**example)
    if isinstance(example, (tuple, list)):
        return model(*example)
    return model(example)


def _arg(args, kwargs, index, name, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _dims(dim, ndim: int) -> List[int]:
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, (list, tuple, th.Size)):
        return [int(d) % ndim for d in dim]
    return [int(dim) % ndim]


# =============================================================================
# Rules: (graph, name, args, kwargs, out) -> None, setting the outputs' tags
# =============================================================================
_RULES: Dict[str, Callable] = {}


def _rule(*names):
    def deco(fn):
        for n in names:
            _RULES[n] = fn
        return fn
    return deco


_NON_TENSOR = (
    "get:shape", "get:ndim", "get:dtype", "get:device", "get:is_cuda", "get:requires_grad",
    "get:grad_fn", "get:layout", "size", "dim", "numel", "stride", "item", "tolist",
    "__len__", "is_contiguous", "data_ptr", "element_size", "__bool__", "__index__",
    "__int__", "__float__", "__format__", "__repr__", "__hash__", "is_floating_point",
    "nelement", "get_device", "storage_offset", "is_complex", "__iter__",
)


@_rule(*_NON_TENSOR)
def _r_skip(g, name, args, kwargs, out):
    return


@_rule(
    "relu", "relu_", "relu6", "gelu", "silu", "silu_", "sigmoid", "sigmoid_", "tanh", "tanh_",
    "elu", "elu_", "selu", "celu", "leaky_relu", "leaky_relu_", "hardtanh", "hardtanh_",
    "hardswish", "hardsigmoid", "mish", "softplus", "softsign", "logsigmoid", "tanhshrink",
    "threshold", "threshold_", "dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout",
    "feature_alpha_dropout", "abs", "neg", "__neg__", "__abs__", "exp", "exp_", "log", "log_",
    "sqrt", "rsqrt", "sin", "cos", "square", "reciprocal", "sign", "clamp", "clamp_", "clip",
    "contiguous", "clone", "detach", "detach_", "float", "half", "double", "bfloat16", "to",
    "type", "type_as", "cpu", "cuda", "requires_grad_", "softmax", "log_softmax", "_softmax",
    "normalize", "erf", "floor", "ceil", "round", "trunc", "nan_to_num", "cumsum", "cumprod",
    "get:data", "get:real", "silu", "softmin", "gumbel_softmax", "fill_", "zero_", "__invert__",
    "logical_not", "isnan", "isinf", "isfinite", "bool", "long", "int", "short", "byte", "char",
    "masked_scatter", "float_power", "expm1", "log1p", "sinh", "cosh", "atan", "asin", "acos",
    "deg2rad", "rad2deg", "frac", "positive",
)
def _r_pointwise(g, name, args, kwargs, out):
    src = next((t for t in _tensors_in((args, kwargs))), None)
    for o in _tensors_in(out):
        if src is not None and tuple(src.shape) == tuple(o.shape):
            g._set(o, list(g._tag(src)))
        else:
            g._generic(name, args, kwargs, out)
            return


@_rule(
    "add", "add_", "__add__", "__radd__", "__iadd__", "sub", "sub_", "__sub__", "__rsub__",
    "__isub__", "rsub", "mul", "mul_", "__mul__", "__rmul__", "__imul__", "div", "div_",
    "__truediv__", "__rtruediv__", "__itruediv__", "__div__", "true_divide", "floor_divide",
    "__floordiv__", "__rfloordiv__", "pow", "pow_", "__pow__", "__rpow__", "maximum",
    "minimum", "fmax", "fmin", "where", "masked_fill", "masked_fill_", "lerp", "lerp_",
    "addcmul", "addcmul_", "addcdiv", "addcdiv_", "eq", "ne", "lt", "le", "gt", "ge",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "logical_and", "logical_or",
    "logical_xor", "__and__", "__or__", "__xor__", "__iand__", "__ior__", "fmod", "remainder",
    "__mod__", "atan2", "hypot", "copysign", "xlogy", "copy_", "mul_scalar", "sigmoid_focal",
    "bitwise_and", "bitwise_or", "squared_difference", "prelu_backward_placeholder",
)
def _r_elementwise(g, name, args, kwargs, out):
    ins = list(_tensors_in((args, kwargs)))
    outs = list(_tensors_in(out))
    if not outs:
        return
    o = outs[0]
    tags = g._elementwise(o.shape, ins, name)
    if ins and ins[0] is o:                          # in place: couples, keeps its tags
        return
    g._set(o, tags)


@_rule("max", "min")
def _r_maxmin(g, name, args, kwargs, out):
    ins = list(_tensors_in((args, kwargs)))
    if len(ins) >= 2:
        return _r_elementwise(g, name, args, kwargs, out)
    return _r_reduce(g, name, args, kwargs, out)


@_rule("sum", "mean", "amax", "amin", "prod", "std", "var", "norm", "logsumexp", "all", "any",
       "argmax", "argmin", "count_nonzero", "median", "nansum", "nanmean", "std_mean", "var_mean",
       "mode", "kthvalue")
def _r_reduce(g, name, args, kwargs, out):
    x = args[0] if args and isinstance(args[0], th.Tensor) else kwargs.get("input")
    if not isinstance(x, th.Tensor):
        return
    tags = g._tag(x)
    dim = _arg(args, kwargs, 1, "dim", None)
    if name == "norm":
        dim = _arg(args, kwargs, 2, "dim", None)
        keepdim = bool(_arg(args, kwargs, 3, "keepdim", False))
    else:
        keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    if isinstance(dim, (int, list, tuple)) or dim is None:
        reduced = set(_dims(dim, x.ndim)) if x.ndim else set()
    else:
        reduced = set(range(x.ndim))
    for o in _tensors_in(out):
        if o.ndim == x.ndim:
            new = [None if d in reduced else tags[d] for d in range(x.ndim)]
        else:
            new = [tags[d] for d in range(x.ndim) if d not in reduced]
        g._set(o, new if len(new) == o.ndim else [None] * o.ndim)


@_rule("view", "reshape", "view_as", "reshape_as", "flatten", "unflatten", "squeeze", "squeeze_",
       "unsqueeze", "unsqueeze_", "ravel", "_reshape_alias", "atleast_1d", "atleast_2d")
def _r_reshape(g, name, args, kwargs, out):
    x = args[0] if args and isinstance(args[0], th.Tensor) else kwargs.get("input")
    o = next(_tensors_in(out), None)
    if not isinstance(x, th.Tensor) or o is None:
        return
    tags = g._tag(x)
    if name.endswith("_") and o is x:
        return
    new = g._reshape(tags, x.shape, o.shape, name)
    g._set(o, new)
    mod = g._stack[-1] if g._stack else None
    if name == "unflatten" and isinstance(mod, nn.Unflatten):
        d = int(mod.dim) % x.ndim
        k = len(tuple(mod.unflattened_size))
        g.shape_attrs.append((mod, "unflattened_size", new[d:d + k], tuple(int(v) for v in o.shape[d:d + k])))


@_rule("expand", "expand_as", "broadcast_to")
def _r_expand(g, name, args, kwargs, out):
    x = args[0]
    o = next(_tensors_in(out), None)
    if o is None:
        return
    tags = g._tag(x)
    lead = o.ndim - x.ndim
    new = [None] * lead + [tags[d] if int(x.shape[d]) == int(o.shape[lead + d]) else None
                           for d in range(x.ndim)]
    g._set(o, new)


@_rule("repeat", "tile")
def _r_repeat(g, name, args, kwargs, out):
    x = args[0]
    o = next(_tensors_in(out), None)
    if o is None:
        return
    tags = g._tag(x)
    lead = o.ndim - x.ndim
    new: List[Ids] = [None] * lead
    for d in range(x.ndim):
        ids = tags[d]
        factor = int(o.shape[lead + d]) // max(1, int(x.shape[d]))
        new.append(None if ids is None else (np.tile(ids, factor) if factor > 1 else ids))
    g._set(o, new)


@_rule("permute", "transpose", "transpose_", "t", "t_", "movedim", "moveaxis", "swapaxes",
       "swapdims", "get:T", "get:mT", "adjoint", "get:H", "get:mH")
def _r_permute(g, name, args, kwargs, out):
    x = args[0]
    o = next(_tensors_in(out), None)
    if o is None or not isinstance(x, th.Tensor):
        return
    tags = g._tag(x)
    nd = x.ndim
    order = list(range(nd))
    if name == "permute":
        dims = args[1:] if len(args) > 2 else (args[1] if len(args) > 1 else kwargs.get("dims"))
        order = [int(d) % nd for d in (dims if isinstance(dims, (list, tuple, th.Size)) else [dims])]
    elif name in ("transpose", "transpose_", "swapaxes", "swapdims"):
        a = int(_arg(args, kwargs, 1, "dim0")) % nd
        b = int(_arg(args, kwargs, 2, "dim1")) % nd
        order[a], order[b] = order[b], order[a]
    elif name in ("t", "t_", "get:T"):
        order = order[::-1]
    elif name in ("get:mT", "adjoint", "get:mH"):
        order[-1], order[-2] = order[-2], order[-1]
    elif name in ("movedim", "moveaxis"):
        src = _dims(_arg(args, kwargs, 1, "source"), nd)
        dst = _dims(_arg(args, kwargs, 2, "destination"), nd)
        rest = [d for d in range(nd) if d not in src]
        order = [None] * nd
        for s, d in zip(src, dst):
            order[d] = s
        it = iter(rest)
        order = [next(it) if v is None else v for v in order]
    if name.endswith("_") and o is x:
        g._set(o, [tags[k] for k in order])
        return
    g._set(o, [tags[k] for k in order])


@_rule("__getitem__")
def _r_getitem(g, name, args, kwargs, out):
    x, index = args[0], args[1]
    o = next(_tensors_in(out), None)
    if o is None:
        return
    tags = _index_tags(g, x, index, name)
    g._set(o, tags if tags is not None and len(tags) == o.ndim else [None] * o.ndim)
    if tags is None or len(tags) != o.ndim:
        for ids in g._tag(x):
            g._pin(ids, "indexed in a way neuron edits do not follow")


def _index_tags(g, x, index, name) -> Optional[List[Ids]]:
    tags = g._tag(x)
    if not isinstance(index, tuple):
        index = (index,)
    n_used = sum(1 for i in index if i is not None and i is not Ellipsis
                 and not (isinstance(i, th.Tensor) and i.dtype == th.bool))
    n_used += sum(i.ndim for i in index if isinstance(i, th.Tensor) and i.dtype == th.bool)
    expanded = []
    for i in index:
        if i is Ellipsis:
            expanded.extend([slice(None)] * (x.ndim - n_used))
        else:
            expanded.append(i)
    while sum(1 for i in expanded if i is not None) < x.ndim:
        expanded.append(slice(None))
    advanced = [i for i in expanded if isinstance(i, (th.Tensor, list))]
    if len(advanced) > 1:
        return None
    out: List[Ids] = []
    d = 0
    for i in expanded:
        if i is None:
            out.append(None)
            continue
        ids = tags[d] if d < len(tags) else None
        if isinstance(i, (int, np.integer)) or (isinstance(i, th.Tensor) and i.ndim == 0 and i.dtype != th.bool):
            g._pin(ids, "a fixed position of these neurons is read")
            d += 1
            continue
        if isinstance(i, slice):
            size = int(x.shape[d])
            start, stop, step = i.indices(size)
            if ids is not None:
                if (start, stop, step) != (0, size, 1):
                    g._pin(ids, "a fixed slice of these neurons is read")
                out.append(ids[start:stop:step])
            else:
                out.append(None)
            d += 1
            continue
        if isinstance(i, (th.Tensor, list)):
            t = th.as_tensor(i)
            if t.dtype == th.bool:
                for k in range(t.ndim):
                    g._pin(tags[d + k] if d + k < len(tags) else None, "boolean mask indexing")
                out.append(None)
                d += t.ndim
                continue
            pos = t.detach().cpu().numpy().astype(np.int64)
            if ids is None:
                out.extend([None] * max(1, pos.ndim))
            else:
                sel = ids[pos % len(ids)]
                if sel.ndim <= 1:
                    out.append(sel.reshape(-1))
                else:
                    g._pin(ids, "multi-dimensional gather")
                    out.extend([None] * sel.ndim)
            d += 1
            continue
        return None
    return out


@_rule("__setitem__")
def _r_setitem(g, name, args, kwargs, out):
    x, index, value = args[0], args[1], args[2] if len(args) > 2 else None
    if not isinstance(x, th.Tensor):
        return
    view = _index_tags(g, x, index, name)
    if not isinstance(value, th.Tensor):
        return
    if view is None:
        for ids in g._tag(x) + g._tag(value):
            g._pin(ids, "written through an index neuron edits do not follow")
        return
    vt = g._tag(value)
    lead = len(view) - value.ndim
    for k in range(value.ndim):
        if lead + k >= 0 and int(value.shape[k]) > 1:
            g._couple(view[lead + k], vt[k], "written into a slice")


@_rule("narrow")
def _r_narrow(g, name, args, kwargs, out):
    x = args[0]
    d = int(_arg(args, kwargs, 1, "dim")) % x.ndim
    start = int(_arg(args, kwargs, 2, "start"))
    length = int(_arg(args, kwargs, 3, "length"))
    tags = list(g._tag(x))
    if tags[d] is not None:
        tags[d] = tags[d][start:start + length]
    g._set(next(_tensors_in(out)), tags)


@_rule("select")
def _r_select(g, name, args, kwargs, out):
    x = args[0]
    d = int(_arg(args, kwargs, 1, "dim")) % x.ndim
    tags = list(g._tag(x))
    g._set(next(_tensors_in(out)), tags[:d] + tags[d + 1:])


@_rule("index_select")
def _r_index_select(g, name, args, kwargs, out):
    x = args[0]
    d = int(_arg(args, kwargs, 1, "dim")) % x.ndim
    index = _arg(args, kwargs, 2, "index")
    tags = list(g._tag(x))
    if tags[d] is not None:
        tags[d] = tags[d][index.detach().cpu().numpy().astype(np.int64)]
    g._set(next(_tensors_in(out)), tags)


@_rule("split", "chunk", "tensor_split", "split_with_sizes", "unsafe_split", "unbind")
def _r_split(g, name, args, kwargs, out):
    x = args[0] if args and isinstance(args[0], th.Tensor) else kwargs.get("input")
    parts = [t for t in (out if isinstance(out, (list, tuple)) else [out]) if isinstance(t, th.Tensor)]
    if not isinstance(x, th.Tensor) or not parts:
        return
    if name == "unbind":
        d = int(_arg(args, kwargs, 1, "dim", 0)) % x.ndim
        tags = g._tag(x)
        g._pin(tags[d], "unbound into separate tensors")
        for p in parts:
            g._set(p, tags[:d] + tags[d + 1:])
        return
    default_dim = 0
    d = _arg(args, kwargs, 2, "dim", default_dim)
    d = int(d) % x.ndim
    tags = g._tag(x)
    ids = tags[d]
    sizes = [int(p.shape[d]) for p in parts]
    if ids is not None and len(parts) > 1 and len(set(sizes)) == 1 and sum(sizes) == len(ids):
        g._grid_on(ids, len(parts), f"`{name}` into {len(parts)} equal parts")
    offset = 0
    for p, s in zip(parts, sizes):
        new = list(tags)
        new[d] = None if ids is None else ids[offset:offset + s]
        g._set(p, new)
        offset += s


@_rule("cat", "concat", "concatenate", "hstack", "vstack")
def _r_cat(g, name, args, kwargs, out):
    seq = args[0] if args else kwargs.get("tensors")
    tensors = [t for t in seq if isinstance(t, th.Tensor)]
    o = next(_tensors_in(out), None)
    if o is None or not tensors:
        return
    d = int(_arg(args, kwargs, 1, "dim", 0)) % o.ndim
    tensors = [t for t in tensors if t.ndim == o.ndim]
    new: List[Ids] = [None] * o.ndim
    for k in range(o.ndim):
        if k == d:
            pieces, tagged = [], False
            for t in tensors:
                ids = g._tag(t)[k]
                if ids is None:
                    pieces.append(np.full(int(t.shape[k]), -1, dtype=np.int64))
                else:
                    tagged = True
                    pieces.append(ids)
            new[k] = np.concatenate(pieces) if tagged else None
        else:
            sub = [t for t in tensors]
            found = [g._tag(t)[k] for t in sub if g._tag(t)[k] is not None and int(t.shape[k]) == int(o.shape[k])]
            fixed = any(g._tag(t)[k] is None and int(t.shape[k]) > 1 for t in sub)
            for other in found[1:]:
                g._couple(found[0], other, f"`{name}` of two tensors")
            if found and fixed:
                g._pin(found[0], f"`{name}` with a tensor whose size cannot change")
            new[k] = found[0] if found else None
    g._set(o, new)


@_rule("stack")
def _r_stack(g, name, args, kwargs, out):
    seq = args[0] if args else kwargs.get("tensors")
    tensors = [t for t in seq if isinstance(t, th.Tensor)]
    o = next(_tensors_in(out), None)
    if o is None or not tensors:
        return
    d = int(_arg(args, kwargs, 1, "dim", 0)) % o.ndim
    inner = g._elementwise(tensors[0].shape, tensors, name)
    g._set(o, inner[:d] + [None] + inner[d:])


# --- layers -------------------------------------------------------------------
def _role(g, tensor: Optional[th.Tensor], dim: int, role: str) -> None:
    if isinstance(tensor, th.Tensor):
        space = g.space_of(tensor, dim)
        if space is not None:
            space.role = role


@_rule("linear")
def _r_linear(g, name, args, kwargs, out):
    x = _arg(args, kwargs, 0, "input")
    w = _arg(args, kwargs, 1, "weight")
    b = _arg(args, kwargs, 2, "bias")
    o = next(_tensors_in(out), None)
    if o is None:
        return
    xt, wt = g._tag(x), g._tag(w)
    g._couple(xt[-1], wt[1], "input of a linear layer")
    _role(g, w, 1, "in")
    if isinstance(b, th.Tensor):
        g._couple(wt[0], g._tag(b)[0], "bias")
    g._set(o, list(xt[:-1]) + [wt[0]])


@_rule("bilinear")
def _r_bilinear(g, name, args, kwargs, out):
    x1, x2, w = args[0], args[1], args[2]
    b = _arg(args, kwargs, 3, "bias")
    wt = g._tag(w)
    g._couple(g._tag(x1)[-1], wt[1], "bilinear input 1")
    g._couple(g._tag(x2)[-1], wt[2], "bilinear input 2")
    _role(g, w, 1, "in")
    _role(g, w, 2, "in")
    if isinstance(b, th.Tensor):
        g._couple(wt[0], g._tag(b)[0], "bias")
    o = next(_tensors_in(out))
    g._set(o, list(g._tag(x1)[:-1]) + [wt[0]])


def _conv_rule(transposed: bool):
    def rule(g, name, args, kwargs, out):
        x = _arg(args, kwargs, 0, "input")
        w = _arg(args, kwargs, 1, "weight")
        b = _arg(args, kwargs, 2, "bias")
        groups = int(_arg(args, kwargs, 6, "groups", 1) or 1)
        o = next(_tensors_in(out), None)
        if o is None:
            return
        xt, wt = g._tag(x), g._tag(w)
        batched = x.ndim == w.ndim
        cdim = 1 if batched else 0
        x_ch = xt[cdim]
        for k in range(2, w.ndim):
            g._pin(wt[k], "kernel size")
        for k in range(cdim + 1, x.ndim):
            g._pin(xt[k], "spatial size")
        in_dim, out_dim = (0, 1) if transposed else (1, 0)
        _role(g, w, in_dim, "in")
        c_in = int(x.shape[cdim])
        if groups == 1:
            g._couple(x_ch, wt[in_dim], "input of a convolution")
            out_ids = wt[out_dim]
        elif not transposed and int(w.shape[1]) == 1 and groups == c_in and int(w.shape[0]) == c_in:
            g._couple(x_ch, wt[0], "depthwise convolution: channel in = channel out")
            g._pin(wt[1], "depthwise kernel")
            out_ids = wt[0]
        elif transposed:
            for ids in (x_ch, wt[0], wt[1]):
                g._pin(ids, "grouped transposed convolution")
            out_ids = None
        else:
            # Grouped: a position goes with the same position in every group.
            if x_ch is not None and len(x_ch) % groups == 0 and len(x_ch) // groups > 1:
                cells = x_ch.reshape(groups, -1)
                inner = g._fresh(cells.shape[1], "conv groups")
                g.grids.append(_Grid(cells, [None, inner], "convolution groups"))
                g._couple(inner, wt[1], "grouped convolution input")
            else:
                g._pin(x_ch, "grouped convolution input")
                g._pin(wt[1], "grouped convolution input")
            out_ids = wt[0]
            g._grid_on(wt[0], groups, "convolution groups")
        if isinstance(b, th.Tensor):
            g._couple(out_ids, g._tag(b)[0], "bias")
        new: List[Ids] = [None] * o.ndim
        if batched:
            new[0] = xt[0]
        new[cdim] = out_ids
        g._set(o, new)
    return rule


for _n in ("conv1d", "conv2d", "conv3d"):
    _RULES[_n] = _conv_rule(False)
for _n in ("conv_transpose1d", "conv_transpose2d", "conv_transpose3d"):
    _RULES[_n] = _conv_rule(True)


@_rule("batch_norm", "instance_norm")
def _r_batch_norm(g, name, args, kwargs, out):
    x = _arg(args, kwargs, 0, "input")
    o = next(_tensors_in(out), None)
    if o is None:
        return
    xt = g._tag(x)
    ch = xt[1] if x.ndim > 1 else xt[0]
    names = ("running_mean", "running_var", "weight", "bias")
    for k, pname in enumerate(names):
        t = _arg(args, kwargs, k + 1, pname)
        if isinstance(t, th.Tensor) and t.ndim == 1:
            g._couple(ch, g._tag(t)[0], name)
    g._set(o, list(xt))


@_rule("group_norm")
def _r_group_norm(g, name, args, kwargs, out):
    x = _arg(args, kwargs, 0, "input")
    groups = int(_arg(args, kwargs, 1, "num_groups"))
    o = next(_tensors_in(out), None)
    xt = g._tag(x)
    ch = xt[1]
    for k, pname in ((2, "weight"), (3, "bias")):
        t = _arg(args, kwargs, k, pname)
        if isinstance(t, th.Tensor):
            g._couple(ch, g._tag(t)[0], name)
    g._grid_on(ch, groups, "group norm groups")
    if o is not None:
        g._set(o, list(xt))


@_rule("layer_norm", "rms_norm")
def _r_layer_norm(g, name, args, kwargs, out):
    x = _arg(args, kwargs, 0, "input")
    shape = _arg(args, kwargs, 1, "normalized_shape")
    w = _arg(args, kwargs, 2, "weight")
    b = _arg(args, kwargs, 3, "bias") if name == "layer_norm" else None
    o = next(_tensors_in(out), None)
    xt = g._tag(x)
    k = len(shape) if isinstance(shape, (list, tuple, th.Size)) else 1
    if not isinstance(w, th.Tensor):
        for d in range(x.ndim - k, x.ndim):
            g._pin(xt[d], f"{name} without weights (its shape is a constant)")
    for t in (w, b):
        if isinstance(t, th.Tensor):
            tt = g._tag(t)
            for j in range(k):
                g._couple(xt[x.ndim - k + j], tt[j], name)
    if o is not None:
        g._set(o, list(xt))


@_rule("prelu")
def _r_prelu(g, name, args, kwargs, out):
    x, w = args[0], args[1]
    o = next(_tensors_in(out), None)
    xt = g._tag(x)
    if isinstance(w, th.Tensor) and w.numel() > 1 and x.ndim > 1:
        g._couple(xt[1], g._tag(w)[0], name)
    if o is not None:
        g._set(o, list(xt))


@_rule("embedding")
def _r_embedding(g, name, args, kwargs, out):
    ids_t = _arg(args, kwargs, 0, "input")
    w = _arg(args, kwargs, 1, "weight")
    o = next(_tensors_in(out), None)
    wt = g._tag(w)
    g._pin(wt[0], "the vocabulary (rows looked up by id)")
    if o is not None:
        g._set(o, list(g._tag(ids_t)) + [wt[1]])


@_rule("one_hot")
def _r_one_hot(g, name, args, kwargs, out):
    x = args[0]
    o = next(_tensors_in(out), None)
    if o is not None:
        g._set(o, list(g._tag(x)) + [None])


def _matmul_tags(g, a, b, name) -> Optional[List[Ids]]:
    at, bt = list(g._tag(a)), list(g._tag(b))
    a_shape, b_shape = list(a.shape), list(b.shape)
    a_vec, b_vec = a.ndim == 1, b.ndim == 1
    if a_vec:
        at, a_shape = [None] + at, [1] + a_shape
    if b_vec:
        bt, b_shape = bt + [None], b_shape + [1]
    g._couple(at[-1], bt[-2], f"`{name}` contraction")
    _role(g, b, b.ndim - 2 if not b_vec else 0, "in")
    _role(g, a, a.ndim - 1, "in")
    # batch dims broadcast
    nb = max(len(a_shape), len(b_shape)) - 2
    batch: List[Ids] = []
    for d in range(nb):
        ka = d - (nb - (len(a_shape) - 2))
        kb = d - (nb - (len(b_shape) - 2))
        ia = at[ka] if ka >= 0 and a_shape[ka] > 1 else None
        ib = bt[kb] if kb >= 0 and b_shape[kb] > 1 else None
        if ia is not None and ib is not None:
            g._couple(ia, ib, f"`{name}` batch")
        elif (ia is None) != (ib is None):
            size_a = a_shape[ka] if ka >= 0 else 1
            size_b = b_shape[kb] if kb >= 0 else 1
            if size_a > 1 and size_b > 1:
                g._pin(ia if ia is not None else ib, f"`{name}` batch of fixed size")
        batch.append(ia if ia is not None else ib)
    tags = batch + [at[-2], bt[-1]]
    if a_vec:
        tags.pop(-2)
    if b_vec:
        tags.pop(-1)
    return tags


@_rule("matmul", "__matmul__", "__rmatmul__", "bmm", "mm", "mv")
def _r_matmul(g, name, args, kwargs, out):
    a, b = args[0], args[1]
    if name == "__rmatmul__":
        a, b = b, a
    o = next(_tensors_in(out), None)
    if o is None or not isinstance(a, th.Tensor) or not isinstance(b, th.Tensor):
        return
    tags = _matmul_tags(g, a, b, name)
    g._set(o, tags if tags is not None and len(tags) == o.ndim else [None] * o.ndim)


@_rule("baddbmm", "addmm", "addbmm", "addmv")
def _r_addmm(g, name, args, kwargs, out):
    base, a, b = args[0], args[1], args[2]
    o = next(_tensors_in(out), None)
    if o is None:
        return
    tags = _matmul_tags(g, a, b, name)
    tmp = th.empty(o.shape, device="meta")
    g._tags[id(tmp)] = (tmp, tags)
    g._set(o, g._elementwise(o.shape, [tmp, base], name))


@_rule("einsum")
def _r_einsum(g, name, args, kwargs, out):
    eq = args[0]
    ops = list(args[1:])
    if len(ops) == 1 and isinstance(ops[0], (list, tuple)):
        ops = list(ops[0])
    o = next(_tensors_in(out), None)
    if o is None or "..." in eq:
        return g._generic(name, args, kwargs, out)
    lhs, rhs = eq.replace(" ", "").split("->") if "->" in eq else (eq.replace(" ", ""), None)
    terms = lhs.split(",")
    seen: Dict[str, Ids] = {}
    for term, t in zip(terms, ops):
        tt = g._tag(t)
        for letter, ids, size in zip(term, tt, t.shape):
            if int(size) == 1:
                continue
            if letter in seen:
                g._couple(seen[letter], ids, f"einsum `{letter}`")
            else:
                seen[letter] = ids
    if rhs is None:
        rhs = "".join(sorted(c for c in set(lhs.replace(",", "")) if lhs.replace(",", "").count(c) == 1))
    g._set(o, [seen.get(c) for c in rhs])


@_rule("scaled_dot_product_attention")
def _r_sdpa(g, name, args, kwargs, out):
    q, k, v = args[0], args[1], args[2]
    mask = _arg(args, kwargs, 3, "attn_mask")
    o = next(_tensors_in(out), None)
    qt, kt, vt = g._tag(q), g._tag(k), g._tag(v)
    g._couple(qt[-1], kt[-1], "attention: query . key")
    g._couple(kt[-2], vt[-2], "attention: key / value positions")
    nb = q.ndim - 2
    for d in range(nb):
        for other, ot in ((k, kt), (v, vt)):
            kd = d - (nb - (other.ndim - 2))
            if kd >= 0 and int(other.shape[kd]) == int(q.shape[d]) and int(q.shape[d]) > 1:
                g._couple(qt[d], ot[kd], "attention heads")
    if isinstance(mask, th.Tensor):
        mt = g._tag(mask)
        target = list(qt[:-2]) + [qt[-2], kt[-2]]
        lead = len(target) - mask.ndim
        for j in range(mask.ndim):
            if lead + j >= 0 and int(mask.shape[j]) > 1:
                g._couple(target[lead + j], mt[j], "attention mask")
    if o is not None:
        g._set(o, list(qt[:-1]) + [vt[-1]])


def _spatial_rule(g, name, args, kwargs, out):
    x = _arg(args, kwargs, 0, "input")
    if not isinstance(x, th.Tensor):
        return
    xt = g._tag(x)
    for d in range(2, x.ndim):
        g._pin(xt[d], f"spatial size through `{name}`")
    for o in _tensors_in(out):
        new = [None] * o.ndim
        if o.ndim >= 2 and x.ndim >= 2:
            new[0], new[1] = xt[0], xt[1]
        g._set(o, new)


for _n in ("max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d",
           "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d", "adaptive_avg_pool1d",
           "adaptive_avg_pool2d", "adaptive_avg_pool3d", "lp_pool1d", "lp_pool2d", "lp_pool3d",
           "fractional_max_pool2d", "fractional_max_pool3d", "interpolate", "upsample",
           "upsample_nearest", "upsample_bilinear", "max_pool2d_with_indices", "max_pool1d_with_indices",
           "max_pool3d_with_indices", "adaptive_max_pool2d_with_indices", "grid_sample",
           "local_response_norm", "_max_pool2d", "_max_pool1d", "_max_pool3d"):
    _RULES[_n] = _spatial_rule


@_rule("pad", "_pad", "constant_pad_nd")
def _r_pad(g, name, args, kwargs, out):
    x = args[0]
    pad = list(_arg(args, kwargs, 1, "pad"))
    xt = list(g._tag(x))
    o = next(_tensors_in(out), None)
    for k in range(len(pad) // 2):
        d = x.ndim - 1 - k
        if pad[2 * k] or pad[2 * k + 1]:
            g._pin(xt[d], "padded")
            xt[d] = None
    if o is not None:
        g._set(o, xt)


@_rule("pixel_shuffle", "pixel_unshuffle", "channel_shuffle", "native_channel_shuffle", "roll",
       "flip", "fliplr", "flipud", "rot90", "sort", "topk", "argsort", "gather", "scatter",
       "scatter_add", "take", "take_along_dim", "masked_select", "nonzero", "unique",
       "unfold", "fold", "im2col", "col2im", "diagonal", "diag", "tril", "triu", "trace",
       "multi_head_attention_forward", "lstm", "gru", "rnn_tanh", "rnn_relu", "ctc_loss")
def _r_pin_all(g, name, args, kwargs, out):
    g.unknown_ops[name] = g.unknown_ops.get(name, 0) + 1
    for t in _tensors_in((args, kwargs)):
        for ids in g._tag(t):
            g._pin(ids, f"goes through `{name}`, which neuron edits do not handle")
    if name in ("tril", "triu", "flip", "roll") and isinstance(args[0], th.Tensor):
        for o in _tensors_in(out):
            g._set(o, list(g._tag(args[0])))


@_rule("zeros", "ones", "full", "empty", "rand", "randn", "randint", "arange", "linspace",
       "logspace", "eye", "tensor", "as_tensor", "from_numpy", "new_zeros", "new_ones",
       "new_full", "new_empty", "new_tensor", "zeros_like_untagged", "randperm", "multinomial",
       "bernoulli", "normal", "empty_strided", "scalar_tensor", "tril_indices", "triu_indices")
def _r_factory(g, name, args, kwargs, out):
    return


@_rule("zeros_like", "ones_like", "empty_like", "rand_like", "randn_like", "full_like", "randint_like")
def _r_like(g, name, args, kwargs, out):
    x = args[0]
    o = next(_tensors_in(out), None)
    if o is not None and isinstance(x, th.Tensor) and tuple(x.shape) == tuple(o.shape):
        g._set(o, list(g._tag(x)))


# =============================================================================
# Edits
# =============================================================================
def neuron_axis(module: nn.Module) -> int:
    """The axis of ``module.weight`` that lists its output neurons."""
    if isinstance(module, (nn.modules.conv._ConvTransposeNd, nn.Embedding)):
        return 1
    return 0


def _primary_name(module: nn.Module) -> str:
    w = getattr(module, "weight", None)
    if isinstance(w, th.Tensor) and w.ndim > 0:
        return "weight"
    for n, p in module.named_parameters(recurse=False):
        if p.ndim > 0:
            return n
    return ""


def _primary(module: nn.Module) -> Tuple[Optional[th.Tensor], int]:
    w = getattr(module, "weight", None)
    if isinstance(w, th.Tensor) and w.ndim > 0:
        return w, neuron_axis(module) if w.ndim > 1 else 0
    for p in module.parameters(recurse=False):
        if p.ndim > 0:
            return p, 0
    return None, 0


class NeuronEditor:
    """Prune / add neurons of a model, from its :class:`NeuronGraph`."""

    def __init__(self, model: nn.Module, example: Any):
        self.model = model
        self.example = example
        self._graph: Optional[NeuronGraph] = None
        self._signature = None
        self.graph   # trace now: a model it cannot run fails here, not at the first edit

    @property
    def graph(self) -> NeuronGraph:
        """The coupling graph of the model as it is now (re-traced when its
        parameters changed shape behind our back, e.g. a checkpoint restore)."""
        signature = _signature(self.model)
        if self._graph is None or signature != self._signature:
            self.example = _to_device(self.example, _device_of(self.model))
            self._graph = NeuronGraph.trace(self.model, self.example)
            self._signature = signature
        return self._graph

    @graph.setter
    def graph(self, value: NeuronGraph) -> None:
        self._graph = value
        self._signature = _signature(self.model)

    def __getstate__(self):
        # Checkpoints pickle the model with its editor: keep it lean, the
        # graph is re-traced on first use.
        state = dict(self.__dict__)
        state["_graph"] = None
        state["_signature"] = None
        return state

    # ---------------------------------------------------------------- queries
    def neuron_nodes(self, module: nn.Module) -> Optional[np.ndarray]:
        tensor, axis = _primary(module)
        if tensor is None:
            # A layer without weights (activation, pooling ...): the neurons
            # it passes on, i.e. its producer's.
            ids = self.graph.module_out.get(id(module))
            return None if ids is None else ids.copy()
        space = self.graph.space_of(tensor, axis)
        if space is None:
            return None
        return np.arange(space.offset, space.offset + space.size, dtype=np.int64)

    def pinned_reason(self, module: nn.Module) -> Optional[str]:
        """Why the neurons of *module* cannot be edited (None: they can)."""
        nodes = self.neuron_nodes(module)
        if nodes is None:
            return "this layer has no weights"
        g = self.graph
        reasons = [g._pinned_roots.get(int(g._roots[n])) for n in nodes.tolist()]
        if all(reasons):
            return reasons[0]
        return None

    def editable(self, module: nn.Module) -> bool:
        return self.pinned_reason(module) is None

    def coupled(self, module: nn.Module, indices: Sequence[int]) -> Dict[str, List[int]]:
        """``{"module.param[dim]": [indices]}`` going with *indices* of *module*."""
        nodes = self._closure(self._seed(module, indices), 0)
        return self._by_tensor(nodes)

    # --------------------------------------------------------------- closure
    def _seed(self, module, indices) -> List[int]:
        nodes = self.neuron_nodes(module)
        if nodes is None:
            raise NeuronEditError(f"{type(module).__name__} has no neurons to edit")
        out = []
        for i in indices:
            i = int(i)
            if not -len(nodes) <= i < len(nodes):
                raise NeuronEditError(f"neuron {i} out of range ({len(nodes)} neurons)")
            if nodes[i] < 0:
                raise NeuronEditError("these neurons cannot change: a dimension that cannot change")
            out.append(int(nodes[i]))
        return out

    def _closure(self, seed: Iterable[int], strategy: int) -> set:
        g = self.graph
        removed: set = set()
        queue = [int(g._roots[n]) for n in seed]
        done_roots: set = set()
        pending: List[Tuple[int, int]] = []
        while queue or pending:
            while queue:
                root = queue.pop()
                if root in done_roots:
                    continue
                done_roots.add(root)
                why = g._pinned_roots.get(root)
                if why:
                    raise NeuronEditError(f"these neurons cannot change: {why}")
                for node in g._members.get(root, [root]):
                    removed.add(node)
                    for gi, j, i in g._axis_of.get(node, ()):
                        grid = g.grids[gi]
                        plane = np.take(grid.cells, i, axis=j).ravel()
                        if (plane < 0).any():
                            raise NeuronEditError(
                                f"these neurons cannot change: {grid.why} (part of it has a fixed size)")
                        queue.extend(int(g._roots[c]) for c in plane.tolist())
                    pending.extend(g._cell_of.get(node, ()))
            still = []
            for gi, pos in pending:
                grid = g.grids[gi]
                coords = np.unravel_index(pos, grid.cells.shape)
                covered = any(a is not None and a[c] >= 0 and int(a[c]) in removed
                              for a, c in zip(grid.axes, coords))
                if covered:
                    continue
                choices = [(j, int(a[c])) for j, (a, c) in enumerate(zip(grid.axes, coords))
                           if a is not None and a[c] >= 0
                           and not g._pinned_roots.get(int(g._roots[int(a[c])]))]
                if not choices:
                    raise NeuronEditError(
                        f"these neurons cannot change: {grid.why} (no dimension of it can shrink)")
                sizes = [grid.cells.shape[j] for j, _ in choices]
                order = sorted(range(len(choices)), key=lambda k: (-sizes[k], -choices[k][0]))
                pick = choices[order[strategy % len(order)]]
                still.append(pick[1])
            pending = []
            queue.extend(int(g._roots[n]) for n in still)
        return removed

    def _shape_updates(self, nodes: set, op: str, count: int):
        out = []
        for mod, attr, tags, sizes in self.graph.shape_attrs:
            new = []
            for ids, size in zip(tags, sizes):
                if ids is None:
                    new.append(size)
                    continue
                hit = sum(1 for n in ids.tolist() if n in nodes)
                new.append(size - hit if op == "prune" else size + hit * count)
            if tuple(new) != tuple(sizes) and not any(o is mod and a == attr for o, a, _ in out):
                out.append((mod, attr, tuple(new)))
        return out

    def _by_tensor(self, nodes: Iterable[int]) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for space, idx in self._plan(nodes).items():
            out[space.name] = idx
        return out

    def _plan(self, nodes: Iterable[int]) -> Dict[_Space, List[int]]:
        g = self.graph
        offsets = np.array([s.offset for s in g.spaces], dtype=np.int64)
        plan: Dict[int, List[int]] = {}
        for node in sorted(set(int(n) for n in nodes)):
            k = int(np.searchsorted(offsets, node, side="right") - 1)
            space = g.spaces[k]
            if space.kind == "virtual":
                continue
            plan.setdefault(k, []).append(node - space.offset)
        return {g.spaces[k]: v for k, v in plan.items()}

    # ------------------------------------------------------------------ edits
    def prune(self, module: nn.Module, indices: Sequence[int]) -> EditResult:
        if self.neuron_nodes(module) is None:
            raise NeuronEditError(f"{type(module).__name__} has no neurons to edit")
        indices = sorted({int(i) % len(self.neuron_nodes(module)) for i in indices})
        if not indices:
            raise NeuronEditError("no neuron to prune")
        if len(indices) >= len(self.neuron_nodes(module)):
            raise NeuronEditError("cannot prune every neuron of a layer")
        return self._edit(module, "prune", self._seed(module, indices), count=len(indices))

    def add(self, module: nn.Module, count: int = 1) -> EditResult:
        count = int(count)
        if count < 1:
            raise NeuronEditError("count must be a positive integer")
        nodes = self.neuron_nodes(module)
        if nodes is None:
            raise NeuronEditError(f"{type(module).__name__} has no neurons to edit")
        return self._edit(module, "add", [int(nodes[-1])], count=count)

    def _edit(self, module, op: str, seed: List[int], count: int) -> EditResult:
        current = self.graph
        self.example = _to_device(self.example, _device_of(self.model))
        tried, failures = set(), []
        target, axis = _primary(module)
        for strategy in range(3):
            # Each strategy lays the grids (heads, groups ...) out differently.
            try:
                nodes = self._closure(seed, strategy)
            except NeuronEditError as exc:
                failures.append(exc)
                continue
            plan = self._plan(nodes)
            key = tuple(sorted((s.offset, tuple(v)) for s, v in plan.items()))
            if key in tried:
                continue
            tried.add(key)
            if op == "prune":
                own = set(self.neuron_nodes(module).tolist())
                if len(own & nodes) >= len(own):
                    failures.append(NeuronEditError(
                        "cannot prune every neuron of a layer (its coupled neurons -- heads, "
                        "groups -- would all go)"))
                    continue
            snapshot = _Snapshot(self.model, plan)
            try:
                for mod, attr, value in self._shape_updates(nodes, op, count):
                    snapshot.attrs.append((mod, {attr: getattr(mod, attr)}))
                    setattr(mod, attr, value)
                changed = _apply(self.model, plan, op, count)
                graph = NeuronGraph.trace(self.model, self.example)
                if graph.output_shapes != current.output_shapes:
                    raise NeuronEditError(
                        f"the model output changed shape ({current.output_shapes} -> {graph.output_shapes})")
            except Exception as exc:          # roll back, try the next layout
                snapshot.restore()
                failures.append(exc)
                logger.debug("neuron edit (%s, strategy %d) rolled back: %s", op, strategy, exc)
                continue
            self.graph = graph
            res = EditResult(op=op, changes=changed)
            res.layers = []
            for ch in changed:
                if ch.module_name not in res.layers:
                    res.layers.append(ch.module_name)
                n = len(ch.positions) if op == "prune" else sum(k for _, k in ch.inserts)
                res.total += n
                if ch.module is module and ch.name == _primary_name(module) and ch.dim == axis:
                    res.neurons = n
            if not res.neurons:          # a weightless layer: its producer's neurons
                res.neurons = len(seed) if op == "prune" else count
            return res
        # The most telling reason: the model failing to run, else a refusal.
        crash = next((e for e in failures if not isinstance(e, NeuronEditError)), None)
        if crash is not None:
            raise NeuronEditError(f"the model does not run after this {op} ({crash}); its forward "
                                  "probably hard-codes a size -- derive it from the tensors instead")
        raise failures[0] if failures else NeuronEditError(f"nothing to {op}")


# --- applying a plan ------------------------------------------------------------
def _owners(model: nn.Module) -> Dict[int, List[Tuple[nn.Module, str, str]]]:
    owners: Dict[int, List[Tuple[nn.Module, str, str]]] = {}
    for mod in model.modules():
        for pname, p in mod._parameters.items():
            if p is not None:
                owners.setdefault(id(p), []).append((mod, pname, "param"))
        for bname, b in mod._buffers.items():
            if b is not None:
                owners.setdefault(id(b), []).append((mod, bname, "buffer"))
    return owners


def _tensor_names(model: nn.Module) -> Dict[int, str]:
    names = {}
    for n, p in model.named_parameters(remove_duplicate=True):
        names[id(p)] = n
    for n, b in model.named_buffers(remove_duplicate=True):
        names[id(b)] = n
    return names


_SIZE_ATTRS = ("in_features", "out_features", "in_channels", "out_channels", "groups",
               "num_features", "num_channels", "normalized_shape", "num_embeddings",
               "embedding_dim", "num_parameters", "in_neurons", "out_neurons")


class _Snapshot:
    def __init__(self, model: nn.Module, plan: Dict[_Space, List[int]]):
        self.owners = _owners(model)
        self.tensors = {id(s.tensor): s.tensor for s in plan}
        self.old: Dict[int, th.Tensor] = {}
        self.attrs: List[Tuple[nn.Module, Dict[str, Any]]] = []
        mods = []
        for tid in self.tensors:
            for mod, _, _ in self.owners.get(tid, ()):
                if mod not in mods:
                    mods.append(mod)
        for mod in mods:
            self.attrs.append((mod, {a: getattr(mod, a) for a in _SIZE_ATTRS if a in mod.__dict__}))
            frozen = getattr(mod, "_wl_frozen_rows", None)
            if frozen is not None:
                frozen.restore(mod)

    def restore(self) -> None:
        for tid, tensor in self.tensors.items():
            for mod, name, kind in self.owners.get(tid, ()):
                (mod._parameters if kind == "param" else mod._buffers)[name] = tensor
        for mod, attrs in self.attrs:
            for a, v in attrs.items():
                setattr(mod, a, v)


def _insert_positions(positions: List[int], count: int) -> List[Tuple[int, int]]:
    """For each run of consecutive template positions, (insert at, how many)."""
    runs, start = [], None
    prev = None
    for p in sorted(positions):
        if start is None:
            start = prev = p
        elif p == prev + 1:
            prev = p
        else:
            runs.append((start, prev))
            start = prev = p
    if start is not None:
        runs.append((start, prev))
    return [(end + 1, (end - begin + 1) * count) for begin, end in runs]


def _grown(tensor: th.Tensor, dim: int, positions: List[int], count: int, role: str) -> th.Tensor:
    pieces, cursor = [], 0
    data = tensor.detach()
    for at, n in _insert_positions(positions, count):
        pieces.append(data.narrow(dim, cursor, at - cursor))
        shape = list(data.shape)
        shape[dim] = n
        pieces.append(_init_slice(data, dim, shape, role))
        cursor = at
    pieces.append(data.narrow(dim, cursor, data.shape[dim] - cursor))
    return th.cat(pieces, dim=dim)


def _init_slice(data: th.Tensor, dim: int, shape: List[int], role: str) -> th.Tensor:
    if not data.is_floating_point():
        return th.zeros(shape, dtype=data.dtype, device=data.device)
    if role == "in":                                   # read by nobody yet: the function is kept
        return th.zeros(shape, dtype=data.dtype, device=data.device)
    if data.ndim == 1:                                 # norms, biases, statistics: like the others
        return th.full(shape, float(data.float().mean()) if data.numel() else 0.0,
                       dtype=data.dtype, device=data.device)
    std = float(data.float().std()) if data.numel() > 1 else 0.02
    return th.randn(shape, dtype=data.dtype, device=data.device) * (std if np.isfinite(std) and std > 0 else 0.02)


def _apply(model: nn.Module, plan: Dict[_Space, List[int]], op: str, count: int) -> List[TensorChange]:
    owners = _owners(model)
    names = {id(m): n for n, m in model.named_modules()}
    by_tensor: Dict[int, List[Tuple[_Space, List[int]]]] = {}
    for space, idx in plan.items():
        by_tensor.setdefault(id(space.tensor), []).append((space, idx))
    changed: List[TensorChange] = []
    depthwise = {}
    for tid, entries in by_tensor.items():
        for mod, _, _ in owners.get(tid, ()):
            if isinstance(mod, nn.modules.conv._ConvNd) and not mod.transposed and \
                    mod.groups > 1 and mod.groups == mod.in_channels == mod.out_channels:
                depthwise[id(mod)] = True
    for tid, entries in by_tensor.items():
        tensor = entries[0][0].tensor
        new = tensor.detach()
        grad = tensor.grad.detach() if isinstance(tensor, th.Tensor) and tensor.grad is not None else None
        for space, idx in sorted(entries, key=lambda e: -e[0].dim):
            if op == "prune":
                gone = set(idx)
                keep = th.as_tensor([i for i in range(new.shape[space.dim]) if i not in gone],
                                    dtype=th.long, device=new.device)
                new = new.index_select(space.dim, keep)
                if grad is not None:
                    grad = grad.index_select(space.dim, keep)
            else:
                new = _grown(new, space.dim, idx, count, space.role)
                if grad is not None:
                    grad = _grown(grad, space.dim, idx, count, "in")
        own = owners.get(tid, ())
        if not own:
            continue
        if own[0][2] == "param":
            fresh = nn.Parameter(new.clone(), requires_grad=tensor.requires_grad)
            if grad is not None:
                fresh.grad = grad.clone()
        else:
            fresh = new.clone()
        for mod, name, kind in own:
            (mod._parameters if kind == "param" else mod._buffers)[name] = fresh
            label = names.get(id(mod), type(mod).__name__)
            for space, idx in entries:
                changed.append(TensorChange(
                    mod, label, name, space.dim,
                    positions=list(idx) if op == "prune" else [],
                    inserts=_insert_positions(idx, count) if op == "add" else []))
            _sync_attrs(mod, depthwise.get(id(mod), False))
    return changed


def _sync_attrs(mod: nn.Module, was_depthwise: bool = False) -> None:
    w = getattr(mod, "weight", None)
    w = w if isinstance(w, th.Tensor) else None
    if isinstance(mod, nn.Linear) and w is not None:
        mod.out_features, mod.in_features = int(w.shape[0]), int(w.shape[1])
    elif isinstance(mod, nn.modules.conv._ConvNd) and w is not None:
        if was_depthwise:
            mod.groups = int(w.shape[0])
        if mod.transposed:
            mod.in_channels, mod.out_channels = int(w.shape[0]), int(w.shape[1]) * mod.groups
        else:
            mod.out_channels, mod.in_channels = int(w.shape[0]), int(w.shape[1]) * mod.groups
    elif isinstance(mod, nn.modules.batchnorm._NormBase):
        ref = w if w is not None else getattr(mod, "running_mean", None)
        if isinstance(ref, th.Tensor):
            mod.num_features = int(ref.shape[0])
    elif isinstance(mod, nn.GroupNorm) and w is not None:
        mod.num_channels = int(w.shape[0])
    elif isinstance(mod, nn.LayerNorm) and w is not None:
        mod.normalized_shape = tuple(int(s) for s in w.shape)
    elif isinstance(mod, nn.Embedding) and w is not None:
        mod.num_embeddings, mod.embedding_dim = int(w.shape[0]), int(w.shape[1])
    elif isinstance(mod, nn.PReLU) and w is not None:
        mod.num_parameters = int(w.numel())
    elif isinstance(mod, nn.Bilinear) and w is not None:
        mod.out_features, mod.in1_features, mod.in2_features = map(int, w.shape)
    # WeightsLab's per-layer bookkeeping (in / out neuron counts).
    in_name, out_name = getattr(mod, "super_in_name", None), getattr(mod, "super_out_name", None)
    if out_name and hasattr(mod, out_name) and "out_neurons" in mod.__dict__:
        mod.out_neurons = getattr(mod, out_name)
    if in_name and hasattr(mod, in_name) and "in_neurons" in mod.__dict__:
        mod.in_neurons = getattr(mod, in_name)
