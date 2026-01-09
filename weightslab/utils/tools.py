import io
import xxhash
import types
import logging
import collections
import torch as th
import torch.nn as nn
import numpy as np
import random

from typing import Union
from copy import deepcopy
from typing import List, Union


# Global logger
logger = logging.getLogger(__name__)


def seed_everything(seed):
    """Seed everything for reproducibility."""
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
# -------------------------- Utils Functions ---------------------------------
# ----------------------------------------------------------------------------
def extract_in_out_params(module: nn.Module) -> List[int | str]:
    """
    Detects and returns the primary input and output dimension parameters
    for a given PyTorch module instance, based on commmon templates.
    For single weight tensor (e.g., nn.BatchNorm), in=out.
    """

    # 1. Like Linear Layers use 'features' template
    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
        in_dim = module.in_features
        in_name = "in_features"
        out_dim = module.out_features
        out_name = "out_features"
        return in_dim, out_dim, in_name, out_name

    # 2. Like Convolutional Layers (Conv1d, Conv2d, Conv3d) use 'channels'
    # template
    if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
        in_dim = module.in_channels
        in_name = "in_channels"
        out_dim = module.out_channels
        out_name = "out_channels"
        # TODO (GP): Hardcoded for now, but should be wrapped somehow, i.e.,
        # TODO (GP): you customize how you define layers flag like transposed.
        if 'transposed' in module._get_name():
            module.wl_transposed = True
        return in_dim, out_dim, in_name, out_name

    # 3. Like BatchNorm Layers use 'num_features' template
    if hasattr(module, 'num_features'):
        # For BatchNorm, in_dim and out_dim are the same
        in_dim = module.num_features
        in_name = "num_features"
        out_dim = module.num_features
        out_name = "num_features"
        module.wl_same_flag = True
        return in_dim, out_dim, in_name, out_name

    # 4. Layers using in or out shape/size attributes
    shape_attrs = [
        i for i in list(module.__dict__.keys())
        if '_size' in i or '_shape' in i
    ]
    if len(shape_attrs) and 'flatten' in module._get_name():
        in_shape_attrs = [attr for attr in shape_attrs if 'in_' in attr]
        out_shape_attrs = [attr for attr in shape_attrs if 'in_' in attr]
        # OneWay layers, e.g., UnFlatten or BatchNorm layers
        if not len(in_shape_attrs) or not len(out_shape_attrs):
            in_dim = getattr(module, shape_attrs[0])
            in_name = shape_attrs[0]
            out_dim = getattr(module, shape_attrs[0])
            out_name = shape_attrs[0]
        elif len(in_shape_attrs) == len(out_shape_attrs):
            in_dim = getattr(module, in_shape_attrs[0])
            in_name = in_shape_attrs[0]
            out_dim = getattr(module, out_shape_attrs[0])
            out_name = out_shape_attrs[0]
        module.wl_same_flag = True
        return in_dim, out_dim, in_name, out_name

    # 5. Catch all or return None for non-parameterized layers
    return None, None, None, None


def get_children(module: nn.Module):
    """
        Get module children (other modules).
    """
    # Return the current module, e.g., conv2d_with_ops
    if is_module_with_ops(module):
        return [module]

    # If current module is not with_ops, i.e., not learnable, i.e., maxpool2d
    # Find next deps with_ops
    flatt_children = []
    for child in module.children():
        flatt_children.extend(get_children(child))

    return flatt_children


def get_module_device(module: nn.Module) -> th.device:
    """
    Retrieves the device (CPU or CUDA) of a th.nn.Module.
    """
    # Use next(module.parameters()) to get the first parameter tensor
    try:
        # Check the device of the first parameter found
        return next(module.parameters()).device
    except StopIteration:
        # If the module has no parameters (e.g., nn.ReLU, nn.Sequential,
        # or containers),
        # it doesn't have a device of its own. It defaults to the CPU.
        # This is the safest fallback, though you might need context for
        # the exact device.
        return th.device("cpu")


def rename_with_ops(module: nn.Module) -> nn.Module:
    """
        Add WithNeuronOps string to each nn.module name.
    """
    # 1. Store the original class name
    original_name = module._get_name()

    # 2. Define the new name
    new_name = f"{original_name}WithNeuronOps" if "WithNeuronOps" not in original_name \
        else original_name

    # 3. Create a custom method to return the new name
    def new_get_name(self):
        return new_name

    # 4. Monkey patch the module's _get_name() method
    # This is the function PyTorch calls when printing the model hierarchy.
    module._get_name = types.MethodType(new_get_name, module)


def is_module_with_ops(module: nn.Module) -> bool:
    return "WithNeuronOps" in module._get_name()


# Helper to retrieve module instance by its submodule path
def get_module_by_name(model: nn.Module, name: str) -> nn.Module | None:
    """
        Safely retrieves a module instance from the model based on its FX
        target name.
    """
    try:
        return model.get_submodule(name)
    except AttributeError:
        return getattr(model, name, None)


def what_layer_type(module: nn.Module):
    in_attrs = [i for i in list(module.__dict__.keys()) if 'in_' in i]
    out_attrs = [i for i in list(module.__dict__.keys()) if 'out_' in i]
    shape_attrs = [
        i for i in list(module.__dict__.keys())
        if 'size_' in i or '_size' in i or
        'shape_' in i or '_shape' in i
    ]

    # Find and return layer type based on the attributes found
    if len(in_attrs) and len(out_attrs):
        return 1
    elif len(shape_attrs):
        return 2
    else:
        return 0


def make_safelist(x):
    return [x] if not isinstance(x, list) else x


def model_op_neurons(model, layer_id=None, dummy_input=None, op=None, rand=False):
    """
        Test function to iteratively update neurons for each layer,
        then test inference. Everything match ?
    """
    seed_everything(42) if rand else None  # Set seed for reproducibility
    n_layers = len(model.layers)
    for n in range(n_layers-1, 0, -1):
        if rand and th.rand(1) > 0.5 and layer_id is None and dummy_input is None:
            continue
        if layer_id is not None:
            if layer_id >= 0:
                if n != layer_id:
                    continue
            else:
                if n != n_layers + layer_id:  # - -layer_id != + -layer_id
                    continue
        logger.debug(f'\nOperate on neurons at layer {n}')
        if op is None:
            with model as m:
                logger.debug('Adding operation - 5 neurons added.')
                m.operate(n, {0, 1}, op_type=1)
                m(dummy_input) if dummy_input is not None else None
            with model as m:
                logger.debug('Reseting operation - every neurons reset.')
                m.operate(n, {}, op_type=4)
                m(dummy_input) if dummy_input is not None else None
            with model as m:
                logger.debug('Freezing operation - last neuron froze.')
                m.operate(n, {-1}, op_type=3)
                m(dummy_input) if dummy_input is not None else None
            with model as m:
                logger.debug('Pruning operation - first neuron removed.')
                m.operate(n, {0}, op_type=2)
                m(dummy_input) if dummy_input is not None else None
        else:
            with model as m:
                m.operate(
                    n,
                    {-1},
                    op_type=op
                )
                m(dummy_input) if dummy_input is not None else None


def reindex_and_compress_blocks(data_dict, block_size, offset_index=0):
    """
    Re-indexes the dictionary keys and shifts the neuron value ranges to ensure
    they remain contiguous starting from 0, after removing an intermediate
    block.

    Args:
        data_dict (dict): The dictionary with non-contiguous keys and value
        ranges.
        block_size (int): The fixed size of each neuron block (e.g., 256).

    Returns:
        dict: The re-indexed dictionary with contiguous keys and compressed
        values.
    """
    # 1. Sort the remaining blocks by their original keys to maintain order
    # The dictionary keys must be sorted to ensure the blocks are processed
    # sequentially.
    sorted_blocks = collections.OrderedDict(sorted(data_dict.items()))

    reindexed_dict = {}

    # 2. Iterate through the remaining blocks, assigning a new contiguous index
    for new_index in range(len(list(sorted_blocks.items()))):
        new_index = new_index + offset_index
        index_batch = new_index // block_size
        # Calculate the new starting point for the range.
        # This point ensures the range is contiguous (0 * size, n * size, ...)
        new_start = block_size*index_batch
        new_end = new_start + block_size

        # Create the new contiguous range
        new_range = list(range(new_start, new_end))

        # Assign the new key and the compressed value range
        reindexed_dict[new_index] = new_range

    return reindexed_dict


def get_layer_trainable_parameters_neuronwise(layer: th.nn.Module):
    """
        Count the number of neurons with associated lr != 0.
    """
    # TODO (GP) Review function; seems like not working as expected with conv.
    # TODO (GP) when having kernel size (counts now only in out params. wo.
    # TODO (GP) corr. to kernel weights).
    trainable_params = 0
    for learnable_tensor_name in layer.learnable_tensors_name:
        trainable_params += getattr(layer, learnable_tensor_name).numel()
        trainable_params -= len(
            layer.neuron_2_lr[
                learnable_tensor_name
            ]
        )
        if learnable_tensor_name in layer.incoming_neuron_2_lr:
            trainable_params -= len(
                layer.incoming_neuron_2_lr[
                    learnable_tensor_name
                ]
            )
    return trainable_params


def normalize_dicts(a):
    offset_index = 0
    for deps_name_ in a:
        if len(a[deps_name_]) == 0:
            continue
        channel_size = len(
            list(
                a[
                    deps_name_
                ].values()
            )[-1]
        )
        # if dict has several items that are bypass of the module,
        # we split the new neurons between the two inputs channels
        # from the two input tensors, and so re index every neurons
        # with unique sequential indexs.
        if offset_index > 0:
            tmp_ = deepcopy(a[deps_name_])
            a[deps_name_].clear()
            for k in range(len(tmp_.items())):
                a[deps_name_][
                    k + offset_index
                ] = [k + offset_index]
        a[deps_name_] = reindex_and_compress_blocks(
            a[deps_name_],
            channel_size,
            offset_index=offset_index
        )
        indexs = list(a[
            deps_name_
        ].keys())
        offset_index += len(
            indexs
        ) if len(indexs) else 0
    return a


def reversing_indices(n_neurons, indices_set):
    """
        Reverse index from -x to x, given a set of indices, and sort them
        from higher to lower.

        Args:
            n_neurons (int): The total number of neurons.
            indices_set (Set[int]): The indices to reverse.
        Returns:
            List[int]: The reversed indices.

        Example:
        >>> reversing_indices(10, {-3, -5, 8})
        [7, 5, 8]
    """
    return sorted(
        {
            neg_idx for i in indices_set
            if -n_neurons <= (
                neg_idx :=
                (i if i < 0 else -(n_neurons - i))
            ) <= -1
        }
    )[::-1]


def _npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize array to .npy in memory (deterministic, includes dtype+shape)."""
    buf = io.BytesIO()
    # allow_pickle=False to avoid pickle metadata and ensure deterministic representation
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _canonical_raw_bytes(arr: np.ndarray) -> bytes:
    """Produce a canonical raw-bytes representation:
       - C contiguous
       - native endianness
       - dtype+shape are NOT included (caller must include them if needed)
       - does not canonicalize NaN bit patterns (only semantic NaNs),
         so prefer npy serialization for full safety.
    """
    a = np.ascontiguousarray(arr)
    # make sure dtype is native-endian
    if a.dtype.byteorder not in ('=', '|'):
        a = a.byteswap().newbyteorder()
    return a.tobytes()


def array_id_2bytes(
    arr: np.ndarray,
    *,
    include_shape_dtype: bool = True,
    use_npy_serialization: bool = True,
    return_hex: bool = False,
    tronc_1byte: bool = True,
    hex_upper: bool = False,
) -> Union[str, int, bytes]:
    """
    Generate an 8-byte ID for a numpy array.

    Parameters:
    - arr: numpy array
    - method:
        - "sha256" or "sha256-trunc": compute SHA-256 and take first 8 bytes.
        - "xxh64": use xxhash.xxh64 (faster, non-crypto) if available.
    - include_shape_dtype: if True, include shape and dtype in the hashed input
      (recommended so arrays with same raw bytes but different shape/dtype are distinct).
    - use_npy_serialization: if True, use np.save(.npy) serialization as the input to hash.
      If False, hash raw canonical bytes (faster, but less safe in some edge cases).
    - return_hex: return 16-char hex string if True, else returns int.
    - hex_upper: if True and return_hex True, hex will be uppercase.

    Returns:
    - 16-char hex-string (default) or integer (0..2**64-1) or raw bytes if you change code.
    """
    if use_npy_serialization:
        data = _npy_bytes(arr)
    else:
        parts = []
        if include_shape_dtype:
            # Include a stable textual header for shape and dtype to avoid ambiguity
            parts.append(f"{arr.shape};{str(arr.dtype)};".encode("utf-8"))
        parts.append(_canonical_raw_bytes(arr))
        data = b"".join(parts)

    h = xxhash.xxh64()
    h.update(data)
    digest8 = h.digest()  # 8 bytes

    if return_hex:
        hexs = digest8.hex()
        return hexs.upper() if hex_upper else hexs
    else:
        # big-endian integer
        if tronc_1byte:
            return int.from_bytes(digest8, byteorder="big", signed=False) % (10**8)
        else:
            return int.from_bytes(digest8, byteorder="big", signed=False)
