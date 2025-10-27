import torch
import torch.fx
import traceback

from torch._dispatch.python import enable_python_dispatcher
from torch.fx.node import Node, map_aggregate
from typing import Any, Tuple, NamedTuple, Optional, Dict, Any, List
from torch.fx._compatibility import compatibility
from torch._guards import detect_fake_mode


__all__ = ['TensorMetadata', 'ShapeProp']


@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    # TensorMetadata is a structure containing pertinent information
    # about a tensor within a PyTorch program.

    # General Tensor metadata
    shape: List
    dtype: torch.dtype
    requires_grad: bool
    stride: Tuple[int, ...]
    memory_format: Optional[torch.memory_format]

    # Quantization metadata
    is_quantized: bool
    qparams: Dict[str, Any]


def _extract_tensor_metadata(result: torch.Tensor) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride()

    memory_formats = {
        torch.contiguous_format,
        torch.channels_last,
        torch.channels_last_3d,
    }

    memory_format = None

    for query_format in memory_formats:
        if result.is_contiguous(memory_format=query_format):
            memory_format = query_format
            break

    is_quantized = result.is_quantized
    qparams: Dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            # type: ignore[assignment]
            qparams["scale"] = result.q_scale()
            # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()
        elif qscheme in {
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
            torch.per_channel_symmetric
        }:
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream

            # type: ignore[assignment]
            qparams["scale"] = result.q_per_channel_scales().tolist()
            # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()
            # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()

    return TensorMetadata(
        shape,
        dtype,
        requires_grad,
        stride,
        memory_format,
        is_quantized,
        qparams
    )


@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``torch.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = torch.fx.symbolic_trace(model)
        sample_input = torch.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape)

        The output of this code is:

        x torch.float32 torch.Size([50, 1000])
        linear1 torch.float32 torch.Size([50, 100])
        clamp_1 torch.float32 torch.Size([50, 100])
        linear2 torch.float32 torch.Size([50, 10])
        output torch.float32 torch.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed
         fake_mode (FakeTensorMode): A fake mode for copying the gm

    """
    def __init__(self, gm, fake_mode=None):
        super().__init__(gm)
        self.model = gm

        if fake_mode is None:
            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor
            # Note:
            # We need fake execution cause the inputs are fake, however, we
            # cannot fakify the module
            # - because we need to write to the tensor_meta of the real module.
            # So we fakify to
            # produce a result (L131 below), to extract tensor meta, and then
            # keep going.
            #
            # If we were to fakify, we would write to the wrong node, and then
            # downstream fusion would be missing the tensor_meta.
            #
            # See torch/_inductor/overrides.py for where this is called
            # upstream of fusion.
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None

        self.real_module = self.module

    def run_node(self, n: Node) -> Any:
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with "
                f"meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['tensor_meta'] = meta

        n.meta['type'] = type(result)
        return result

    def propagate(self, *args):
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        if self.fake_mode is not None:
            fake_args = [self.fake_mode.from_tensor(t) if isinstance(
                t,
                torch.Tensor
            ) else t for t in args]
        else:
            fake_args = args
        return super().run(*fake_args)


if __name__ == "__main__":
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            super().__init__()
            # --- Layer 1: Conv2d (3 -> 16 channels) ---
            # Output spatial size is maintained (32x32 -> 32x32)
            self.conv1 = torch.nn.Conv2d(in_channels=D_in, out_channels=16,
                                         kernel_size=3, padding=1)

            # --- Layer 2: Conv2d (16 -> 32 channels) ---
            # Spatial size is halved (32x32 -> 16x16)
            self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                         kernel_size=3, stride=2, padding=1)

            self.f = torch.nn.Flatten()
            self.linear1 = torch.nn.Linear(512, 128)
            self.linear2 = torch.nn.Linear(128, 8192)

            # --- Unflattening (Required before final Conv2d) ---
            # Reshapes the 8192 tensor back to (32, 16, 16)
            # dim=1 means we are reshaping starting from the channel dimension
            self.unflatten = torch.nn.Unflatten(
                dim=1,
                unflattened_size=(32, 16, 16)
            )

            # --- Layer 5: Conv2d (32 -> 10 channels) ---
            # Final convolution layer for 10 classes
            self.conv3 = torch.nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=1
            )

        def forward(self, x):
            x = self.f(self.conv2(self.conv1(x)))
            h_relu = self.linear1(x).clamp(min=0)
            x = self.linear2(h_relu)
            y_pred = self.conv3(self.unflatten(x))

            return y_pred

    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    dummy_input = torch.randn(1, D_in, 8, 8)
    model = TwoLayerNet(D_in, H, D_out)
    model(dummy_input)
    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(dummy_input)

    for node in gm.graph.nodes:
        print(
            node.name,
            node.meta['tensor_meta'].dtype,
            node.meta['tensor_meta'].shape
        )

    print('Bye World')
