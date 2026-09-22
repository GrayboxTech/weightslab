"""wl's default evaluation must put the batch where the model is.

Regression for: every batch of an on-demand evaluation failing with
"Expected all tensors to be on the same device, but found at least two
devices, cuda:0 and cpu!", repeated once per batch as a bare debug line.

The cause was device resolution that swallowed every failure and returned
None — and a None device meant the batch was never moved at all.
"""
import pytest
import torch
import torch.nn as nn

from weightslab.src import _resolve_module_device, _move_to_device


class _BufferOnly(nn.Module):
    """A module with buffers but no parameters — parameters() is empty."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.zeros(3), persistent=False)


class _NoTensors(nn.Module):
    pass


class _Opaque:
    """A proxy that forwards neither parameters() nor buffers()."""


class TestResolveModuleDevice:
    def test_reads_the_parameter_device(self):
        assert _resolve_module_device(nn.Linear(2, 2)).type == "cpu"

    def test_falls_back_to_buffers_when_there_are_no_parameters(self):
        # A None here is what broke evaluation: the batch then never moves.
        assert _resolve_module_device(_BufferOnly()) is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_wins_when_the_model_is_on_cuda(self):
        assert _resolve_module_device(nn.Linear(2, 2).cuda()).type == "cuda"
        assert _resolve_module_device(_BufferOnly().cuda()).type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_prefers_the_accelerator_when_the_module_reveals_nothing(self):
        # Better to send the batch to CUDA than to leave it on the CPU and
        # meet a CUDA model there.
        assert _resolve_module_device(_NoTensors()).type == "cuda"
        assert _resolve_module_device(_Opaque()).type == "cuda"


class TestMoveToDevice:
    def test_moves_a_plain_tensor(self):
        assert _move_to_device(torch.zeros(2), torch.device("cpu")).device.type == "cpu"

    def test_descends_into_tuples_lists_and_dicts(self):
        dev = torch.device("cpu")
        payload = (torch.zeros(2), "sample-id", {"m": torch.zeros(2)}, [torch.zeros(2)])
        out = _move_to_device(payload, dev)
        assert out[0].device.type == "cpu"
        assert out[1] == "sample-id"          # non-tensors pass through
        assert out[2]["m"].device.type == "cpu"
        assert out[3][0].device.type == "cpu"

    def test_preserves_container_types(self):
        dev = torch.device("cpu")
        assert isinstance(_move_to_device([torch.zeros(2)], dev), list)
        assert isinstance(_move_to_device((torch.zeros(2),), dev), tuple)

    def test_none_device_and_none_value_are_no_ops(self):
        t = torch.zeros(2)
        assert _move_to_device(t, None) is t
        assert _move_to_device(None, torch.device("cpu")) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_cpu_batch_reaches_a_cuda_model(self):
        """The end-to-end shape of the original failure."""
        model = nn.Linear(4, 2).cuda()
        batch = (torch.zeros(3, 4), ["id0", "id1", "id2"], torch.zeros(3))
        device = _resolve_module_device(model)
        inputs = _move_to_device(batch[0], device)
        assert model(inputs).shape == (3, 2)
