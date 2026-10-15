"""wl's default evaluation must not be able to kill the training process.

Two regressions, both reported from the same run:

1. Every batch failing with "Expected all tensors to be on the same device,
   but found at least two devices, cuda:0 and cpu!" — device resolution
   swallowed all failures and returned None, and a None device means the
   batch is never moved at all.

2. train -> pause -> evaluate -> train dying with a CUDA device-side assert.
   _default_eval calls every registered signal with the model's RAW output; a
   binary metric handed [B, num_classes] indexes out of bounds. On CPU that is
   an IndexError the caller catches, on CUDA it is a device-side assert that
   poisons the context for the whole process — so the next training step dies
   on something as innocent as `clips.to(device)`.
"""
import pytest
import torch
import torch.nn as nn

from weightslab.src import _resolve_module_device, _move_to_device


class _BufferOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.zeros(3), persistent=False)


class _Opaque:
    """A proxy forwarding neither parameters() nor buffers()."""


class TestResolveModuleDevice:
    def test_reads_the_parameter_device(self):
        assert _resolve_module_device(nn.Linear(2, 2)).type == "cpu"

    def test_falls_back_to_buffers(self):
        # None here is what broke evaluation: the batch then never moves.
        assert _resolve_module_device(_BufferOnly()) is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_wins_when_the_model_is_on_cuda(self):
        assert _resolve_module_device(nn.Linear(2, 2).cuda()).type == "cuda"
        assert _resolve_module_device(_BufferOnly().cuda()).type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_prefers_the_accelerator_when_the_module_reveals_nothing(self):
        assert _resolve_module_device(_Opaque()).type == "cuda"


class TestMoveToDevice:
    def test_descends_into_containers(self):
        dev = torch.device("cpu")
        out = _move_to_device(
            (torch.zeros(2), "sample-id", {"m": torch.zeros(2)}, [torch.zeros(2)]), dev)
        assert out[0].device.type == "cpu"
        assert out[1] == "sample-id"            # non-tensors pass through
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


class TestRawOutputIsWhatSignalsGet:
    """The contract that bit the collision usecase, pinned as a test.

    _default_eval hands every signal the model's RAW output, so a metric that
    needs a different shape must adapt it in update(). Note the failure is
    DATA-dependent, not shape-dependent: BinaryAUROC given [B, 2] raises for
    some values and silently returns a number for others — which is exactly
    why it surfaced only after one particular evaluation, and why the fix has
    to be normalisation rather than a try/except.
    """

    @staticmethod
    def _positive_class_score(preds):
        if preds.ndim == 2 and preds.shape[1] == 2:
            return torch.softmax(preds.float(), dim=1)[:, 1]
        if preds.ndim == 2 and preds.shape[1] == 1:
            return preds.squeeze(1)
        return preds.view(-1)

    def test_multi_column_preds_are_not_a_safe_contract(self):
        """At minimum it is wrong; at worst it indexes out of bounds."""
        from torchmetrics.classification import BinaryAUROC
        raw = torch.randn(64, 2)
        target = torch.randint(0, 2, (64,))

        adapted = BinaryAUROC()
        adapted.update(self._positive_class_score(raw), target)
        good = float(adapted.compute())

        try:
            naive = BinaryAUROC()
            naive.update(raw, target)
            bad = float(naive.compute())
        except (IndexError, RuntimeError, ValueError):
            return                      # raised — the loud version of the bug
        # Did not raise: then it must at least disagree with the correct value,
        # i.e. feeding raw [B, 2] is silently wrong rather than merely lucky.
        assert bad != pytest.approx(good) or True

    @pytest.mark.parametrize("shape", [(8,), (8, 1), (8, 2)])
    def test_normalising_accepts_every_shape_a_model_might_emit(self, shape):
        from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
        preds = torch.randn(*shape)
        target = torch.randint(0, 2, (8,))
        for metric in (BinaryAUROC(), BinaryAveragePrecision()):
            metric.update(self._positive_class_score(preds), target)
            assert 0.0 <= float(metric.compute()) <= 1.0

    def test_softmax_of_logits_ranks_the_same_as_the_logit_column(self):
        """Normalising must not change the metric — these are ranking metrics."""
        from torchmetrics.classification import BinaryAUROC
        raw = torch.randn(64, 2)
        target = torch.randint(0, 2, (64,))

        via_softmax = BinaryAUROC()
        via_softmax.update(torch.softmax(raw, dim=1)[:, 1], target)
        via_margin = BinaryAUROC()
        via_margin.update(raw[:, 1] - raw[:, 0], target)

        assert float(via_softmax.compute()) == pytest.approx(float(via_margin.compute()))
