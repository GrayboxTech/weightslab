# ViT model-editing experiment

This experiment tests WeightsLab model editing instead of assuming that an
operation succeeded because the API returned without raising.

"ViT B-12" is interpreted here as **ViT-Base with 12 transformer encoder
blocks**. The torchvision implementation is `vit_b_16`: Base width/depth with
16x16 input patches.

## What is tested

There are two separate scripts because they answer different questions.

`probe_full_vit.py` is the compatibility gate for wrapping and editing the
complete ViT-B/16 model. It verifies:

1. Ordinary ViT forward/backward training works.
2. WeightsLab can build its editable dependency graph.
3. A classifier-head freeze can be applied.
4. Training can resume after the edit.

The current implementation is expected to stop at step 2. Torch FX traces the
model, but WeightsLab dependency mapping reaches `MultiheadAttention`, which
does not expose the neuron-operation interface expected by
`generate_index_maps`. The JSON report records the exact exception. Run without
`--allow-unsupported` when using this as a CI gate.

`run_head_experiment.py` is the working control experiment. It:

1. Builds a real 12-block ViT-B/16 backbone.
2. Generates deterministic patterned images without a network download.
3. Caches frozen CLS embeddings from the backbone.
4. Trains a WeightsLab-wrapped MLP classification head.
5. Adds hidden neurons and checks that the downstream input shape changes.
6. Checks that the optimizer is rebuilt with the new parameters.
7. Freezes a neuron and asserts that its gradient is zero.
8. Unfreezes it and asserts that its gradient becomes non-zero.
9. Resumes training and writes all graph, loss, and assertion results to JSON.

This control is intentionally limited to the head. Editing the 768-dimensional
transformer representation would require synchronized changes across attention
projections, residual paths, positional embeddings, and LayerNorm parameters;
the current dependency engine does not model those relationships reliably.

## Setup

From the repository root:

```bash
bash weightslab/examples/PyTorch/model-editing/vit-model-editing/setup.sh
```

The default environment is `.venv-vit-edit`. Override it with
`VIT_EDIT_VENV=/absolute/path` if needed.

## Run both experiments

```bash
bash weightslab/examples/PyTorch/model-editing/vit-model-editing/run_all.sh
```

Reports are written under `outputs/vit_model_editing/`, which is gitignored:

- `full_vit_compatibility_report.json`
- `vit_head_editing_report.json`

The combined runner allows the known full-ViT incompatibility so the supported
head experiment still runs. For a strict full-model gate:

```bash
.venv-vit-edit/bin/python \
  weightslab/examples/PyTorch/model-editing/vit-model-editing/probe_full_vit.py
```

That command exits non-zero until complete ViT dependency mapping works.

## Useful variants

Fast local smoke run (default, no downloads):

```bash
.venv-vit-edit/bin/python \
  weightslab/examples/PyTorch/model-editing/vit-model-editing/run_head_experiment.py \
  --image-size 32 --train-samples 16 --eval-samples 8
```

Standard 224x224 input geometry with randomly initialized weights:

```bash
.venv-vit-edit/bin/python \
  weightslab/examples/PyTorch/model-editing/vit-model-editing/run_head_experiment.py \
  --image-size 224
```

Pretrained ImageNet backbone (downloads torchvision weights):

```bash
.venv-vit-edit/bin/python \
  weightslab/examples/PyTorch/model-editing/vit-model-editing/run_head_experiment.py \
  --image-size 224 --pretrained
```

CPU and CUDA are supported. MPS is intentionally excluded because the current
`ModelInterface` device normalization maps non-CUDA devices back to CPU.

## Reading a result

A passing head report must have:

- `status: "passed"`
- `checks.add_propagated: true`
- `checks.optimizer_rebuilt: true`
- `checks.frozen_gradient_norm: 0.0`
- `checks.unfrozen_gradient_norm > 0`
- `checks.forward_after_edit: true`
- `checks.training_resumed: true`

These are behavioral checks against the live model and optimizer, not only API
shape checks.
