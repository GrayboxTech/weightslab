# Model-editing experiments

This directory contains architecture-specific experiments that validate
WeightsLab model editing against live forward, backward, optimizer, and resumed
training behavior.

Available experiments:

- [`vit-model-editing`](vit-model-editing/README.md): ViT-B/16 full-model
  compatibility probe and supported editable-head control experiment.

Add future model families as sibling directories so their compatibility limits
and behavioral assertions remain independently runnable.
