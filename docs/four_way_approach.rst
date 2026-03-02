Four-Way SDK Approach
=====================

WeightsLab is organized around four complementary capabilities:

1. Model interaction
2. Data exploration
3. Hyperparameter management
4. Logger and signal tracking

These capabilities are designed to be used together in one training script.

.. mermaid::

   flowchart LR
     HP[Hyperparameters] --> M[Model Interaction]
     D[Data Exploration] --> M
     M --> L[Logger and Signals]
     L --> D
     L --> HP

Typical integration flow
------------------------

- Register hyperparameters first so all components use one shared configuration source.
- Wrap dataset/dataloader to expose sample IDs and per-sample operations.
- Wrap model, optimizer, losses, and metrics.
- Start WeightsLab services and run training.
- Use tags/discards/signals to iteratively improve data and model behavior.
