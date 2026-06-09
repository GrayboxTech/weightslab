"""WeightsLab framework integrations.

Each subpackage targets one ML framework and exposes a two-name surface:
  * `WLAwareTrainer` — framework-Trainer subclass that wires WL.
  * `WLAwareDataset` — framework-Dataset subclass that returns the WL
    preview-protocol tuple.
"""
