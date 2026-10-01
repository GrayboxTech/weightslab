# Hard-example diagnostics

Planning draft for the project after the model-editing API. Read the
[meeting brief](../../../../../docs/proposals/hard-example-diagnostics.md) first.

## What runs today

- `plan.py`: expands `experiment.json` into 12 planned Waterbirds runs; standard
  library only. It does not download data or launch training.
- `smoke.py`: runs two actual CPU training branches on a tiny synthetic fixture,
  starting from identical weights. One continues unchanged; the other uses the
  public WeightsLab neuron-addition API. Exports predictions, graph snapshots,
  checkpoint identity, edit history and ordinary/rare-group metrics.
- `CONTRACTS.md`: proposed case, diagnostic, intervention and comparison contracts
  for the backend and Studio. No new RPC or Studio screen is implemented yet.

Synthetic fixture metrics test the plumbing. They are **not Waterbirds, ViT,
or evidence that editing improves real rare cases**. No image attribution or
data download is implemented in this scaffold.

From the repository root, with WeightsLab's dependencies installed:

```bash
python weightslab/examples/PyTorch/wl-model-editing/hard-example-diagnostics/plan.py \
  --output /tmp/weightslab-hard-example-plan.json

WL_NO_TELEMETRY=1 PYTHONPATH=. python \
  weightslab/examples/PyTorch/wl-model-editing/hard-example-diagnostics/smoke.py \
  --output-dir /tmp/weightslab-hard-example-smoke
```

Use a fresh output directory for each smoke run. Output is `report.json` and
`checkpoint.pt` alongside WeightsLab's state. The runner fails if propagation,
optimizer parameter binding, checkpoint parity, or finite training checks fail.
It does not assert that the widened model wins.

The smoke fork restores learned head parameters, starts each branch's local step
count at zero, and uses fresh SGD without momentum in both arms. It is not a full
WeightsLab checkpoint/replay implementation; the report records parent model age
separately. Signal tracking starts after an edit so hooks bind to current tensors.

## Next implementation steps

- [ ] Add a Waterbirds adapter with versioned metadata and unchanged official splits.
- [ ] Cache features from a pinned pretrained ViT-B/16 in evaluation mode; retain
  image IDs, image paths, preprocessing and backbone weight fingerprint.
- [ ] Select a known failure group on validation and freeze the case manifest.
- [ ] Extend the paired runner to the four planned sampling/capacity arms.
- [ ] Record per-case loss/margin and selected-layer activations; reuse
  `wl.save_signals` and `wl.track_model_signals` for their appropriate scopes.
- [ ] Capture step histories before and after edits; verify signal hooks rebind
  to replaced parameters and do not duplicate events.
- [ ] Add class-conditioned image attribution through the full frozen backbone.
- [ ] Add a read-only comparison view and intervention history, then Studio transport.
- [ ] Re-run the full ViT capability probe before any structural backbone editing.

Prototype lives under `wl-model-editing` so later architectures can share the
contracts. Keep real GPU experiment reports outside source control; commit the
recipe, split manifest hashes and compact reviewed summaries instead.
