# Hard-example diagnostics

Controlled experiment and diagnostic prototype after the model-editing API. Read the
[meeting brief](../../../../../docs/proposals/hard-example-diagnostics.md) first.
For presentation, use the shorter
[meeting handout](../../../../../docs/proposals/hard-example-meeting-2026-10-08.md).

## What runs today

- `plan.py`: expands `experiment.json` into 12 planned Waterbirds runs; standard
  library only. Rejects an edited control, duplicate seeds/arms, invalid budgets
  and mismatched optimizer policy. It does not download data or launch training.
- `test_plan.py`: standard-library tests for the paired plan and its guardrails.
- `smoke.py`: runs two actual CPU training branches on a tiny synthetic fixture,
  starting from identical weights. One continues unchanged; the other uses the
  public WeightsLab neuron-addition API. Exports predictions, graph snapshots,
  checkpoint identity, edit history and ordinary/rare-group metrics.
- `CONTRACTS.md`: proposed case, diagnostic, intervention and comparison contracts
  for the backend and Studio. No new RPC or Studio screen is implemented yet.
- `prepare_waterbirds.py`: official metadata adapter and pinned frozen ViT feature cache.
- `run_waterbirds.py`: four arms across three seeds, validation-selected target,
  locked test manifest, checkpoint/shape/optimizer checks, per-case predictions
  and sampled layer diagnostics. Uses public `wl.compare_predictions`.
- `attribute_waterbirds.py`: full-image Integrated Gradients with checkpoint
  prediction parity, shared display scales and numerical-completeness warnings.
- `build_demo.py` + `demo_template.html`: self-contained offline results page.
- `test_experiment.py`: small CPU sampler and attribution numerical tests.

Synthetic fixture metrics test the plumbing. They are **not Waterbirds, ViT,
or evidence that editing improves real rare cases**. Real results come from
the Waterbirds runner below, not the synthetic fixture.

From the repository root, with WeightsLab's dependencies installed:

```bash
python weightslab/examples/PyTorch/wl-model-editing/hard-example-diagnostics/test_plan.py

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

## Run the real-data experiment

Requires the repository's torch/torchvision/Pillow/numpy dependencies. A GPU is
recommended for the frozen feature cache and image attribution. The measured
pilot used torch 2.9.1+cu128, torchvision 0.24.1+cu128 and an NVIDIA L40.

Download and unpack the official `waterbird_complete95_forest2water2` archive
from the [dataset authors](https://github.com/kohpangwei/group_DRO#waterbirds).
Preserve its `metadata.csv`, image paths and official splits. Check the source
dataset terms before redistributing images. Dataset-root below is the directory
containing `metadata.csv`, not its parent. Use fresh cache/run/output paths.

From the repository root:

```bash
export WL_NO_TELEMETRY=1 WEIGHTSLAB_OPENCODE_AUTOINSTALL=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=.
experiment_dir=weightslab/examples/PyTorch/wl-model-editing/hard-example-diagnostics

python "$experiment_dir/prepare_waterbirds.py" \
  --dataset-root /path/to/waterbird_complete95_forest2water2 \
  --output-dir /path/to/new-feature-cache
python "$experiment_dir/run_waterbirds.py" \
  --cache /path/to/new-feature-cache/features.pt --output-dir /path/to/new-run
python "$experiment_dir/attribute_waterbirds.py" \
  --run-dir /path/to/new-run --dataset-root /path/to/waterbird_complete95_forest2water2

curl -fL https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js \
  --output /path/to/chart-4.5.1.js
python "$experiment_dir/build_demo.py" --run-dir /path/to/new-run \
  --dataset-root /path/to/waterbird_complete95_forest2water2 \
  --chart-js /path/to/chart-4.5.1.js --output /path/to/demo/index.html
python -m http.server 8877 --bind 127.0.0.1 --directory /path/to/demo
```

Open `http://localhost:8877/` or open the generated HTML directly. All charts,
sample images and attribution overlays are embedded; the page needs no network
connection. Chart.js is pinned and SHA-256 verified by the builder. The page is
read-only: changing a filter replays saved evidence, not new training or edits.

The report retains all held-out predictions; preview images are explicitly
post-hoc illustrations. Attribution uses a fixed validation subset, not those
test illustrations. Known dataset label issues and failed numerical attribution
checks remain visible. Branch-local age starts at zero; the parent was trained
for the configured baseline budget. This is learned-head-parameter replay,
not a full training-state/ledger restore. Only load trusted generated caches.

Run targeted CPU checks:

```bash
python "$experiment_dir/test_plan.py"
python "$experiment_dir/test_experiment.py"
python -m pytest tests/diagnostics/test_prediction_comparison.py -q
```

## Remaining integration work

- [ ] Register per-case exports with the Studio data ledger; the offline runner
  currently exports them to JSON and uses `wl.track_model_signals` for model signals.
- [ ] Add live Studio transport, bounded diagnostic requests and stale-revision handling.
- [ ] Verify pause/edit/resume and browser/server synchronization end to end.
- [ ] Add full-state checkpoint replay beyond learned-head-parameter forks.
- [ ] Re-run the full ViT capability probe before any structural backbone editing.

Prototype lives under `wl-model-editing` so later architectures can share the
contracts. Keep real GPU experiment reports outside source control; commit the
recipe, split manifest hashes and compact reviewed summaries instead.
