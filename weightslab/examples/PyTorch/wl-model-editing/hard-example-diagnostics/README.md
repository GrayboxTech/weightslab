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

### Reading the attribution view

Open the **Image attribution** tab. The default overlays attribution on the
photograph. Orange supports the fixed dataset-class logit; blue opposes it.
The class being explained is labelled separately from each model's prediction.
Hover or tap either overlay to inspect the same location in both models and
read its signed contribution and change. Adjust opacity to locate contributions;
the grayscale treatment is display-only and never changes the model input.
**Heatmap only (advanced)** is optional, not the default.

The default detail level sums signed attribution inside 16 × 16 pixel regions,
so opposite contributions within a region can cancel. **Individual pixels**
shows the original signed, channel-summed values. Both models use the same
symmetric scale, pooled across all five snapshots for that image. The readable
scale caps magnitude at the pooled 99th percentile and reports the saturated
fraction for each displayed map. **Full range** disables clipping. Scales are
not comparable across different images or detail levels. These are not attention
maps, and color differences alone do not establish improved predictions.

New attribution exports include unclipped little-endian float32 maps. To update
an older demo without retraining or rebuilding the feature cache:

```bash
python "$experiment_dir/attribute_waterbirds.py" --run-dir /path/to/existing-run \
  --dataset-root /path/to/waterbird_complete95_forest2water2 \
  --output /path/to/attribution-ui.json
python "$experiment_dir/build_demo.py" --refresh-demo /path/to/old-demo/index.html \
  --attribution /path/to/attribution-ui.json --chart-js /path/to/chart-4.5.1.js \
  --output /path/to/new-demo/index.html
```

Refreshing reuses embedded images/results and validates protocol, attribution
seed/target, validation cases, inputs, labels and predictions. Output must be a
new file; the existing demo is preserved. A legacy export without raw maps shows
an explicit unavailable message rather than an invented heatmap.

Run targeted CPU checks:

```bash
python "$experiment_dir/test_plan.py"
python "$experiment_dir/test_experiment.py"
python "$experiment_dir/test_demo.py"
python -m pytest tests/diagnostics/test_prediction_comparison.py -q
node --test "$experiment_dir/test_attribution_view.cjs"
```

## Growth-initialization follow-up and live editing lab

This follow-up reuses the original parent checkpoints and feature cache. It
adds output-preserving growth and fixed-order pruning ablations. The test set
was already inspected in the original pilot; do not present this as a new
untouched holdout. See [the report for Alex](../../../../../docs/proposals/blog-packet-2026-10-29.md).

From the repository root, with the dependencies above installed:

```bash
export WEIGHTSLAB_PROJECTION=0 WEIGHTSLAB_OPENCODE_AUTOINSTALL=0 WL_NO_TELEMETRY=1
experiment_dir=weightslab/examples/PyTorch/wl-model-editing/hard-example-diagnostics
python "$experiment_dir/run_growth_followup.py" \
  --parent-dir /path/to/original-run --cache /path/to/features.pt \
  --output /path/to/new-growth-run --source-commit "$(git rev-parse HEAD)"
python "$experiment_dir/verify_growth_followup.py" \
  --run-dir /path/to/new-growth-run --cache /path/to/features.pt \
  --output /path/to/new-growth-run/verification.json
python "$experiment_dir/live_growth_lab.py" \
  --parent-dir /path/to/original-run --followup-dir /path/to/new-growth-run \
  --cache /path/to/features.pt --demo-html /path/to/demo/index.html \
  --output /path/to/live-audit --port 8878
```

Open `http://localhost:8878/`. The server binds to loopback, checks Host/Origin
and a per-process request token, and accepts only the fixed experiment controls.
For a remote host, forward the port with `ssh -L 8878:127.0.0.1:8878 <host>`.
Do not expose this experimental server directly to the public network.

Every click restores a checkpoint, applies real `add_neurons`/`prune_neurons`
operations where requested, and reevaluates all validation or test examples.
Trained-stage requests load the completed 250-step checkpoints; they do not
train in the HTTP request. The main reference is the equally trained no-growth
head under the same sampling policy. A separate pruning-impact line compares
against the selected checkpoint before removal. These are different controls.

The live process edits only the MLP head over cached frozen ViT features.
Output preservation zeroes the new classifier columns after growth; it is an
experiment-local policy, not a new option on the public editing API. All
checkpoints remain immutable. Only load trusted generated PyTorch caches.

For browser QA and actual screenshots, install Playwright in your tooling
environment and its Chromium browser. With the recorded viewer on port 8877
and live lab on port 8878:

```bash
node "$experiment_dir/capture_blog_packet.cjs" /path/to/packet
python "$experiment_dir/test_growth_followup.py"
python "$experiment_dir/test_live_growth_lab.py"
```

`RECORDED_DEMO_URL`, `LIVE_LAB_URL` and `PLAYWRIGHT_CHROMIUM_EXECUTABLE` can
override the capture defaults. The script writes seven screenshots, captions,
and the actual live request results after all assertions pass. It does not
fabricate predictions or modify saved experiment checkpoints.

To assemble the report packet, put the completed `followup.json` and
`verification.json` in an evidence directory, then run:

```bash
pip install markdown
python "$experiment_dir/build_blog_packet.py" \
  --evidence-dir /path/to/evidence \
  --report-md docs/proposals/blog-packet-2026-10-29.md \
  --recorded-demo /path/to/demo/index.html --output /path/to/packet
```

The output directory must already contain the capture script's screenshots and
browser evidence. The builder writes a readable HTML report, per-seed JSON,
checksums and a ZIP next to the packet directory. Keep dataset-bearing artifacts
outside Git and review image permissions before publishing them.

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
