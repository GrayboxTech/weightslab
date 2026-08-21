# WeightsLab — Detection Data Exploration & Relabeling (dataloader only)

A WeightsLab example with **no model, optimizer, loss, or training loop** —
only a `Dataset`/`DataLoader` registered with `wl.watch_or_edit(..., flag="data")`.
It exists purely to browse, inspect, and tag a detection dataset, then export a
tagged subset to a relabeling tool (CVAT / Label Studio / V7). If you're
looking for the trainable version of this same dataset, see
[`../wl-detection`](../wl-detection).

## Quick start

```bash
cd weightslab/examples/PyTorch/wl-detection-relabeling
pip install -r ../wl-detection/requirements.txt   # same deps, no extra training-only packages needed
python main.py
```

The **first run downloads** the Penn-Fudan Pedestrian dataset (~50 MB, ~170
real photos, one class: `person`) into `./data/`. Then open Weights Studio
(e.g. `http://localhost:5173`) — every sample and its ground-truth bounding
box is already there; nothing needs to run first.

## The workflow this demonstrates

1. **Explore** — browse the grid or List view; ground-truth boxes render as
   overlays (`task_type = "detection"`).
2. **Inspect** — open a sample's modal view for the full-size image + boxes.
3. **Tag** — flag samples that need another look. The script seeds 10 samples
   with a `tag:ToReview` boolean tag on startup (`seed_review_tag_count` in
   `config.yaml`; set to `0` to start clean) so there's something to export
   right away, but real usage is tagging whatever you actually flag while
   browsing — right-click a sample in the grid, or just ask the in-app chat
   agent (it has full tool access to this running process) something like
   *"tag the 5 blurriest samples as ToReview"* or *"mark sample 12 as
   ToReview"* (`wl.tag_samples([...], "ToReview")` under the hood).
4. **Export for relabeling** — hand off just the tagged subset:

   ```bash
   # CLI, from a second terminal (connects over gRPC to the running process)
   weightslab export -f cvat --tag ToReview

   # Or ask the chat agent directly:
   #   "export the samples tagged ToReview to CVAT format for relabeling"

   # Or the Weights Studio UI's Export button (format picker + tag picker)
   ```

   Only samples carrying the `ToReview` tag end up in the exported file — see
   [`../../../../docs/export.rst`](../../../../docs/export.rst) for the full
   tag-filter reference (also works with categorical tags, and with
   `wl.export_annotations(..., tags=[...])` from Python).

## Why no training loop

Every other PyTorch example in this repo (`wl-detection`, `wl-classification`,
...) wires a model + optimizer + loss and calls `guard_training_context`/
`guard_testing_context` around each step, plus `wl.start_training(timeout=...)`
to let the UI attach before stepping starts. None of that applies here:

- `watch_or_edit(..., flag="data", ...)` preloads every sample's ground-truth
  boxes and metadata into the dataframe **at registration time**
  (`preload_labels`/`preload_metadata` default to `True`) — the grid is fully
  populated before a single batch is ever pulled from the loader. There is no
  "step" to gate, so `guard_training_context`/`guard_testing_context` are
  skipped entirely.
- `wl.start_training()` only sleeps for a timeout then resumes the pause
  controller — meaningless without a stepping loop, so it's skipped too.
- `wl.serve(...)` + `wl.keep_serving()` are still called, so the process stays
  up and inspectable/taggable/exportable for as long as you need it.

If you want to iterate the loader anyway (e.g. to sanity-check `det_collate`
output shapes), a plain `for batch in loader: ...` works — it's a real
`DataLoader`, just never wrapped in a guard context here.

## Files

```
utils/data.py   PennFudanDetectionDataset + det_collate, copied unchanged from
                ../wl-detection/utils/data.py (same dataset, no model/loss
                utilities needed here).
main.py         Registers hyperparameters + the dataset, seeds a demo
                "ToReview" tag, serves, and keeps serving. No model/optimizer/
                loss/training loop.
config.yaml     Dataset + serving config only (no model/optimizer/training
                hyperparameters, since none exist in this usecase).
```

## Using your own dataset

Same as [`../wl-detection`](../wl-detection#using-your-own-dataset-eg-traffic-lights):
write a `Dataset` whose `get_items(idx, ...)` returns
`(image_tensor, uid, target, metadata)` with `target` an `[N, 6]`
`[x1, y1, x2, y2, class_id, confidence]` array normalized to `[0, 1]`, set
`self.task_type = "detection"` / `self.class_names`, and swap it in for
`PennFudanDetectionDataset` in `main.py`.
