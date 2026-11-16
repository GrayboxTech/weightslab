# Waterbirds: a controlled editing experiment

8 October 2026 · measured pilot · branch `srini/hard-example-diagnostics`

## Presentation takeaway

> We can inspect a recurring failure, try a controlled intervention, and show
> both improvements and regressions. In this experiment, more exposure to the
> rare group helped; adding neurons alone did not. Inspection also surfaced
> known label problems among the hardest examples.

This is a useful diagnostic result, not evidence that adding capacity reliably
improves a transformer. The backbone was frozen; only the MLP head was edited.

## Measured results

Mean test accuracies across paired seeds 17, 29 and 43:

| Arm | Waterbird on land | Change vs control | Common cases | Overall empirical |
|---|---:|---:|---:|---:|
| Continue | 59.45% | — | 98.19% | 84.32% |
| Add 16 head neurons | 57.11% | −2.34 pp | 96.84% | 82.74% |
| Balanced sampling | 82.87% | +23.42 pp | 95.56% | 90.94% |
| Widen + balanced sampling | 74.71% | +15.26 pp | 93.76% | 86.00% |

Common cases pool landbird/land and waterbird/water by their test counts.
The balanced arm loses **2.63 percentage points** on this measure, exceeding
the proposed one-point regression budget. No arm satisfies the complete pilot
target. Three seeds are preliminary evidence, not a significance claim.

Target-group paired gains, seed order 17 / 29 / 43:

- Widen: −3.27 / −0.62 / −3.12 pp.
- Balanced: +22.90 / +23.99 / +23.36 pp.
- Combined: +14.95 / +12.46 / +18.38 pp.

The report also records the training-frequency-weighted benchmark average;
it is not interchangeable with the empirical test average above.

## Protocol and verification

- Official Waterbirds splits: 4,795 training, 1,199 validation, 5,794 test.
- Frozen torchvision ViT-B/16 `IMAGENET1K_V1`, its pinned preprocessing,
  768-dimensional features, editable 64-unit MLP head, widened to 80 units.
- 500 baseline steps per seed; four branches with 250 steps each. Fresh SGD
  without momentum in every branch; learning rate 0.01, training batch size 64.
- Target group chosen by lowest mean baseline validation accuracy, before test
  prediction inspection. All official test cases and labels retained.
- Matching sampled batches within capacity pairs; immediate-edit and trained
  results recorded separately. Retained weights, dependent layer shapes and
  optimizer parameter references checked after every structural edit.
- Two complete runs produced identical final per-case predictions and sampled
  training histories. Independently replayed all 12 saved final heads on all
  5,794 held-out cases per head; prediction labels matched exactly.
- Frozen features extracted on NVIDIA L40; torch 2.9.1+cu128 and torchvision
  0.24.1+cu128. Checkpoint, metadata, preprocessing, protocol and runner/API
  source hashes are retained in the generated evidence bundle.

Final protocol hash:
`7902607cf99bc35a06b1518d6e23ba09371b5058b83c7d24e6267005a595557e`.
The run used the working source before its publication commit; the report
explicitly records `source_dirty` and exact runner/comparison source hashes.

## What to show in five minutes

1. **Results:** start with balanced sampling and the waterbird/land group.
   Point out the +23.42-point gain and −2.63-point common-case tradeoff.
2. **Model editing:** select “Add 16 neurons.” Show 64 → 80 hidden units and
   the classifier input growing to 80 while the 768-dimensional backbone stays
   fixed. Compare the immediate edit effect with the trained outcome.
3. **Cases:** toggle corrected/regressed examples and inspect predictions,
   confidence and margins. Images are post-hoc illustrations; metrics use the
   full held-out set. Aggregated charts and single-seed images are labelled.
4. **Attribution:** inspect the same validation image through two trained heads.
   These are class-conditioned Integrated Gradients, not changed attention.
   Approximation warnings remain visible where convergence checks failed.
5. **History:** open the provenance panel and show the common checkpoint,
   intervention, optimizer checks and locked protocol.

## Feature contribution

`wl.compare_predictions(control, intervention)` is a new stateless public API.
It validates supplied checkpoint/split/preprocessing/budget/optimizer metadata,
joins stable sample IDs, rejects changed cohorts or labels, and returns
per-group accuracy changes plus corrected and regressed IDs. It does not
silently hide missing cases. Its checks do not independently prove that
supplied provenance is truthful; the runner and replay verifier supply that
additional evidence here.

The branch also provides the dataset adapter, frozen-feature cache, controlled
runner, saved-checkpoint verifier, numerical attribution checks and offline
interactive demo. [API reference](../prediction_comparison.rst) ·
[reproduction commands](../../weightslab/examples/PyTorch/wl-model-editing/hard-example-diagnostics/README.md).

## Important boundaries

The authors flag Eastern Towhees, Western Meadowlarks and Western Wood Pewees
as incorrectly labelled waterbirds. Some locked hard validation cases are
Eastern Towhees. The UI flags this; the experiment does not relabel or remove
them after looking at results. [Authors' dataset note](https://github.com/kohpangwei/group_DRO#waterbirds).

Integrated Gradients used 32–128 Gauss-Legendre samples and a zero normalized
input baseline. Six of the 20 baseline/control/intervention maps failed the
recorded absolute-plus-relative completeness tolerance; warnings are retained.
Do not use small visual differences as causal proof. Method reference:
[Integrated Gradients and completeness checks](https://captum.ai/docs/extension/integrated_gradients).

The page is an offline recorded-run prototype, not a live Studio screen. No new
RPC, browser/server synchronization, structural transformer editing, full
training-state restore, or reliable real-world rare-animal improvement is
claimed. Those remain separate integration/research work.
