# Waterbirds model-editing experiment: report for Alex

Date: 29 October 2026. Status: completed exploratory experiments; share with the
caveats below. This is evidence for an engineering workflow, not a claim that
adding neurons improves models generally.

## Experiment goal

We wanted to distinguish three explanations for a recurring classification
failure: insufficient exposure to rare examples, insufficient head capacity,
and disruption caused by the edit's initialization. We then tested whether
trained predictions actually depended on the added neurons by removing them.

The tool makes those comparisons inspectable. An engineer can choose a growth
policy, restore a checkpoint, prune a fixed number of added units, and evaluate
the same population. The UI reports the dependent layer shapes, prediction
changes, group accuracies, and checkpoint identity for that request.

## Dataset and model

Waterbirds composites bird photographs onto land or water backgrounds. The
failure group is waterbirds on land, selected on validation in the original
pilot. This is a benchmark of background correlation, not a demonstration about
naturally occurring rare species. See the [dataset authors' description](https://github.com/kohpangwei/group_DRO#waterbirds).

| Item | Setting |
|---|---|
| Model | Frozen torchvision ViT-B/16, IMAGENET1K_V1 weights |
| Edited component | MLP head: Linear(768, 64), LeakyReLU(0.1), Linear(64, 2) |
| Growth | Add 16 hidden units; classifier input grows from 64 to 80 |
| Splits | Official train 4,795; validation 1,199; test 5,794 |
| Test groups | Landbird/land 2,255; landbird/water 2,255; waterbird/land 642; waterbird/water 642 |
| Common-case metric | Pooled landbird/land + waterbird/water accuracy; 2,897 images |
| Seeds | 17, 29, 43; same parent checkpoint within each seed |
| Training | Parent: 500 steps. Each branch: 250 more steps, batch 64 |
| Optimizer | Fresh SGD in every branch, learning rate 0.01, momentum 0 |
| Sampling | Original training frequencies or equal expected exposure per group |
| Matched training | Identical sampled-batch hashes across growth policies within each seed/sampling pair |
| Execution | Cached frozen ViT features; training on NVIDIA L40; live head inference on CPU |

The backbone is not edited. No transformer attention parameters change. These
are learned-head-parameter forks, not restoration of the full training ledger
or optimizer history. No new training happens when a UI button is clicked.

## Experiments and results

### 1. Sampling versus additional capacity

The original pilot compared continued training, random growth, balanced
sampling, and random growth plus balanced sampling. It showed that balanced
exposure helped the selected group, but also reduced common-case accuracy.
The recorded inspector retains that experiment's results and provenance.

### 2. Growth initialization: six arms, three paired seeds

The follow-up reran the four controls and added an output-preserving policy
under each sampling scheme. It calls `model.add_neurons`, then zeros **only the
new classifier input columns**. Incoming weights remain randomly initialized,
so the new units can learn. All retained parameters must remain unchanged.

This is an experiment-local policy, not a new public WeightsLab option or a
novel initialization method. Function-preserving network expansion is
established prior work; see [Net2Net](https://arxiv.org/abs/1511.05641). Our policy
is not a claim to reproduce that paper's exact algorithm.

The table reports mean test accuracy across three seeds. Percentage-point
differences below use unrounded values. The full machine-readable packet also
includes each seed and all pruning levels.

| Growth policy | Sampling | Waterbird/land accuracy | Common-case accuracy |
|---|---|---:|---:|
| No growth | Original | 59.45% | 98.19% |
| No growth | Balanced | 82.87% | 95.56% |
| Random +16 | Original | 57.11% | 96.84% |
| Random +16 | Balanced | 74.71% | 93.76% |
| Preserve outputs +16 | Original | 59.55% | 98.18% |
| Preserve outputs +16 | Balanced | 83.13% | 95.48% |

Random growth changed test logits immediately, before training, by a maximum
of 17.73–19.84 across the three seeds. Output-preserving growth kept the maximum
change at or below 2.87e-6, within the predeclared 1e-5 tolerance. This establishes
the initialization behavior; it does not establish an accuracy benefit.

With balanced sampling, preserved growth exceeded random growth by 8.41 points
on the target group. However, it exceeded the **equally sampled no-growth
control** by only 0.26 points and reduced common-case accuracy by 0.08 points.
The experiment does not establish a capacity advantage. Compared with original
sampling/no growth, the target gain was 23.68 points, with a 2.72-point common-case
regression: it fails the proposed one-point regression guardrail.

### 3. Remove the added neurons after training

We pruned the last 0, 4, 8, or 16 added units in a fixed order, with no ranking
based on evaluation results and no retraining after removal. The dependent
classifier width contracted through the actual WeightsLab editing API.

For preserved growth with balanced sampling, removing all 16 units reduced
mean target accuracy from 83.13% to 81.15% (−1.97 points). This demonstrates a
contribution within those trained checkpoints, not that extra capacity was
necessary: the no-growth balanced control already reached 82.87%.

| Seed | Preserve + balanced | After pruning all 16 | Change |
|---|---:|---:|---:|
| 17 | 83.33% | 82.24% | −1.09 pp |
| 29 | 83.64% | 81.00% | −2.65 pp |
| 43 | 82.40% | 80.22% | −2.18 pp |

Pruning is **not rollback**. The retained units also learned during continuation.
Returning to width 64 therefore does not recover the original 64-unit model.

## How to demonstrate the tool

1. Open the recorded inspector's Results tab. Select balanced sampling and
   Waterbird/land. Show both the rare-group improvement and common-case cost.
2. Open Image attribution. Keep Overlay on image selected, choose a validation
   image, and hover or tap a region. Both maps show the same coordinates and
   shared scale, with predictions and signed attribution differences alongside.
3. Open the live editing lab. Select seed 17, test replay, original sampling,
   immediate stage, random growth, and no pruning. Click Apply. The returned
   logits changed before any training, despite valid propagated layer shapes.
4. Change only the growth policy to preserve outputs and apply again. The
   dimensions still expand, but predictions remain unchanged within tolerance.
5. Select the trained stage and balanced sampling. Apply, then prune all 16
   added units and apply again. The execution log records `model.prune_neurons`,
   the width becomes 64, and the pruning-impact line compares against the same
   checkpoint before removal. The main table separately compares with the
   equally trained no-growth reference.

Every live request restores an immutable checkpoint and recomputes inference;
edits do not accumulate. Checkpoint hashes and the request ID are in the UI
and persisted audit. This is a **custom experiment UI**, not proof of Weights
Studio/gRPC synchronization. That integration is still outstanding.

## Screenshot packet

The packet contains actual browser captures, not mockups. `screenshots.json`
records captions; `browser-evidence.json` records the live requests.

| Figure | File | What it demonstrates |
|---|---|---|
| 1 | 01-recorded-overview.png | Controls, full-group results, and validation curves |
| 2 | 02-attribution.png | Image overlays and linked region inspection; both shown predictions remain wrong |
| 3 | 03-live-random-growth.png | Output disruption before training under random growth |
| 4 | 04-live-preserved-growth.png | Same starting checkpoint with output-preserving growth |
| 5 | 05-live-trained-added-units.png | Trained head compared with the matched no-growth control |
| 6 | 06-live-remove-added-units.png | Actual pruning, dependent shapes, and measured removal effect |
| 7 | 07-live-mobile.png | Narrow-screen controls and results |

Attribution uses Integrated Gradients for the dataset-label logit, not
attention. Four validation images and seed 17 are fixed, independently of the
test-example browser. Region view sums signed contributions inside 16×16
regions; signs can cancel. Grayscale is display-only, and the color scale is
shared across snapshots of the same image. Failed numerical checks remain
visible. A passing [completeness check](https://captum.ai/docs/extension/integrated_gradients)
does not validate a visual explanation or a prediction. Stronger colors are
not evidence of a better model.

## Validation and evidence limits

- Independently replayed 18 saved heads using plain PyTorch functional layers:
  125,874 saved validation/test prediction rows matched predicted labels exactly.
- Recomputed all four group accuracies for 108 full-split evaluations, including
  every pruning level. All matched. Maximum CPU replay logit difference from
  saved GPU inference was 6.20e-6 (verification tolerance 1e-4).
- Checkpoint, cache, protocol and prediction-file hashes are retained. Training
  also checks retained parameters, dependency shapes and optimizer bindings.
- Browser QA exercised 48 attribution views, 17 live requests, invalid edit and
  missing-token rejection, and 320/390px layouts. No browser errors were observed.
- Targeted Python tests: 37 passed plus 19 subtests; attribution renderer: 6
  passed. This is not a claim that the repository's entire CI suite passed.
- Follow-up test data had already been inspected in the original pilot. These
  are exploratory results, not a fresh-holdout confirmation or a significance
  claim. No hyperparameter search or generalization to other backbones is shown.
- The dataset authors list known label issues. Official labels were preserved;
  warnings remain visible on relevant examples. Examples are selected post hoc
  for illustration, never used as the metric denominator.
- Bird/background imagery comes from the benchmark's underlying datasets.
  Confirm permission for individual images before publishing them in a blog;
  the code repository's license is not an image-redistribution license.

## Reproducibility

Use the commands in the experiment folder's README. The original pipeline
prepares features and checkpoints; `run_growth_followup.py` creates all six
arms; `verify_growth_followup.py` independently replays them;
`live_growth_lab.py` serves the actual editing process; and
`capture_blog_packet.cjs` records browser evidence. The packet builder packages
measured data and screenshots without model files or credentials.

- Training source: `af70f6f8f94f7e67602b3b14b0958ff8756893fb`.
- Follow-up protocol SHA-256:
  `a3e1b9ccdc8bd540d12c1082c4997312b40b9c1ec6130da64095d03933e6ad3a`.
- Completed report SHA-256:
  `a5e3e5863d2bc19d22e5a67f6f090998d80dd90b134c76f68df51bd3d5298234`.

The UI and verifier were added after that training commit; their source hashes
are recorded separately in live request and verification evidence.

## PR feedback status

PR #311 is open against `dev-modeling`; the maintainer made it ready for review.
The visible reviewer comment says the generated template worked and asks to
reply to/fix comments. At the 29 October check, both the GitHub review-thread
and review-comment APIs returned no inline threads for #311 or #307. No unseen
review request is claimed resolved. A permalink is needed for any comments
outside those visible PR conversations. The older draft #307 is unchanged.
