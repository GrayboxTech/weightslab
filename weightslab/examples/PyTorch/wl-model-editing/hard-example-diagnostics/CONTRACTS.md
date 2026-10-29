# Proposed diagnostic contracts — version 0

Design proposal, not a public API commitment. Prototype with JSON artifacts;
agree the schema before changing the shared proto and regenerating both clients.

## Objects and ownership

| Object | Required identity / fields | Purpose |
|---|---|---|
| CaseSet | dataset/version, split, manifest SHA-256, case IDs, subgroup definition, selection checkpoint, reviewer note | Reproduce the same cases; detect split leakage |
| Case | stable sample ID, label, group, media reference, optional anchor ID and pair type | Keep hard positives/negatives meaningful relative to a representation |
| ModelSnapshot | run ID, checkpoint SHA-256, architecture revision, model age, backbone/preprocessing fingerprints, graph | Tie measurements to exact weights and architecture |
| LayerRef | snapshot ID, module path, live layer ID, shape | Live IDs are valid only within a wrapped model; resolve module paths again after reload |
| CaseDiagnostic | snapshot, case ID, logits, loss, true-label margin, prediction, selected-layer summaries | Compare the same example before and after |
| LayerHistory | snapshot, layer ref, step, scope, sample/batch IDs where applicable, gradient/activation/update values | Distinguish per-case observations from batch/step aggregates |
| Intervention | event ID, parent checkpoint, expected architecture revision, target path/ID, operation, arguments, reason, before/after shapes, optimizer policy, status | Explain and replay the engineer's action |
| Comparison | control/intervention run IDs, shared parent checkpoint, split hash, training budgets, per-group counts/metrics, corrected/regressed IDs | Separate intervention effects from ordinary continued training |

Missing/unavailable measurements must be `null` with an availability reason,
not zero. Version artifacts, keep numeric values finite, and use UTC timestamps.

## Reuse today's backend

- Structure: `model.get_model_graph()` and `model.get_layer_info()`.
- Per-sample evidence: `wl.save_signals` and sample history queries.
- Per-step health: `wl.watch_or_edit(..., track_model_signals=True)` or
  `wl.track_model_signals`; never broadcast a batch gradient norm onto samples.
- Editable operations: public model editing methods from PR #287.
- Lifecycle: `wl.guard_training_context`, `wl.guard_testing_context`, `wl.start_training`.

Extra work: checkpoint-to-checkpoint weight deltas, bounded per-case activation
capture, attribution metadata, experiment branch comparisons and event history.
For resized tensors, compare retained rows/columns and summarize new parameters
separately; never subtract different shapes or present new weights as drift.

## Proposed UI requests (conceptual, not existing endpoint names)

1. `ListCases(case_set, cursor, limit)` returns a bounded case page.
2. `InspectCases(snapshot, case_ids, layer_paths, requested_signals)` returns a
   bounded diagnostic payload; cache by checkpoint, preprocessing and case IDs.
3. `GetLayerHistory(run, layer_path, step_range, resolution)` downsamples curves.
4. `PreviewIntervention(snapshot, operation)` returns affected shapes and support status.
5. `ApplyIntervention(expected_revision, operation, reason)` pauses at a training
   boundary and applies under the existing architecture lock. Reject stale
   revisions and unsupported operations before mutation. On failure, report
   the error and restore a known checkpoint before allowing training to resume.
6. `CompareRuns(control, intervention, case_set)` joins results by stable case ID.

An edit acknowledgement must carry the new architecture revision, affected
layers, optimizer rebinding status and refreshed graph. The browser must discard
stale diagnostics and refetch that revision before enabling another edit. Save
an event only as successful after shape checks and a forward pass succeed.

## Attribution contract

Store method, target class/logit, baseline definition, input preprocessing,
checkpoint, seed, numerical approximation settings and shared display scale.
For attention include layer/head/token selection. For Integrated Gradients
include convergence delta. A CLS embedding is a vector, not a spatial heatmap.

Frozen ViT + head edit: backbone attention and backbone feature outputs should
stay fixed for an identical input in eval mode. Output-conditioned input
attribution can change. Verify this distinction in the UI and experiment tests.
If perturbation is added, label whether the changed object is a Q/K weight,
an attention logit/probability, or an activation; do not call them interchangeable.

## Integration acceptance

- Select the same case and layer in UI/backend; snapshot and revision agree.
- Pause, edit, validate propagation and optimizer parameter references, resume.
- Refetch graph and diagnostics; ignore stale replies from before the edit.
- Recover the same predictions from a saved checkpoint plus edit/event recipe.
- Record failures without claiming a completed intervention or improvement.
- Compare fixed held-out cases and display corrected as well as regressed samples.
