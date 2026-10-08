Paired prediction comparisons
=============================

``wl.compare_predictions(control, intervention)`` compares two recorded model
branches on the same held-out cases. It is stateless, torch-free, and does not
require a live ledger, training process, or Studio server.

Record the baseline checkpoint once, fork the branches, and evaluate both after
equal training budgets. Supply JSON-compatible dictionaries:

.. code-block:: python

   import weightslab as wl

   # Use real SHA-256 digests from your artifacts, not placeholder values.
   provenance = {
       "parent_checkpoint_sha256": parent_checkpoint_digest,
       "case_manifest_sha256": held_out_manifest_digest,
       "preprocessing_sha256": preprocessing_digest,
       "evaluation_split": "test",
       "training_steps": 250,
       "optimizer_policy": "fresh_identical_optimizer_in_all_arms",
   }
   control = {**provenance, "run_id": "seed17-continue", "cases": control_cases}
   intervention = {**provenance, "run_id": "seed17-widen", "cases": edited_cases}
   comparison = wl.compare_predictions(control, intervention)
   print(comparison["groups"], comparison["corrected_ids"], comparison["regressed_ids"])

Each case contains ``sample_id`` (unique nonempty string), ``group`` (nonempty
string), and ``label`` / ``prediction`` (nonnegative integers). Optional
``true_label_margin`` is a finite number or ``None``. Other fields are ignored.
IDs may arrive in different orders; the comparison joins them by identity.

Safety checks
-------------

- Both inputs need distinct run IDs and matching checkpoint, case-manifest,
  preprocessing, split, training-step and optimizer-policy metadata.
- Digests must be lowercase 64-character SHA-256 strings. Budgets are
  nonnegative integer step counts, not elapsed time or epochs.
- Missing or duplicated cases, changed labels/groups, and nonfinite margins
  fail. The function never silently intersects two different cohorts.
- ``TypeError`` identifies non-mapping snapshots/cases; ``ValueError`` identifies
  other invalid or unpaired input. Inputs are not modified.

Result and boundaries
---------------------

The versioned result includes overall and per-group empirical accuracy,
accuracy differences, counts, corrected/regressed IDs, and sorted per-case
outcomes. Accuracies/differences are **fractions**, not percentages. Missing
margins remain ``None``. Convert differences to percentage points in the UI.

This validates the **supplied provenance**, not the actual files, training
implementation, or truth of the recorded metadata. It does not certify causal
mechanisms, statistical significance, fairness, or checkpoint replay. Callers
must separately verify identical parent weights, shape/optimizer correctness,
sampling policies and saved-artifact hashes. Dataset-weighted averages and
multi-seed uncertainty are the experiment's responsibility.

The ``wl-model-editing/hard-example-diagnostics`` example exercises this API
in a four-arm Waterbirds pilot and exports an offline comparison page. That
page is a recorded-run prototype, not a new live Studio screen or RPC.
