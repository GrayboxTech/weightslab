Experiment Versioning
=====================

WeightsLab versions an experiment by content, not by filename. Every time the
model, hyperparameters, or data state changes, ``CheckpointManager`` computes a
new **experiment hash** and checkpoints under it — so resuming, branching from
an earlier state, and reproducing a run are all the same mechanism: load the
hash you want.

Directory structure
--------------------

``CheckpointManager(root_log_dir=...)`` lays out everything under one root:

.. code-block:: text

   root_log_dir/
       checkpoints/
           manifest.yaml          # hash chronology: created/last_used, latest_weight_checkpoint
           models/
               <model_hash>/
                   <hash>_step_000100.pt
                   <hash>_architecture.pkl        # only if dump_model_architecture=True
           HP/
               <hp_hash>/
                   <hp_hash>_config.yaml
           data/
               <data_hash>/
                   <data_hash>_data_snapshot.json  # metadata + RNG state
                   <data_hash>_data_snapshot.parquet
           loggers/
               loggers.duckdb      # on-disk signal-history database

``root_log_dir`` comes from the top-level ``root_log_dir`` key in your
hyperparameters config (or the ``root_log_dir=`` kwarg on
``wl.watch_or_edit(..., flag="model"/"data"/...)`` for a per-object override)
— see :doc:`configuration`.

The experiment hash
--------------------

The **24-byte combined hash** is three 8-byte segments concatenated:

.. code-block:: text

   HP(8) + MODEL(8) + DATA(8) = 24-byte experiment hash

- **HP** — hyperparameters snapshot (everything in the registered config).
- **MODEL** — model architecture + the step it was initialized at.
- **DATA** — per-sample tags and discard state.

Each segment is also used as a directory name on its own (``models/<model_hash>/``,
``HP/<hp_hash>/``, ``data/<data_hash>/``), so unrelated experiments that happen
to share, say, the same model architecture reuse that one architecture file
instead of duplicating it.

Changing only the data state (tagging/discarding samples) changes the DATA
segment and produces a new combined hash, while HP and MODEL segments stay the
same — so the model directory is reused and only a new data/weights
checkpoint is written. The same applies to changing only hyperparameters or
only the model.

Pending vs. immediate changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``update_experiment_hash()`` detects what changed and either:

- **dumps immediately** (``dump_immediately=True``) — writes the new
  checkpoint right away, or
- **marks pending** — remembers what changed but defers the write until
  ``save_pending_changes()`` is called (e.g. when training resumes after an
  edit made while paused).

This avoids writing a checkpoint for every intermediate edit while the UI or
an agent is actively reshaping hyperparameters/data before resuming training.

The manifest and auto-resume
------------------------------

``checkpoints/manifest.yaml`` tracks every experiment hash ever seen under
that root, with ``created``/``last_used`` timestamps, the ``latest_hash``, and
each hash's ``latest_weight_checkpoint``/``latest_weight_step``. Constructing
``CheckpointManager(root_log_dir=...)`` on an **existing** root automatically
resumes the latest hash — no explicit ``load_state()`` call needed:

.. code-block:: python

   from weightslab.components.checkpoint_manager import CheckpointManager

   # If root_log_dir already has checkpoints, this resumes the latest one.
   manager = CheckpointManager(root_log_dir="./logs/my_experiment")

Branching from an older state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because every state is addressed by its hash, "branching" is just loading an
older hash and continuing training — the next change produces a new hash
alongside (not replacing) the branch point:

.. code-block:: python

   manager.load_state(older_hash, force=True)
   # ... continue training; the next change creates a new hash that
   # descends from older_hash, leaving the original branch on disk intact.

Step-aware checkpoint selection
---------------------------------

Within one experiment hash, multiple weight checkpoints accumulate at
different training steps (``<hash>_step_000100.pt``, ``..._step_000200.pt``,
...). ``load_checkpoint(exp_hash, target_step=...)`` picks the checkpoint
closest to ``target_step`` (ties break toward the higher step); omit
``target_step`` to get the latest one.

Reproducibility
-----------------

Every weight checkpoint and data snapshot also carries:

- **RNG state** (Python/NumPy/Torch) — restored on load so the exact sampling
  order resumes.
- **Dataloader iteration state** — how far each registered dataloader had
  progressed.

so resuming a checkpoint continues training deterministically rather than
just reloading weights.

Signal-history logging
-------------------------

Training curves (losses, metrics, per-sample signals) persist to an on-disk
DuckDB file at ``checkpoints/loggers/loggers.duckdb``, keyed by
``(metric_name, experiment_hash, step)``. See :doc:`logger` for how signals
get there in the first place. Because rows are namespaced by experiment hash,
switching between branches never overwrites another branch's curves — they
coexist in the same file and the UI/queries filter by hash.

``checkpoint_manager`` config options
----------------------------------------

Pass these under the ``checkpoint_manager`` key of your hyperparameters
config (see :doc:`configuration`) to control what gets dumped on a change:

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Option
     - Default
     - Description
   * - ``enable_checkpoints``
     - ``True``
     - Master switch — set ``False`` to disable all checkpoint dumping.
   * - ``dump_model_architecture``
     - ``False``
     - Pickle the full model object (structure + code), not just weights.
       Skipped by default to keep large-model checkpoints small; when
       skipped, a new hash whose model didn't change stores a small
       *reference* to the hash that has the architecture instead of
       duplicating it.
   * - ``dump_model_state``
     - ``True``
     - Save the model's ``state_dict()``.
   * - ``dump_optimizer_state``
     - ``True``
     - Save the optimizer's ``state_dict()`` alongside the model weights.
   * - ``dump_data_state``
     - ``True``
     - Save the per-sample tag/discard snapshot.
   * - ``dump_config_state``
     - ``True``
     - Save the hyperparameters YAML for this hash.

.. code-block:: yaml

   # hyperparameters.yaml
   root_log_dir: ./logs/my_experiment
   checkpoint_manager:
     enable_checkpoints: true
     dump_model_architecture: false
     dump_optimizer_state: true

Multi-root experiments
--------------------------

Point ``root_log_dir`` at a single experiment's own root (the common case
above) — or at a **parent directory that fans out into several independent
experiment roots**, e.g. sweeps or restarts each given their own
sub-directory:

.. code-block:: text

   scorer_exp/
       lr_tests/
           v1/checkpoints/manifest.yaml
           v2/checkpoints/manifest.yaml
       dataShuffling_tests/
           v1/checkpoints/manifest.yaml
           v2/checkpoints/manifest.yaml

.. code-block:: python

   # Point at the parent directory directly -- no need to know in advance
   # which sub-experiment is the freshest one.
   manager = CheckpointManager(root_log_dir="./scorer_exp")

``CheckpointManager`` recursively searches up to 3 levels of subdirectories
below the given root for anything containing ``checkpoints/manifest.yaml``
(so ``scorer_exp/lr_tests/v1`` — 2 levels down — is found; deeper than that
is not). Given what it finds:

- **Nothing found** (a brand-new, empty directory) — behaves exactly as
  before: a fresh experiment is started there.
- **One root found** (directly, or the sole nested one) — adopted as-is,
  identical to pointing at it directly.
- **Several roots found** — the one whose manifest was **most recently
  updated** is adopted as the effective root: its latest model weights,
  hyperparameters, and data state are loaded, and any further training
  writes new checkpoints there too. It's exactly as if you had pointed
  ``root_log_dir`` at that sub-directory directly.

Regardless of how many roots are found, **signal-history curves are merged
from every one of them** into the active logger, so training curves read as
one continuous history across all the discovered roots (not just the winner's
own) — useful when the sub-directories are really sequential attempts at the
same run rather than unrelated experiments. Curves merge at the database row
level and are namespaced by each root's own experiment hash, so this is
purely additive: nothing is overwritten, and merging the same sibling twice
(which can happen since a logger may be created before or after the manager)
never duplicates rows.

Introspecting a multi-root resolution:

.. code-block:: python

   manager.given_root_log_dir     # what you actually passed in
   manager.root_log_dir           # the effective root that was adopted
   manager.discovered_root_dirs   # every nested root that was found
   manager.is_multi_root          # True when more than one root was found
