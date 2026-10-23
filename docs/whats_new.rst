.. _whats-new:

What's New
==========

The last 10 stable releases of WeightsLab, newest first. Each card links to
its full release notes on GitHub, where the complete "What's Changed" commit
list lives.

.. note::

   Only stable releases appear here. Every tag also publishes a
   ``vX.Y.Z.devN`` pre-release to TestPyPI for verification; those are build
   plumbing rather than user-facing changes.

.. tip::

   Upgrade with ``pip install --upgrade weightslab``. Setting up for the first
   time? Start at :doc:`quickstart`.

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v2.0.3
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   October 20, 2026

   .. rst-class:: wl-release-version

   v2.0.3

   - `PR #312 <https://github.com/GrayboxTech/weightslab/pull/312>`__

     **Summary**

     This PR adds learned 2D/3D projections during training so model representations
     can be explored live in the Studio. By default, it uses a parametric UMAP trained
     alongside the model, allowing the projection to track representation changes as
     training progresses and to embed new samples without refitting. The projection
     can be configured per model, turned off when needed, and rendered alongside
     training progress in the Studio.

     *Full description in the pull request.*

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v2.0.2
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   October 1, 2026

   .. rst-class:: wl-release-version

   v2.0.2

   - `PR #306 <https://github.com/GrayboxTech/weightslab/pull/306>`__

     **Summary**

     - Improve experiment-view synchronization to operate incrementally instead of
       rebuilding the full dataset.
     - Reduce H5 update overhead and lock duration for large experiments.
     - Improve error reporting for failed synchronization and view-building
       operations.
     - Prevent training logs and tqdm output from appearing in unrelated notebook
       cells.
     - Fix communication issues caused by process-wide stdout/stderr redirection.
     - Fix the 'weightslab se' command for certificate generation.

     *Full description in the pull request.*

   - `PR #287 <https://github.com/GrayboxTech/weightslab/pull/287>`__

     **Summary**

     - add a versioned, JSON-serializable model graph and per-layer/per-neuron
       inspection API
     - add validated dependency-aware helpers for add, prune, freeze, unfreeze, reset,
       and perturb operations
     - make freeze/unfreeze idempotent at the public API boundary
     - document the model graph ontology, dependency types, and editing workflow
     - add regression coverage for graph serialization, live structural edits, frozen
       state, and invalid targets

     *Full description in the pull request.*

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v2.0.1
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   August 26, 2026

   .. rst-class:: wl-release-version

   v2.0.1

   - `PR #300 <https://github.com/GrayboxTech/weightslab/pull/300>`__

     UI Fixes and upgrade docs

   - `PR #299 <https://github.com/GrayboxTech/weightslab/pull/299>`__

     Add Opencode binary forgotten in v2

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v2.0.0
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   August 21, 2026

   .. rst-class:: wl-release-version

   v2.0.0


   **What's New ?**


   **New Features**

   - **Runs management**, unified UI to browse, organize, rename, and inspect
     experiment runs.

   - **Error bands & outlier highlighting**, curves now display statistical bands and
     visually emphasize anomalous steps.

   - **Relabelling export**, export tagged/annotated data to external tools (CVAT, V7,
     etc.) for downstream relabelling workflows.

   - **Integrated OpenCode Agent**, full agent loop support (code generation,
     training, monitoring, report creation) directly inside WeightsLab.

   - **Multimodal data support**, unified handling of images, videos, metadata, and
     structured signals.

   - **Automatic resource monitoring**, GPU/CPU/RAM usage tracked and surfaced during
     training and agent operations.

   - **Dynamic HTML report generation**, multi‑section experiment reports with plots,
     dataset analysis, training insights, and test results.


   **Fixes & Improvements**

   - **Agent stability improvements**, better token management, reliable process
     detaching, consistent initialization, and workspace‑safe lifecycle.

   - **Plotting upgrades**

   - Correct zoom behavior across large step ranges

   - Bright color palettes in light mode

   - Outlier visualization improvements

   - Right‑click actions: BBS, highlight, hide curve, step notes, load weights, color
     changes

   - **Signal pipeline fixes**, improved decimation, preservation of special points,
     kernel stability, and classification logic.

   - **DB performance improvements**, safer handling of large histories, better
     compaction, and reduced memory pressure.

   - **Tag painter fixes**, more reliable tagging, discarding, and annotation
     workflows.

   - **Workspace & session recovery**, restart window reloads ongoing sessions,
     history, and conversation context.

   - **UI polish**

   - Search bar cleanup

   - Agent input bar sync

   - Regex‑based research plots

   - Updated sandbox modes

   - Improved multimodal previews

   - **Cross‑platform testing**, validated on Windows, Ubuntu, Jupyter, and Google
     Colab.


   **Developer Experience**

   - **Unified configuration**, examples now rely on clean cfg files instead of
     hardcoded defaults.

   - **Improved CLI**, better agent commands, clearer ``/clear`` and ``/compact``,
     stable loop behavior.

   - **Changelog & documentation updates**, new “What’s New”, migration notes (W&B /
     v51 / 3LC), updated examples, and expanded UI documentation.


   **Experimental & Advanced**

   - **Video generation workflows**, multi‑input styles, real‑world models, and
     dataset‑driven video tasks.

   - **Image generation workflows**, PyTorch‑based generation paths integrated with
     agent prompts.

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.5.0
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   August 5, 2026

   .. rst-class:: wl-release-version

   v1.5.0

   - `PR #285 <https://github.com/GrayboxTech/weightslab/pull/285>`__

     - This PR introduces Google Collabs fixes and new integration:
     - Google Collab fixes
     - Jupyter Notebook Integration
     - Live notebook on-training allowing user to query, analyze, and ask agent to
       generate code to analyze ongoing experiment from the UI
     - Report generation with agent from code, CLI, or UI

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.4.2
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   August 4, 2026

   .. rst-class:: wl-release-version

   v1.4.2

   - `PR #283 <https://github.com/GrayboxTech/weightslab/pull/283>`__

     **Summary**

     - Wraps the "Install from TestPyPI/PyPI (wait until indexed)" + "Verify package
       import" steps in both ``test-install-from-pip-dev-release`` and
       ``test-install-from-pip-main`` in a single retried unit
       (``nick-fields/retry@v3``, ``max_attempts: 3``, ``retry_on: any``).
     - Guards against transient failures unrelated to the package itself: the registry
       hasn't finished indexing a fresh upload yet, or a one-off network blip during
       install/import.

     *Full description in the pull request.*

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.4.0
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   July 24, 2026

   .. rst-class:: wl-release-version

   v1.4.0

   - `PR #266 <https://github.com/GrayboxTech/weightslab/pull/266>`__

     > Re-opened from #265 with the Ultralytics example notebooks excluded (kept on
     ``dev`` only, not ready for this release).

     **WeightsLab v1.4.0, Release Notes**

     **🚀 Highlights**

     - **Collab compatibility improvements**
     - PyTorch notebooks support
     - UL notebooks integrations
     - Dependency fixes (notably around ``numpy > 2`` and ``protobuf``)
     - **Docker dependency cleanup**
     - Removed Docker-related dependencies/repositories
     - Updated UI Docker bridge (including rename)
     - Updated README and docs accordingly

     *Full description in the pull request.*

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.3.3
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   July 10, 2026

   .. rst-class:: wl-release-version

   v1.3.3

   - `PR #250 <https://github.com/GrayboxTech/weightslab/pull/250>`__

     **v1.3.3 Release**

     **Features & Improvements**

     - **Collab Integration**: Bundled examples with Google Collab support, tunnel
       serving function, and documentation links
     - **UI Enhancements**: Fix grid size button, add restart action in dev mode,
       Firefox input arrows fix
     - **Sorting**: Default decreasing order (high loss values prioritized),
       configurable sort
     - **CLI Upgrades**: Match latest features—hyperparameter modification, experiment
       status check, data tagging/discard, agent queries

     *Full description in the pull request.*

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.3.2
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   July 7, 2026

   .. rst-class:: wl-release-version

   v1.3.2

   - `PR #243 <https://github.com/GrayboxTech/weightslab/pull/243>`__

     Fix histogram generation (previously interpreted numeric values as categorical
     values)

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.3.1
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   July 7, 2026

   .. rst-class:: wl-release-version

   v1.3.1

   - `PR #242 <https://github.com/GrayboxTech/weightslab/pull/242>`__

     - Improve agent:
     - Add new agent actions, e.g., dump model or architecture, data state, and
       modelling actions
     - Change default model used from 70B to "~google/gemini-flash-latest"
     - Optimize agent pipeline
     - Revamp the documentation (new widgets and content)
     - Allow user to generate categorical histogram from metadata
     - Improve CLI
     - New WeightsLab Banner
     - Improve, fix, and add new user weightslab commands (e.g., weightslab cli)


----

`Every release on GitHub → <https://github.com/GrayboxTech/weightslab/releases>`__

.. This page is generated. CI (.github/workflows/whats-new.yml) refreshes it
.. after every stable release; to refresh it by hand:
..     python docs/_scripts/update_whats_new.py
.. Last generated: 2026-10-20
