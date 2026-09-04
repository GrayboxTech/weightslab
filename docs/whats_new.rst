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

     **Test plan**

     Cut a ``-dev`` tag and confirm …

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

     **WeightsLab v1.4.0 —**

     - Release Notes ### 🚀 Highlights
     - **Collab compatibility improvements**
     - PyTorch notebooks support
     - UL notebooks integrations
     - Dependency fixes (notably around ``numpy > 2`` and ``protobuf``)
     - **Docker dependency cleanup**
     - Removed Docker-related dependencies/repositories
     - Updated UI Docker bridge (including rename)
     - Updated README and docs accordingly
     - …

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.3.3
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   July 10, 2026

   .. rst-class:: wl-release-version

   v1.3.3

   - `PR #250 <https://github.com/GrayboxTech/weightslab/pull/250>`__

     ## v1.3.3 Release

     **Features &**

     - Improvements
     - **Collab Integration**: Bundled examples with Google Collab support, tunnel
       serving function, and documentation links
     - **UI Enhancements**: Fix grid size button, add restart action in dev mode,
       Firefox input arrows fix
     - **Sorting**: Default decreasing order (high loss values prioritized),
       configurable sort
     - **CLI Upgrades**: Match latest features—hyperparameter modification, experiment
       status check, data tagging/discard, agent queries
     - **Agent Actions**...

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

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.3.0
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   June 29, 2026

   .. rst-class:: wl-release-version

   v1.3.0

   *(no pull requests found since last release)*

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.2.5
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   June 17, 2026

   .. rst-class:: wl-release-version

   v1.2.5

   - `#207 <https://github.com/GrayboxTech/weightslab/pull/207>`__

     v1.2.5 — 2026-06-17 Fix EMA Sync. from Ultralytics trainer and evaluate mode

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.2.4
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   June 17, 2026

   .. rst-class:: wl-release-version

   v1.2.4

   - `#203 <https://github.com/GrayboxTech/weightslab/pull/203>`__

     - v1.2.4 — 2026-06-17
     - Fix the evaluation mode issue
     - Disable the watchdog for threads by default
     - Fix code quality issues

.. card::
   :link: https://github.com/GrayboxTech/weightslab/releases/tag/v1.2.3
   :class-card: wl-release-card

   .. rst-class:: wl-release-date

   June 16, 2026

   .. rst-class:: wl-release-version

   v1.2.3

   - `#201 <https://github.com/GrayboxTech/weightslab/pull/201>`__

     - v1.2.3
     - Fixes — 2026-06-16
     - Fixed some bugs around metadata fetching from the UI
     - Fixed logger spam and TLS format
     - Add new materials in the weights lab around experiment running inside Docker
       (DinD, and DoutD)


----

`Every release on GitHub → <https://github.com/GrayboxTech/weightslab/releases>`__

.. This page is generated. To refresh it after a release:
..     python docs/_scripts/update_whats_new.py
.. Last generated: 2026-08-21
