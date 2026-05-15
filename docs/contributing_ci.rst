CI & Release Workflows
======================

This page explains how continuous integration works in WeightsLab and what
contributors can expect when opening a pull request or pushing a tag.

Overview
--------

Four GitHub Actions workflows run automatically depending on the branch, tag, or
pull request involved.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Workflow
     - Trigger
     - Purpose
   * - **Code CI**
     - Push / PR → ``dev``, ``main``
     - Lint, test, build and publish a timestamped dev package to TestPyPI
   * - **Dev Release CI**
     - Tag ``v*-dev`` on any non-main branch
     - Build and publish a versioned pre-release to TestPyPI, create a GitHub pre-release
   * - **Release CI**
     - Tag ``v*`` on ``main``
     - Build and publish to PyPI, create a GitHub release
   * - **Docs CI**
     - Push / PR → ``dev``, ``main``, or tag ``v*`` on ``main``
     - Build and deploy Sphinx docs to GitHub Pages

.. mermaid::

   flowchart TD
     PR["Pull Request → main"] --> CodeCI
     PushDev["Push → dev / main"] --> CodeCI
     CodeCI["Code CI\nlint · test · TestPyPI"]

     TagDev["Tag v1.2.3-dev\n(non-main branch)"] --> DevCI["Dev Release CI\nTestPyPI · GitHub pre-release"]
     TagMain["Tag v1.2.3\n(main only)"] --> ReleaseCI["Release CI\nPyPI · GitHub release"]
     TagMain --> DocsCI

     PushDev --> DocsCI["Docs CI\nGitHub Pages"]


Code CI
-------

Runs on every push to ``dev`` or ``main``, and on pull requests targeting ``main``.
Jobs run sequentially and each stage must pass before the next begins.

.. mermaid::

   flowchart LR
     A[code-quality] --> B[install] --> C[test] --> D[build-and-publish-dev]

**code-quality**
   Checks only the Python files that changed in the push.
   Runs ``ruff`` for style and unused imports, ``pylint`` for errors (score ≥ 7.0),
   and ``vulture`` for dead code.

**install**
   Installs the package in editable mode and verifies it can be imported.

**test**
   Reinstalls the package and runs the full unit test suite across four discovery
   paths: general, model, data, and integrations.

**build-and-publish-dev**
   Builds a wheel and sdist, then publishes to `TestPyPI <https://test.pypi.org>`_
   under a timestamped version (e.g. ``20260515120000.dev3735928559``). Skipped if
   the ``TEST_PYPI_API_TOKEN`` secret is not set.

.. admonition:: What contributors should expect
   :class: note

   Opening a pull request against ``main`` triggers Code CI on your branch.
   All jobs must be green before the PR can be merged. The TestPyPI publish step
   is informational — a failure there does not block merging.


Dev Release CI
--------------

Designed for maintainers who want to share a testable, versioned pre-release
from a feature or development branch without touching ``main``.

**How to trigger it**

Push a tag following the ``v<major>.<minor>.<patch>-dev`` pattern from any
branch except ``main``:

.. code-block:: bash

   git tag v1.2.3-dev
   git push origin v1.2.3-dev

The tag ``v1.2.3-dev`` is converted to the PEP 440 version ``1.2.3.dev0``
before publishing. If a ``v1.2.3-dev0`` tag already exists the next index is
used automatically (``v1.2.3-dev1``, and so on).

**What it does**

1. Verifies the tag does **not** point to a commit on ``main``.
2. Publishes ``weightslab==1.2.3.dev0`` to TestPyPI.
3. Waits for TestPyPI to index the package, then installs and imports it to
   confirm correctness.
4. Creates a **pre-release** on GitHub tagged with the original ``v1.2.3-dev``
   tag.

.. admonition:: Installing a dev release
   :class: tip

   .. code-block:: bash

      pip install --index-url https://test.pypi.org/simple/ \
                  --extra-index-url https://pypi.org/simple \
                  "weightslab==1.2.3.dev0"


Release CI
----------

Runs only when a clean version tag (``v<major>.<minor>.<patch>``, no suffix) is
pushed to a commit that is on ``main``.

.. code-block:: bash

   # From main only
   git tag v1.2.3
   git push origin v1.2.3

**What it does**

1. Verifies the tagged commit is an ancestor of ``main``. Fails immediately
   if the tag points to a non-main commit.
2. Builds the wheel and sdist, capturing the version from the tag via
   ``setuptools-scm``.
3. Publishes to `PyPI <https://pypi.org>`_.
4. Waits for PyPI to index the release, then installs and imports it.
5. Creates a **full GitHub release** with the ``CHANGELOG.md`` as the release
   body and the built artifacts attached.

.. admonition:: Production releases are maintainer-only
   :class: warning

   Only maintainers with write access to ``main`` and the ``PYPI_API_TOKEN``
   secret can publish a production release. External contributors should use
   ``v*-dev`` tags on their branches for shareable builds.


Docs CI
-------

Builds the Sphinx documentation using ``sphinx-multiversion`` and deploys it
to GitHub Pages. Triggered by:

- Pushes to ``dev`` or ``main``
- Tags ``v*`` on ``main`` (documentation is versioned per release)
- Pull requests targeting ``dev`` or ``main`` (build only, no deploy)

Tags with a ``-dev`` suffix (e.g. ``v1.2.3-dev``) do **not** trigger the
Docs CI — only production tags produce a versioned docs snapshot.


Tag convention summary
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Tag
     - Allowed branch
     - Result
   * - ``v1.2.3``
     - ``main`` only
     - PyPI release + GitHub release + versioned docs
   * - ``v1.2.3-dev``
     - Any branch except ``main``
     - TestPyPI pre-release + GitHub pre-release
   * - *(no tag)*
     - ``dev`` or ``main``
     - Code CI runs, timestamped TestPyPI build


Runner infrastructure
---------------------

All CI jobs run on **self-hosted GitHub Actions runners** hosted within the
GrayboxTech infrastructure rather than GitHub-hosted virtual machines. This
means:

- Jobs may start slightly slower if runners are busy (max 3 concurrent jobs).
- The runner environment is a Docker container based on Ubuntu 22.04 with
  Python 3.11, Node.js 20, Docker, and Docker Compose pre-installed.
- Cached dependencies (``pip``, ``ruff``) are shared across runs via a
  dedicated cache volume to speed up repeated jobs.

If a job is stuck in **Queued** state for more than a few minutes, a runner
may be unavailable — contact a maintainer.
