.. _docker-usage:

Docker Usage
===============

**Examples:** ``weightslab/examples/Docker_training/``

WeightsLab supports running the training script — and the entire UI stack —
inside Docker. Three integration patterns are available; they differ in **how
the trainer container gets a Docker daemon** to launch the Envoy proxy and the
Weights Studio frontend alongside it.

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * -
     - **A · Docker-in-Docker**
     - **B · Siblings (DooD)**
     - **C · Self-contained siblings**
   * - Docker daemon
     - Own nested daemon
     - Host daemon (socket mount)
     - Host daemon (socket mount)
   * - Envoy/frontend run
     - Nested inside trainer
     - Siblings on the host
     - Siblings on the host
   * - Starts UI via
     - ``weightslab ui launch``
     - ``weightslab ui launch``
     - Custom ``ui-compose.yml``
   * - ``--privileged``
     - **Required**
     - No
     - No
   * - gRPC ``:50051`` published
     - Not published
     - Yes
     - Yes
   * - Host bind mounts
     - None (co-located fs)
     - Path alignment required
     - **None**
   * - Host setup
     - None
     - ``setup-host.sh``
     - **None**
   * - HTTPS (optional)
     - ``WEIGHTSLAB_TLS=1``
     - ``WEIGHTSLAB_TLS=1``
     - ``WEIGHTSLAB_TLS=1``
   * - Windows / Docker Desktop
     - Yes
     - Awkward (use WSL2)
     - **Yes**

.. note::

   - **A (DinD)** — fully self-contained; needs ``--privileged``. Best when
     you want isolation or are on Windows.
   - **B (DooD)** — uses the stock ``weightslab ui launch``; cleanest on a
     Linux host with path alignment already done.
   - **C (self-contained siblings)** — no host prep, no bind mounts,
     works natively on Windows.

.. _docker-dind:

Option A: Docker-in-Docker (DinD)
----------------------------------

**Source:** ``weightslab/examples/Docker_training/docker_in_docker/``

The trainer container starts its **own inner Docker daemon** and uses
``weightslab ui launch`` inside it. Because the inner daemon shares the
trainer container's filesystem, all paths resolve without any host-side setup.

Wiring diagram
~~~~~~~~~~~~~~

.. code-block:: text

   host browser
     → localhost:5173 ── (re-published) ──► trainer ──► inner frontend :5173
     → localhost:8080 ── (re-published) ──► trainer ──► inner Envoy :8080
                                                            │ grpc-backend:host-gateway
                                                            ▼
                                         in-process gRPC backend :50051
                                         (same container as trainer)

Configuration requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 5 40 30 25

   * - #
     - Requirement
     - Location
     - Why
   * - 1
     - ``privileged: true`` on the trainer service
     - ``docker-compose.yml``
     - inner ``dockerd`` cannot run otherwise
   * - 2
     - Start the inner ``dockerd`` + wait for socket
     - ``entrypoint.sh``
     - nested daemon that hosts Envoy + frontend
   * - 3
     - Persist ``/var/lib/docker`` with a named volume
     - ``docker-compose.yml``
     - caches image pulls across runs
   * - 4
     - Publish ``5173`` + ``8080`` to the host
     - ``docker-compose.yml``
     - browser reaches the *nested* containers
   * - 5
     - ``GRPC_BACKEND_PORT=50051``
     - ``docker-compose.yml``
     - Envoy's ``grpc-backend:host-gateway`` dials this port
   * - 6
     - ``WEIGHTSLAB_SKIP_DOCKER_OPS=1`` before ``ui launch``
     - ``entrypoint.sh``
     - skips the in-container rebuild; pulls the image only
   * - 7
     - Order: ``ui launch`` → ``start example``
     - ``entrypoint.sh``
     - UI stack must be up before the backend starts

GPU access
~~~~~~~~~~

Add the ``deploy`` block to the trainer service in ``docker-compose.yml``:

.. code-block:: yaml

   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]

The host also needs the **NVIDIA Container Toolkit** installed (``sudo
nvidia-ctk runtime configure --runtime=docker``). Comment out the ``deploy``
block on hosts without an NVIDIA GPU/toolkit — ``docker compose up`` will
otherwise fail with *"could not select device driver nvidia"*.

Running
~~~~~~~

.. code-block:: bash

   # from weightslab/examples/Docker_training/docker_in_docker/
   docker compose up --build

Open http://localhost:5173. Stop with ``Ctrl+C`` or ``docker compose down``.

.. dropdown:: Enable HTTPS / mTLS (optional)
   :color: secondary

   Only needed for remote or production access. For local development, plain
   HTTP at http://localhost:5173 works without any certificates.

   DinD is the simplest TLS option because certs, Envoy, and the backend are
   co-located:

   .. code-block:: bash

      WEIGHTSLAB_TLS=1 docker compose up --build

   Then trust the generated CA on the host browser (once):

   .. code-block:: powershell

      # Windows — pull the CA out of the running container and trust it
      docker cp weightslab_trainer_dind:/root/.weightslab-certs/ca.crt .
      Import-Certificate -FilePath .\ca.crt -CertStoreLocation Cert:\CurrentUser\Root

   Open https://localhost:5173.

   What ``WEIGHTSLAB_TLS=1`` does in ``entrypoint.sh``:

   1. Runs ``weightslab ui launch --certs``, generating certs into
      ``WEIGHTSLAB_CERTS_DIR`` and configuring Envoy + the frontend for HTTPS.
   2. Exports ``GRPC_TLS_ENABLED=1`` + ``GRPC_TLS_CERT_DIR`` so the gRPC
      backend also speaks TLS (required to prevent Envoy upstream 503s).

.. _docker-siblings-c:

Option C: Self-contained siblings (recommended for Windows)
------------------------------------------------------------

**Source:** ``weightslab/examples/Docker_training/siblings_self_contained/``

The trainer mounts the **host Docker socket** and starts Envoy + the frontend
as sibling containers — but unlike Option B, it **never bind-mounts a host
path**. The Envoy config (and, for TLS, the certs) are delivered via **named
volumes over the socket**, so this option works on **Windows / Docker Desktop**
with zero host preparation.

Wiring diagram
~~~~~~~~~~~~~~

.. code-block:: text

   host browser
     → localhost:5173 ──► frontend container (sibling, HTTP)
     → localhost:8080 ──► Envoy container   (sibling, config from named volume)
                            │ grpc-backend:host-gateway → host:50051
                            ▼
                         host:50051 ──► trainer's gRPC backend :50051

Configuration requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 5 40 30 25

   * - #
     - Requirement
     - Location
     - Why
   * - 1
     - Mount ``/var/run/docker.sock``
     - ``docker-compose.yml``
     - drive the host daemon (start sibling containers)
   * - 2
     - Publish ``50051:50051`` from the trainer
     - ``docker-compose.yml``
     - sibling Envoy dials it via ``host-gateway``
   * - 3
     - Stage Envoy config into ``wl_envoy_cfg`` named volume
     - ``entrypoint.sh``
     - replaces the host bind mount — no host path needed
   * - 4
     - ``ui-compose.yml`` with no host bind mounts
     - ``ui-compose.yml``
     - nothing for the host daemon to resolve on disk
   * - 5
     - ``extra_hosts: grpc-backend:host-gateway`` on the Envoy service
     - ``ui-compose.yml``
     - routes Envoy → host:50051 → trainer
   * - 6
     - ``GRPC_BACKEND_PORT=50051``
     - ``docker-compose.yml``
     - port the backend binds + Envoy dials

How config delivery works without bind mounts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Option C does not call ``weightslab ui launch``. Instead, ``entrypoint.sh``:

1. Renders Envoy's plaintext config from the installed ``weightslab`` package.
2. Pipes it into the ``wl_envoy_cfg`` named volume via the socket using a
   temporary ``busybox`` container — no host path involved.
3. Brings up Envoy + frontend using ``ui-compose.yml``, which reads the config
   from the named volume rather than a bind-mounted host directory.

GPU access
~~~~~~~~~~

Same as Option A — add the ``deploy`` block to the trainer service and install
the NVIDIA Container Toolkit on the host.

Running
~~~~~~~

.. code-block:: bash

   # from weightslab/examples/Docker_training/siblings_self_contained/
   docker compose up --build

Open http://localhost:5173.

**Stopping:**

.. code-block:: bash

   docker compose down                                    # stop trainer
   docker compose -p weightslab_ui -f ui-compose.yml down  # stop UI siblings
   docker volume rm wl_envoy_cfg                         # optional: drop staged config

.. dropdown:: Enable HTTPS / mTLS (optional)
   :color: secondary

   Only needed for remote or production access. For local development, plain
   HTTP at http://localhost:5173 works without any certificates.

   TLS works here without host bind mounts — certs are piped into named volumes
   the same way as ``envoy.yaml``:

   .. code-block:: bash

      WEIGHTSLAB_TLS=1 docker compose up --build

   Then trust the CA on the host browser (once):

   .. code-block:: powershell

      docker cp weightslab_trainer_selfcontained:/root/.weightslab-certs/ca.crt .
      Import-Certificate -FilePath .\ca.crt -CertStoreLocation Cert:\CurrentUser\Root

   Open https://localhost:5173.

   What ``WEIGHTSLAB_TLS=1`` delivers (all via named volumes, no host paths):

   .. list-table::
      :header-rows: 1
      :widths: 35 30 35

      * - Layer
        - What's needed
        - How it arrives
      * - Browser ↔ Envoy
        - ``envoy-server.crt/key``
        - ``wl_envoy_cfg`` volume
      * - Envoy ↔ backend (mTLS)
        - ``envoy-client.crt/key`` + ``ca.crt``
        - ``wl_envoy_cfg`` volume
      * - Browser ↔ frontend
        - ``envoy-server.crt/key``
        - ``wl_nginx_certs`` volume
      * - Backend gRPC server
        - ``backend-server.crt/key`` + ``ca.crt``
        - ``GRPC_TLS_ENABLED=1`` + ``GRPC_TLS_CERT_DIR``

   .. note::

      Generated keys are ``0600`` (owned by root). ``entrypoint.sh`` runs
      ``chmod a+rX`` on cert files in the volumes so Envoy (non-root) can
      read them.

Common notes
------------

- The example starts **paused** (``is_training: false``); start and steer
  training from the UI at http://localhost:5173.
- First run is slow: it pulls ``envoyproxy/envoy`` + ``graybx/weightslab``
  and downloads MNIST for the classification example.
- For either option, to build against the dev branch instead of PyPI:

  .. code-block:: bash

     docker compose build \
       --build-arg WEIGHTSLAB_SPEC="git+https://github.com/GrayboxTech/weightslab.git@dev"
     docker compose up
