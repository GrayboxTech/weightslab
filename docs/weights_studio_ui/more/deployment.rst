Deployment and building
=======================

Bundled examples
----------------

Run a bundled example in one command (installs its requirements automatically)::

    weightslab start example          # classification (default)
    weightslab start example --seg    # segmentation
    weightslab start example --det    # detection
    weightslab start example --3d_det # 3D LiDAR point-cloud detection

In another terminal, start the UI::

    weightslab start

See ``weightslab start example --help`` for all options.

Cloud deployment
----------------

Because the UI is a plain Python process, cloud deployment is straightforward:

1. Install WeightsLab on the server::

     pip install weightslab

2. Run ``weightslab se`` once to generate certificates.

3. Start the backend in your training process (``wl.serve(serving_grpc=True)``).

4. Start the UI process::

     WEIGHTSLAB_UI_HOST=0.0.0.0 weightslab start --port 8080 --certs --no-browser

5. Put a reverse proxy (nginx / ALB / Caddy) in front of port ``8080`` and
   expose only ``443`` publicly.

The UI and backend can run on different machines — set ``--backend-host`` and
``--backend-port`` accordingly.

Example systemd unit
~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

  [Unit]
  Description=Weights Studio UI
  After=network.target

  [Service]
  EnvironmentFile=/etc/weightslab/env
  ExecStart=/usr/local/bin/weightslab start --port 8080 --no-browser
  Restart=on-failure
  RestartSec=5

  [Install]
  WantedBy=multi-user.target

Building the frontend from source
----------------------------------

The pre-built SPA is vendored into ``weightslab/ui/static/``. To rebuild from
the ``weights_studio`` source repository and update the vendored copy::

    # from the weights_studio repo
    npm ci && npm run build

  # from the weightslab repo
  rm -rf weightslab/ui/static/*
  cp -R ../weights_studio/dist/. weightslab/ui/static/
