Quickstart
==========

This page gives you a practical, minimal path to get WeightsLab running.
If you prefer to start from examples, see ``usecases`` right after this setup.

Prerequisites
-------------

- Python v3.10+ installed
- A virtual environment tool like ``venv`` or Conda (optional).
- Your training project available locally.

Install WeightsLab
------------------

Create and activate a virtual environment and install WeightsLab.

.. code-block:: bash

   # From the repository root
   python -m venv .venv

   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS
   # source .venv/bin/activate

   python -m pip install weightslab


1) Try the bundled example
--------------------------

To see WeightsLab working end to end without writing any code, start a bundled
example like the classification example (--cls). It run a small experiment on a classification task:

.. code-block:: bash

   weightslab start example --cls

Then, in another terminal, launch the UI and open the URL printed by the command:

.. code-block:: bash

   weightslab start


2) Local integration in your own Python script (MNIST)
-------------------------------------------------------

Below is your MNIST CNN training pattern, first instrumented with TensorBoard,
then with TensorBoard removed and replaced by WeightsLab.

.. code-block:: python
   :linenos:
   :class: wl-diff-lines

   import torch
   import torch.nn as nn
   import torch.optim as optim
   -  from torch.utils.tensorboard import SummaryWriter
   from torchvision import datasets, transforms
   +  import weightslab as wl


   class CNN(nn.Module):
       def __init__(self):
           super().__init__()
   +      self.input_shape = (1, 28, 28)  # Weightslab necessary input shape for MNIST
           self.net = nn.Sequential(
               nn.Conv2d(1, 32, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(32, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Flatten(),
               nn.Linear(64 * 7 * 7, 10),
           )

       def forward(self, x):
           return self.net(x)


   cfg = {
       "device": "auto",
       "data_root": "./data",
       "data": {
           "train_loader": {
               "batch_size": 64,
           }
       },
       "optimizer": {
           "lr": 1e-3,
       },
   }
   device = "cuda" if torch.cuda.is_available() and cfg["device"] in ["auto", "cuda"] else "cpu"

   train_ds = datasets.MNIST(cfg["data_root"], train=True, download=True, transform=transforms.ToTensor())
   -  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["data"]["train_loader"]["batch_size"], shuffle=True)

   -  model = CNN().to(device)
   -  optimizer = optim.Adam(model.parameters(), lr=cfg.get("optimizer", {}).get("lr", 1e-3))
   -  loss = nn.CrossEntropyLoss(reduction="none")
   -  writer = SummaryWriter(log_dir="./runs/mnist_baseline")
   +  hp = wl.watch_or_edit(cfg, flag="hyperparameters")
   +  model = wl.watch_or_edit(CNN().to(device), flag="model", device=device)
   +  optimizer = wl.watch_or_edit(
   +      optim.Adam(model.parameters(), lr=cfg.get("optimizer", {}).get("lr", 1e-3)),
   +      flag="optimizer",
   +  )
   +  loss = wl.watch_or_edit(
   +      nn.CrossEntropyLoss(reduction="none"),
   +      flag="loss",
   +      signal_name="train/loss",
   +      per_sample=True,
   +      log=True,
   +  )
   +  train_loader = wl.watch_or_edit(
   +      train_ds,
   +      flag="data",
   +      loader_name="train_loader",
   +      batch_size=cfg["data"]["train_loader"]["batch_size"],
   +      shuffle=True,
   +      is_training=True,
   +  )
   +  wl.serve(serving_grpc=True, serving_cli=True)
   +  wl.start_training(timeout=3)

   step = 0
   while 1:
   +      with wl.guard_training_context:
   -      inputs, labels = next(iter(train_loader))
   +          inputs, uids, labels, metadata = next(iter(train_loader))
              inputs, labels = inputs.to(device), labels.to(device)
              optimizer.zero_grad()
              logits = model(inputs)
   -      loss_per_sample = loss(logits, labels)
   +          loss_per_sample = loss(logits, labels, batch_ids=uids, preds=logits)
              loss_per_sample.mean().backward()
              optimizer.step()
      if step % 20 == 0:
          print(f"Loss: {loss_per_sample.mean().item():.4f}")
      step += 1

   -  writer.close()
   +  wl.keep_serving()


Use Weightslab Studio (UI)
--------------------------

For a full visual experiment monitoring workflow (agent, samples, tags, discard/restore, plots), deploy the
Weights Studio web app with the bundled CLI.

**By default the UI runs unsecured (HTTP, no gRPC auth) — no certificates are generated.**
Pass ``--certs`` to generate (if missing) and use TLS certificates + a gRPC auth token:

.. code-block:: bash

   weightslab start              # unsecured HTTP (default)
   weightslab start --certs      # secured HTTPS + gRPC auth (run `weightslab se` first)

.. important::

   When using certs, it is prefered to set manually the ``WEIGHTSLAB_CERTS_DIR`` environment variable so the training backend and any new
   terminal use the **same** certificates — it is the single source of truth for TLS/auth. Please note that this step has to be done before starting the experiment.

Run ``weightslab``, ``weightslab help``, or ``weightslab -h`` to see the banner and the full
command reference (``se``, ``start``, ``start example ...``).

To stop the UI, press ``Ctrl+C`` in the terminal running ``weightslab start``.

Prefer a terminal over a browser? ``weightslab cli`` opens an interactive
console connected to the running experiment (pause/resume, status, evaluate,
tag/discard samples, query the agent, …) — no UI container required:

.. code-block:: bash

   weightslab cli

Full reference for both — every ``weightslab`` subcommand and every console
command, with all flags and defaults — lives in :doc:`user_commands`.


.. tip::

   **Let an AI agent integrate WeightsLab for you.**

   The repository ships with ``AGENTS.md`` — a compact context file that gives
   any AI coding assistant (Claude, Copilot, Cursor, …) a complete picture of
   the WeightsLab API.  Open your training script, attach ``AGENTS.md`` as
   context, and ask:

   .. code-block:: text

      "Using the context in AGENTS.md, integrate WeightsLab into this training script."

   The agent will wire up your model, data loader, loss, and hyperparameters in
   a few edits — no manual API lookup needed.


Recommended next reading
------------------------
Now that you run the classification task and try WeightsLab, you can integrate it into your training script.
To do so, please read the following:

- ``four_way_approach``: understand model/data/hyperparameters/logger together.
