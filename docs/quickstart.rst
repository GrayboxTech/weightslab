Quickstart
==========

This page gives you a practical, minimal path to get WeightsLab running.
If you prefer to start from examples, see :doc:`examples/index` right after this setup.

Prerequisites
-------------

- Python v3.10+ installed
- A virtual environment tool like ``venv`` or Conda (optional).
- Your training project available locally.

Install WeightsLab
------------------

Create and activate a virtual environment and install WeightsLab.

.. code-block:: bash

   python -m pip install weightslab

.. tip::

   For reproducible experiments, you can install in a virtual environment with the following command:

   .. code-block:: bash

      # From the repository root
      python -m venv .venv

      # Windows PowerShell
      .\.venv\Scripts\Activate.ps1
      # Linux/macOS
      # source .venv/bin/activate


Try the bundled example
~~~~~~~~~~~~~~~~~~~~~~~

To see WeightsLab working end to end without writing any code, start one of the bundled
examples (--cls, --seg, --det, --2d_det, --3d_det).
It run a small bundled experiment:

.. code-block:: bash

   weightslab start example --cls

Then, in another terminal, launch the UI and open the URL printed by the command:

.. code-block:: bash

   weightslab start


Local integration in your own Python script (MNIST)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same MNIST CNN, three ways, each shown as a diff: ``-`` lines go away,
``+`` lines are what WeightsLab adds. The first tab starts from a plain PyTorch
script, the other two from one already wired to an experiment tracker. The
**Copy** button on a diff block drops the ``-`` lines and the ``+`` markers, so
what lands in your clipboard is the runnable WeightsLab version.

Migrating a real codebase? The full guides are
:doc:`migration/from_tensorboard` and :doc:`migration/from_wandb`.

.. tab-set::

   .. tab-item:: WeightsLab Integration

      Starting from a plain PyTorch loop, with no tracker of any kind. The
      ``+`` lines are everything WeightsLab needs; the ``-`` lines are what it
      replaces.

      .. code-block:: python
         :linenos:
         :class: wl-diff-lines

         import torch
         import torch.nn as nn
         import torch.optim as optim
         from torchvision import datasets, transforms
         +
         +  import weightslab as wl


         class CNN(nn.Module):
             def __init__(self):
                 super().__init__()
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

         model = CNN().to(device)
         optimizer = optim.Adam(model.parameters(), lr=cfg.get("optimizer", {}).get("lr", 1e-3))
         -  loss = nn.CrossEntropyLoss()
         +  loss = nn.CrossEntropyLoss(reduction="none")
         +
         + # Wrap your objects with WeightsLab to watch and edit them in real time.
         + ## Wrap the hyperparameters first: the studio edits this dict in place.
         +  hp = wl.watch_or_edit(cfg, flag="hyperparameters")
         +
         + ## Wrap the model and the optimizer next.
         +  model = wl.watch_or_edit(model, flag="model", device=device)
         +  optimizer = wl.watch_or_edit(optimizer, flag="optimizer")
         +
         + ## Then the loss. The reduction="none" above is what makes it one value
         + ## *per sample*, which is what lets the studio sort the grid by loss and
         + ## take you from a spike in the curve to the images that caused it.
         +  loss = wl.watch_or_edit(
         +      loss,
         +      flag="loss",
         +      signal_name="train/loss",
         +      log=True,
         +  )
         +
         + ## And the dataset, which comes back as a tracked dataloader.
         +  train_loader = wl.watch_or_edit(
         +      train_ds,
         +      flag="data",
         +      loader_name="train_loader",
         +      batch_size=hp["data"]["train_loader"]["batch_size"],
         +      shuffle=True,
         +      is_training=True,
         +  )
         +
         + # Finally start the WeightsLab backend and keep it running while you train.
         +  wl.serve(serving_grpc=True, serving_cli=True)

         step = 0
         while True:
         +      ## guard_training_context is how pause/resume and the train/test
         +      ## split work -- without it, Play/Pause and the stats misbehave.
         +      with wl.guard_training_context:
         -      inputs, labels = next(iter(train_loader))
         +          inputs, uids, labels = next(train_loader)
         +          inputs, labels = inputs.to(device), labels.to(device)
         +          optimizer.zero_grad()
         +          logits = model(inputs)
         -          loss_per_sample = loss(logits, labels)
         +          loss_per_sample = loss(logits, labels, batch_ids=uids, preds=logits.argmax(1, keepdim=True))
         +          loss_per_sample.mean().backward()
         +          optimizer.step()

             if step % 20 == 0:
                 print(f"Loss: {loss_per_sample.mean().item():.4f}")
             step += 1

      Three things to notice. The tracked ``train_loader`` yields
      ``(inputs, uids, labels)`` , those ``uids`` are what tie a loss value back
      to the sample that produced it, which is why they are handed to the loss
      as ``batch_ids``. The loop is open-ended: you stop it from the studio, not
      with a step budget. And it does not start on its own , run
      ``weightslab start`` in another terminal and press **Play**.

   .. tab-item:: WeightsLab From TensorBoard

      ``SummaryWriter`` is a file handle you push numbers into. WeightsLab has
      no equivalent object: you wrap the thing that *produces* the number once,
      and it reports itself from then on, so the ``add_scalar`` call, and the
      global ``step`` bookkeeping it needs, both leave the loop.

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

         model = CNN().to(device)
         optimizer = optim.Adam(model.parameters(), lr=cfg.get("optimizer", {}).get("lr", 1e-3))
         -  loss = nn.CrossEntropyLoss()
         +  loss = nn.CrossEntropyLoss(reduction="none")   # one value per sample, not per batch
         -  writer = SummaryWriter(log_dir="./runs/mnist_baseline")
         +
         + # Wrap your objects with WeightsLab to watch and edit them in real time.
         + ## Wrap the hyperparameters first
         +  hp = wl.watch_or_edit(cfg, flag="hyperparameters")
         +
         + ## Wrap the model and optimizer next
         +  model = wl.watch_or_edit(model, flag="model", device=device)
         +  optimizer = wl.watch_or_edit(
         +      optimizer,
         +      flag="optimizer",
         +  )
         +
         + ## Then wrap the loss and the dataset
         +  loss = wl.watch_or_edit(
         +      loss,
         +      flag="loss",
         +      signal_name="train/loss",
         +      log=True,
         +  )
         +  train_loader = wl.watch_or_edit(
         +      train_ds,
         +      flag="data",
         +      loader_name="train_loader",
         +      batch_size=hp["data"]["train_loader"]["batch_size"],
         +      shuffle=True,
         +      is_training=True,
         +  )
         +
         + # Finally start the WeightsLab backend and keep it running while you train.
         +  wl.serve(serving_grpc=True, serving_cli=True)

         step = 0
         while True:
         +      with wl.guard_training_context:
         -      inputs, labels = next(iter(train_loader))
         +          inputs, uids, labels = next(train_loader)
         +          inputs, labels = inputs.to(device), labels.to(device)
         +          optimizer.zero_grad()
         +          logits = model(inputs)
         -          loss_per_sample = loss(logits, labels)
         +          loss_per_sample = loss(logits, labels, batch_ids=uids, preds=logits.argmax(1, keepdim=True))
         +          loss_per_sample.mean().backward()
         +          optimizer.step()
         -          writer.add_scalar("train/loss", loss_per_sample.mean().item(), step)

             if step % 20 == 0:
                 print(f"Loss: {loss_per_sample.mean().item():.4f}")
             step += 1

         -  writer.close()

      The loop body has no reporting code left in it at all. What it gained
      instead is ``guard_training_context`` (pause/resume and the train/test
      split) and ``batch_ids=uids`` (which sample each loss value belongs to).

   .. tab-item:: WeightsLab From W&B

      ``wandb.log()`` is a call you make at every point you want a number
      recorded; a watched object records itself. The other change worth
      noticing is ``cfg``: ``wandb.config`` is frozen once ``init()`` returns,
      whereas watched hyperparameters stay editable from the studio while the
      run goes on, so read them out of the dict each step rather than caching
      them in locals.

      .. code-block:: python
         :linenos:
         :class: wl-diff-lines

         import torch
         import torch.nn as nn
         import torch.optim as optim
         -  import wandb
         from torchvision import datasets, transforms
         +  import weightslab as wl


         class CNN(nn.Module):
             def __init__(self):
                 super().__init__()
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
         -  wandb.init(project="mnist", config=cfg)
         -  cfg = dict(wandb.config)   # a record of the run, frozen from here on
         device = "cuda" if torch.cuda.is_available() and cfg["device"] in ["auto", "cuda"] else "cpu"

         train_ds = datasets.MNIST(cfg["data_root"], train=True, download=True, transform=transforms.ToTensor())
         -  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["data"]["train_loader"]["batch_size"], shuffle=True)

         model = CNN().to(device)
         -  wandb.watch(model, log="all")
         optimizer = optim.Adam(model.parameters(), lr=cfg.get("optimizer", {}).get("lr", 1e-3))
         -  loss = nn.CrossEntropyLoss()
         +  loss = nn.CrossEntropyLoss(reduction="none")   # one value per sample, not per batch
         +
         + # Wrap your objects with WeightsLab to watch and edit them in real time.
         + ## wandb.config becomes a live dict -- edit it from the studio mid-run
         +  hp = wl.watch_or_edit(cfg, flag="hyperparameters")
         +
         + ## wandb.watch(model) becomes a watched model, plus the optimizer W&B never saw
         +  model = wl.watch_or_edit(model, flag="model", device=device)
         +  optimizer = wl.watch_or_edit(
         +      optimizer,
         +      flag="optimizer",
         +  )
         +
         + ## wandb.log({"train/loss": ...}) becomes a watched loss that logs itself
         +  loss = wl.watch_or_edit(
         +      loss,
         +      flag="loss",
         +      signal_name="train/loss",
         +      log=True,
         +  )
         +
         + ## and the dataset becomes the tracked table -- no wandb.Table to build
         +  train_loader = wl.watch_or_edit(
         +      train_ds,
         +      flag="data",
         +      loader_name="train_loader",
         +      batch_size=hp["data"]["train_loader"]["batch_size"],
         +      shuffle=True,
         +      is_training=True,
         +  )
         +
         + # wandb.init() becomes: start the backend, keep it up while you train.
         +  wl.serve(serving_grpc=True, serving_cli=True)

         step = 0
         while True:
         +      with wl.guard_training_context:
         -      inputs, labels = next(iter(train_loader))
         +          inputs, uids, labels = next(train_loader)
         +          inputs, labels = inputs.to(device), labels.to(device)
         +          optimizer.zero_grad()
         +          logits = model(inputs)
         -          loss_per_sample = loss(logits, labels)
         +          loss_per_sample = loss(logits, labels, batch_ids=uids, preds=logits.argmax(1, keepdim=True))
         +          loss_per_sample.mean().backward()
         +          optimizer.step()
         -          wandb.log({"train/loss": loss_per_sample.mean().item()}, step=step)

             if step % 20 == 0:
                 print(f"Loss: {loss_per_sample.mean().item():.4f}")
             step += 1

         -  wandb.finish()


      One thing has no equivalent, and is not meant to: there are no sweeps.
      WeightsLab is built around staying inside *one* run and steering it,
      raise the learning rate when the curve flattens, discard the samples
      poisoning it, keep going. If you need a sweep, keep the tool you sweep
      with.


Notebook Code with Google Colab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by opening this notebook:

- `WeightsLab Colab Quickstart <https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Colab/wl-colab-quickstart.ipynb>`_


Use Weightslab Studio (UI)
--------------------------

For a full visual experiment monitoring workflow (agent, samples, tags, discard/restore, plots), deploy the
Weights Studio web app with the bundled CLI.

**Without certificates the UI runs unsecured (HTTP, no gRPC auth).** Once you have
generated them with ``weightslab se``, ``weightslab start`` finds them in
``$WEIGHTSLAB_CERTS_DIR`` (else ``~/.weightslab-certs``) and serves HTTPS + gRPC auth
automatically, the same rule the training backend applies, so both sides agree:

.. code-block:: bash

   weightslab se                 # once: generate TLS certificates + a gRPC auth token
   weightslab start              # HTTPS + gRPC auth when certs exist, HTTP otherwise
   weightslab start --no-certs   # force plain HTTP even when certs exist

.. important::

   When using certs, it is prefered to set manually the ``WEIGHTSLAB_CERTS_DIR`` environment variable so the training backend and any new
   terminal use the **same** certificates, it is the single source of truth for TLS/auth. **Please note that this step has to be done before starting the experiment.**

Run ``weightslab``, ``weightslab help``, or ``weightslab -h`` to see the banner and the full
command reference (``se``, ``start``, ``start example ...``).

To stop the UI, press ``Ctrl+C`` in the terminal running ``weightslab start``.

Prefer a terminal over a browser? ``weightslab cli`` opens an interactive
console connected to the running experiment (pause/resume, status, evaluate,
tag/discard samples, query the agent, …), no UI container required:

.. code-block:: bash

   weightslab cli

Full reference for both, every ``weightslab`` subcommand and every console
command, with all flags and defaults, lives in :doc:`user_commands`.


.. tip::

   **Let an AI agent integrate WeightsLab for you.**

   The repository ships with ``AGENTS.md``, a compact context file that gives
   any AI coding assistant (Claude, Copilot, Cursor, …) a complete picture of
   the WeightsLab API.  Open your training script, attach ``AGENTS.md`` as
   context, and ask:

   .. code-block:: text

      "Using the context in AGENTS.md, integrate WeightsLab into this training script."

   The agent will wire up your model, data loader, loss, and hyperparameters in
   a few edits, no manual API lookup needed. Otherwise use the :doc:`agent_quickstart` to connect the integrated OpenCode agent to a running experiment and have it generate code for you from the UI.


Recommended next reading
------------------------
Now that you run the classification task and try WeightsLab, you can integrate it into your training script.
To do so, please read the following:

- :doc:`agent_quickstart`: connect the natural-language agent to a running
  experiment in four steps.
- :doc:`usage/good_practice/index`: good coding practices with WeightsLab.
- :doc:`four_way_approach`: understand WeightsLab's four-way approach to model/data/hyperparameters/logger and their integrations.
