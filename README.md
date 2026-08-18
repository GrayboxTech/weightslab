<div align="center">
  <a href="https://grayboxtech.github.io/weightslab/latest/index.html">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/weightslab-banner-product-screen.png" />
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/weightslab-banner-product-screen.png" />
      <img alt="WeightsLab Banner" src="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/weightslab-banner-light.png" width="100%" />
    </picture>
  </a>
<br>

</div>
<div align="center">
  <h1>Built for AI Engineers working with messy real-world data</h1>
  <p>Pause training, mine live loss signals to surface mislabels, class imbalance & outliers,<br>then curate your image, video & LiDAR data, without restarting.</p>
</div>

<br>
</div>
<div align="center">
  <a href="https://github.com/GrayboxTech/weightslab/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" /></a>
  <a href="https://github.com/GrayboxTech/weightslab/stargazers"><img src="https://img.shields.io/github/stars/GrayboxTech/weightslab?style=flat&color=5865F2" alt="Stars" /></a>
  <a href="https://pypi.org/project/weightslab/"><img src="https://img.shields.io/pypi/v/weightslab?style=flat&color=5865F2&logo=pypi&logoColor=white" alt="Version" /></a>
  <a href="https://pepy.tech/project/weightslab"><img src="https://img.shields.io/pepy/dt/weightslab?style=flat&color=5865F2&logo=pypi&logoColor=white" alt="PyPI - Downloads" /></a>
  <a href="https://github.com/GrayboxTech/weightslab/actions"><img src="https://img.shields.io/badge/CI-passing-brightgreen?style=flat&logo=githubactions&logoColor=white" alt="CI" /></a>
</div>

<p align="center">
  <a href="https://graybx.com">Website</a>
  ·
  <a href="https://grayboxtech.github.io/weightslab/latest/quickstart.html">Docs</a>
  ·
  <a href="https://youtu.be/GBBDDaJQLWk">Demo (Images)</a>
  ·
  <a href="https://youtu.be/pSng0aIXGCY">Demo (VLA)</a>
  ·
  <a href="https://youtu.be/WetZU_J7Tg8">Demo (LiDAR)</a>

</p>

<br>

## Overview

WeightsLab hooks into your existing PyTorch training loop and exposes a live UI where you can inspect per-sample signals, edit the dataset, and steer training. Without restarting.

## Weightslab in Motion
<div align="center">
  <img width="960" height="540" alt="Image" src="https://github.com/user-attachments/assets/2e8f8846-efcd-4188-9d60-fee0674d1105" />
  <!-- <sub><a href="https://youtu.be/GBBDDaJQLWk">▶ Watch full demo</a></sub> -->
</div>

## Quickstart

**1. Install & Launch**
```bash
pip install weightslab
```
```bash
weightslab start  # launch the UI
```

**2. Start in the cloud**

[![Start coding with GCollab](https://img.shields.io/badge/Start_coding_with_GCollab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Colab/wl-colab-quickstart.ipynb)

**OR Wrap your training script locally**
```python
# wrap the objects in your training script

import weightslab as wl
...
model  = wl.watch_or_edit(model, flag='model')
optim  = wl.watch_or_edit(optim, flag='opt')
loss   = wl.watch_or_edit(loss, flag='signal', name="loss", per_sample=True, log=True)
loader = wl.watch_or_edit(dataset, flag='data', loader_name="train")
...
wl.serve(serving_grpc=True, serving_cli=False)
...
```

> [!TIP]
> Quick examples to get started
> ```bash
> weightslab start example            # classification (default)
> weightslab start example --cls      # classification
> weightslab start example --seg      # segmentation
> weightslab start example --det      # detection
> weightslab start example --clus     # clustering
> ```
> Explore our [sandbox](https://sandbox.graybx.com/).
For a detailed installation guide and advanced configuration: [Documentation](https://grayboxtech.github.io/weightslab/latest/quickstart.html).

<br>

## How can you use it ?

<h4>1. Find bad data fast</h4> Pause training mid-run, sort samples by loss, spot mislabels and outliers before they impact your model.

<h4>2. Fix and resume without re-starting</h4> Relabel or drop samples live, then continue training from the same checkpoint.

<h4>3. Catch model regressions early on</h4> Analyze per-sample loss trajectories to see exactly where the model is struggling.

<br>
<br>

## Resources & Community
<details>
<summary><b>Training script with Weightslab - Step-by-Step Integration</b></summary>

<br>

1. **Add the import** at the top of your script:
```python
   import weightslab as wl
```

2. **Wrap your parameters, model, optimizer, signals, and dataset:**
```python
   parameters      = wl.watch_or_edit(parameters, flag='hp',     ...) # ← WeightsLab monitors your parameters and lets you update them from the UI
   model           = wl.watch_or_edit(model, flag='model', ...) # ← WeightsLab monitors your model state
   optimizer       = wl.watch_or_edit(optim.Adam(...),                         flag='opt',    ...) # ← Tracks optimizer state and lets you update the learning rate from the UI

   train_criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),  flag='signal', name="train_loss/sample", per_sample=True, log=True)   # ← Wrap and plot your signals on the UI
   test_criterion  = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),  flag='signal', name="test_loss/sample",  per_sample=True, log=False)  # ← Per-sample only, plot disabled

   train_loader    = wl.watch_or_edit(train_dataset, flag='data', loader_name="train_loader", ...)  # ← Track your training dataset
   val_loader      = wl.watch_or_edit(val_dataset,   flag='data', loader_name="val_loader",   ...)  # ← Track your validation dataset

```

3. **Run your script, then launch the UI in a separate terminal:**
```bash
   python train.py
   weightslab start
```

4. **Open your browser** at the URL printed by `weightslab start` and inspect your training in real time.

</details>

<details>
<summary><b>Training script with Weightslab - Full Example</b></summary>

<br>

```python
#!/usr/bin/env python3
"""
Basic PyTorch training script with WeightsLab integration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import weightslab as wl


class SimpleModel(nn.Module):
    def __init__(self, input_shape=10, output_shape=1):
        super().__init__()
        self.linear = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        return self.linear(x)


def create_data(n_samples=1000):
    X = torch.randn(n_samples, 10)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return TensorDataset(X, y)


def main():
    parameters = wl.watch_or_edit({}, flag="hyperparameters", poll_interval=1.0) or {}

    model     = wl.watch_or_edit(SimpleModel(), flag='model')
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=0.01), flag='optimizer')
    criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag="loss", signal_name="train-loss-CE", log=True)
    loader    = wl.watch_or_edit(create_data(), flag="data", loader_name="loader", batch_size=8, is_training=True)

    for epoch in range(parameters.get('n_epochs', 5)):
        total_loss = 0
        for batch_X, batch_y in loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Write the history of these samples every x steps
            if model.get_age() % 100 == 0:
                print(f'Dump signals history and dataframe at age {model.get_age()}')
                wl.write_history(
                    # path=None,  # Use root_log_dir by default, filename generated from parameters md5 hash
                    type_of_history="all",
                    graph_name=[
                        'train/clsf_instance',
                        'val/clsf_instance'
                    ],
                    # experiment_hash=None,  Default is 'last', i.e., current experiment hash
                    sample_id=['11', '29', '28', '27', '22'],
                    instance_id=[1, 2, 3]
                )

                # Dump the sample dataframe: all signals plus the loss_shape categorical tag,
                wl.write_dataframe(
                    columns=["signals", "tag:loss_shape"],
                    format='csv'
                    # sample_id=['0', '28']
                    # instance_id=[1, 2],
                )

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
```

</details>

<details>
<summary><b>Migrating from Weights & Biases?</b></summary>

## WeightsLab vs Weights & Biases

Weights & Biases (wandb) tracks experiments. WeightsLab connects training signals back to
the exact samples causing them — so you can fix your data, not just log it.

<br>

```diff
--- train_baseline.py
+++ train_wl.py
@@ -1,11 +1,12 @@
 import argparse
 import torch
 import torch.nn as nn
-from torch.utils.data import DataLoader
 from torchvision import datasets, transforms, models
 from torchmetrics.classification import MulticlassAccuracy

-import wandb
+import weightslab as wl
+from weightslab.components.global_monitoring import (
+    guard_training_context, guard_testing_context)
+
+@wl.signal(name="byte_adjusted_loss", subscribe_to="loss/CE")
+def byte_adjusted_loss(ctx): return ctx.subscribed_value / ctx.image_bytes
+
 def main():
@@ -15,29 +16,38 @@
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     parameters = {"batch_size": 128, "lr": 1e-3}

-    wandb.init(project="cifar10")
-
     transform = transforms.Compose([...])
     train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
     test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
-    train_loader = DataLoader(train_set, batch_size=parameters["batch_size"], shuffle=True, num_workers=2)
-    test_loader  = DataLoader(test_set,  batch_size=256, num_workers=2)
+    wl.watch_or_edit(parameters, flag="hyperparameters")  # live-editable in UI
+
+    train_loader = wl.watch_or_edit(
+        train_set, flag="data", loader_name="train_loader",
+        batch_size=parameters["batch_size"], shuffle=True, is_training=True)
+    test_loader  = wl.watch_or_edit(
+        test_set,  flag="data", loader_name="test_loader",
+        batch_size=256, shuffle=False, is_training=False)

     model     = models.resnet18(weights=None)
     model.fc  = nn.Linear(model.fc.in_features, 10)
     optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"])

-    criterion = nn.CrossEntropyLoss()
-    accuracy  = MulticlassAccuracy(num_classes=10).to(device)
+    criterion = wl.watch_or_edit(nn.CrossEntropyLoss(), flag="loss", signal_name="loss/CE")
+    accuracy  = wl.watch_or_edit(MulticlassAccuracy(num_classes=10).to(device), flag="metric", signal_name="acc")
+
+    wl.serve(serving_grpc=True)

     for epoch in range(1, args.epochs + 1):
         model.train()
         for x, y in train_loader:
+            with guard_training_context:
                 logits = model(x.to(device))
                 loss   = criterion(logits, y.to(device))
                 optimizer.zero_grad(); loss.backward(); optimizer.step()
                 accuracy.update(logits, y)
-            wandb.log({"train/loss": loss.item()})
-        wandb.log({"train/acc": accuracy.compute().item(), "epoch": epoch})
+            wl.save_signals(preds_raw=logits, targets=y,
+                            signals={"metric/accuracy": accuracy.compute().item()})

         model.eval()
         with torch.no_grad():
             for x, y in test_loader:
+                with guard_testing_context:
                     accuracy.update(model(x.to(device)), y)
-        wandb.log({"test/acc": accuracy.compute().item(), "epoch": epoch})
+                wl.save_signals(preds_raw=logits, targets=y,
+                                signals={"metric/accuracy": accuracy.compute().item()})

-    wandb.finish()
+    wl.keep_serving()
```

</details>

<details>
<summary><b>Documentation (API + SDK)</b></summary>

<br>

Find our documentation [online](https://grayboxtech.github.io/weightslab/latest/index.html).

</details>

</details>

<details>
<summary><b>Contributing & Onboarding</b></summary>

<br>

New here (human or AI coding agent)? Start with [AGENTS.md](AGENTS.md) — it
captures the cross-repo architecture (weightslab backend ↔ weights_studio
frontend via the shared proto), the module maps, the integration pattern, where tests live, and the gotchas that aren't obvious from
any single file. It's the fastest way to orient before a first change and contribution.

</details>

<details>
<summary><b>Community</b></summary>

<br>

We're building a community of ML engineers around data-centric training tooling.
Interested in contributing or just want to say hi? → hello [at] graybx [dot] com

</details>
