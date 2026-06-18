<div align="center">
  <img
    src="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/GitHub_banner_WL.png"
    alt="Graybox Logo."
    height="250"
  />

  <!-- <p>
    <a href="https://graybx.com/">Website</a> |
    <a href="https://grayboxtech.github.io/weightslab/">Docs</a> |
    <a href="https://join.slack.com/t/grayboxcommunity/shared_invite/zt-3gtjg2p4y-UmSQC9pgAs8ZNE_gy4D~5A">Slack</a> |
    <a href="https://www.linkedin.com/company/graybx-com/">Linkedin</a>
  </p> -->
</div>

<div align="center">

[![Tests](https://github.com/GrayboxTech/weightslab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GrayboxTech/weightslab/actions/workflows/ci.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/badge/python-3.10%2B-5865F2?style=flat&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/GrayboxTech/weightslab/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/status-release-5865F2?style=flat&logoColor=white)](https://pypi.org/project/weightslab/)
[![Version](https://img.shields.io/pypi/v/weightslab?style=flat&color=5865F2&logo=pypi&logoColor=white)](https://pypi.org/project/weightslab/)
[![Slack](https://img.shields.io/badge/Slack-Join_Us-5865F2?style=flat&logo=slack&logoColor=white)](https://join.slack.com/t/grayboxcommunity/shared_invite/zt-3gtjg2p4y-UmSQC9pgAs8ZNE_gy4D~5A)

</div>



</pred>
</pred style="font-style: italic;">
WeightsLab — Inspect, Edit, and Evolve Neural Networks
By Graybx.
</pre>
</div>


## About WeightsLab
WeightsLab is a powerful tool for editing and inspecting data & AI models.

### What Problems Does It Solve?
WeightsLab addresses critical AI research challenges:

* Dataset insights & optimization
* Overfitting and training plateau
* Over/Under parameterization

### Key Capabilities
The granular statistics and interactive paradigm enable powerful workflows:

* Monitor granular insights on data samples, signals, and weight parameters
* Use the AI agent to:
  * Create slices of data and discard them for the next training iteration
  * Discard low-quality samples from training data
  * Iterative pruning or growing of the architectures (INCOMING feature)


## Quick Start

### Requirements

- **Docker Desktop v4.77 or newer** — required to deploy the Weights Studio UI (`weightslab ui launch`).
- **Docker Compose v2** (the `docker compose` CLI plugin, bundled with Docker Desktop) — *recommended*. The legacy v1 standalone binary (`docker-compose`, **≥ 1.27**) also works: `weightslab ui launch` auto-detects whichever is installed and uses it. Compose v1 below 1.27 is **not** supported.
- **Python >=3.10, <3.15** — to install and run the `weightslab` framework.


### Installation
Install directly on your machine.
> [!TIP]
> Setting a clean Python environment:
> ```bash
> python -m venv weightslab_venv
> ./weightslab_venv/Scripts/activate
> ```

Install our framework:
```bash
pip install weightslab
```

Deploy our interface:
```bash
weightslab ui launch
```

The command `weightslab ui launch` removes any stale weightslab/weights_studio Docker resources that could break the launch, then starts the UI stack. **By default, it runs unsecured (HTTP, no gRPC auth) — no certificates are generated. However, communication are not safe.**
> [!TIP]
> To run secured communication, pass the arguments `--certs`:
> ```bash
> weightslab ui launch --certs   # generates TLS certs + a gRPC auth token if missing, then launches secured
> ```
> When using certs, set `WEIGHTSLAB_CERTS_DIR` so the training backend and any new terminal use the **same** certificates (it is the single source of truth). `weightslab se` and `weightslab ui launch --certs` print the exact export/`setx` command for your shell. You can also generate certs up front with `weightslab se`.

<!--
Want to see it working end-to-end without writing any code? Start a bundled demo (it installs the example's requirements first — no prompts — then trains and serves until you stop it with `Ctrl+C`). It defaults to classification; pick another with a flag:
```bash
weightslab start example          # classification (default)
weightslab start example --cls    # classification
weightslab start example --seg    # segmentation
weightslab start example --det    # detection
weightslab start example --clus   # clustering
weightslab start example --gen    # generation
``` -->
<!-- Then run `weightslab ui launch` in another terminal and open `https://localhost:5173`. -->

<!-- Run `weightslab`, `weightslab help`, or `weightslab -h` at any time to see the banner and the full command reference. -->

> [!IMPORTANT]
> For a detailed installation guide and more advanced features, please see the [Installation Documentation](https://grayboxtech.github.io/weightslab/latest/quickstart.html).


## Quick Training Example

### Step-by-Step Integration

1. **Add the import** at the top of your script:
   ```python
   import weightslab as wl  # ← Include our SDK into your experiment
   ```

2. **Wrap your parameters** with WeightsLab tracking:
   ```python
   model = wl.watch_or_edit(parameters, flag='hp', ...)  # ← Now WeightsLab monitors your parameters and allow you to update them from your UI
   ```

3. **Wrap your model** with WeightsLab tracking:
   ```python
   model = wl.watch_or_edit(SimpleModel(...), flag='model', ...)  # ← Now WeightsLab monitors your model state
   ```

4. **Wrap your optimizer** with WeightsLab tracking:
   ```python
   optimizer = wl.watch_or_edit(optim.Adam(...), flag='opt', ...)  # ← Tracks optimizer state and update optimizer learning rate from your UI
   ```

5. **Wrap your signal** with WeightsLab tracking:
   ```python
    train_criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag='signal', name="train_loss/sample", per_sample=True, log=True)  # ← Tracks this signal and others (metrics, ..etc) from your UI
    test_criterion = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"), flag='signal', name="test_loss/sample", per_sample=True, log=False)  # ← Tracks this signal and others (metrics, ..etc) from your UI - Plot is disabled, only per sample signal
   ```

6. **Wrap your dataset** with WeightsLab tracking:
   ```python
    train_loader = wl.watch_or_edit(train_dataset, flag='data', loader_name="train_loader", ...)  # ← Tracks this dataset and others (validation, test) from your UI
    val_loader = wl.watch_or_edit(val_dataset, flag='data', loader_name="val_loader", ...)  # ← Tracks this dataset and others (validation, test) from your UI
   ```

7. **Run your training script** as usual:
   ```bash
   python train.py
   ```

8. **Launch the UI** in another terminal:
   ```bash
   weightslab ui launch
   ```

9. **Open your browser** to `https://localhost:5173` to track experiment evoluation and results!


### Details

<details>
<summary><b>Here's a complete example showing how to integrate WeightsLab into a basic PyTorch training script:</b></summary>

```python
#!/usr/bin/env python3
"""
Basic PyTorch training script with WeightsLab integration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import weightslab as wl  # ← Import WeightsLab (uses TLS certs from WEIGHTSLAB_CERTS_DIR if present)


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__(input_shape=12, output_shape=2)
        self.linear = nn.Linear(input_shape, 1)

    def forward(self, x):
        return self.linear(x)


# Create synthetic data
def create_data(n_samples=1000):
    X = torch.randn(n_samples, 10)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return TensorDataset(X, y)


# Main training function
def main():
    # Initialize WeightsLab - this creates certificates automatically!
    print("🚀 Initializing WeightsLab...")

    # Load hyperparameters (from YAML if present)
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters = wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        defaults=parameters,
        poll_interval=1.0,
    ) or {}  # Wrap the hyperparameters

    # Wrap your model and optimizer with WeightsLab
    model = wl.watch_or_edit(
      SimpleModel(
        input_shape=parameters.get('model', {}).get('input_shape', 10),
        output_shape=parameters.get('model', {}).get('output_shape', 1)
      )
    )  # ← WeightsLab tracks your model
    optimizer = wl.watch_or_edit(
      optim.Adam(model.parameters(), lr=parameters.get('model', {}).get('optimizer', {}).get('lr', 0.01)),
      flag='optimizer'
    )  # ← WeightsLab tracks optimizer

    # Create and wrap criterion
    criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss",
        signal_name="train-loss-CE",
        log=True  # If log is False, only save per sample value, not plot criterion
    )

    # Create data and dataloader
    dataset = create_data()
    train_loader = wl.watch_or_edit(
        dataset,
        flag="data",
        loader_name="loader",
        batch_size=parameters.get('data', {}).get('train_loader', {}).get('batch_size', 8),
        shuffle=parameters.get('data', {}).get('train_loader', {}).get('shuffle', False),
        is_training=True,  # Is it the training dataloader ?
        compute_hash=parameters.get('data', {}).get('train_loader', {}).get('compute_hash', True),  # Compute hash for train loader to allow dynamic augmentations and dataset sanity check
        preload_labels=parameters.get('data', {}).get('train_loader', {}).get('preload_labels', True),
        preload_metadata=parameters.get('data', {}).get('train_loader', {}).get('preload_metadata', True),
        enable_h5_persistence=parameters.get('data', {}).get('train_loader', {}).get('enable_h5_persistence', True),
        num_workers=parameters.get('data', {}).get('train_loader', {}).get('num_workers', 4)
    )

    # Training loop
    print("🏃 Starting training...")
    print("💡 Launch the UI with: weightslab ui launch")
    print("🌐 Open browser to: https://localhost:5173")
    n_epochs = parameters.get('n_epochs')
    pbar = tqdm.tqdm(range(n_epochs), desc='Training..') if parameters.get('tqdm_display', False) else range(n_epochs)
    for epoch in pbar:  # Train for 5 epochs
        total_loss = 0

        for batch_X, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
```
</details>

<details>
<summary><b>Migrating from wandb? See the diff:</b></summary>

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
+def byte_adjusted_loss(ctx): return ctx.subscribed_value / ctx.image_bytes  # chains on image_bytes
+
 def main():
@@ -15,29 +16,38 @@

     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     parameters = {"batch_size": 128, "lr": 1e-3}

-    wandb.init(project="cifar10")
-
     transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
     ])
     train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=transform)
     test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
-    train_loader = DataLoader(train_set, batch_size=parameters["batch_size"], shuffle=True, num_workers=2)
-    test_loader  = DataLoader(test_set,  batch_size=256,                              num_workers=2)
+    wl.watch_or_edit(parameters, flag="hyperparameters")  # live-editable in UI
+
+    train_loader = wl.watch_or_edit(
+        train_set, flag="data", loader_name="train_loader",
+        batch_size=parameters["batch_size"], shuffle=True, is_training=True)
+    test_loader  = wl.watch_or_edit(
+        test_set,  flag="data", loader_name="test_loader",
+        batch_size=256, shuffle=False, is_training=False)

     model = models.resnet18(weights=None)
     model.fc = nn.Linear(model.fc.in_features, 10)
     model = model.to(device)
     optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"])

-    criterion = nn.CrossEntropyLoss()
-    accuracy  = MulticlassAccuracy(num_classes=10).to(device)
+    criterion = wl.watch_or_edit(
+        nn.CrossEntropyLoss(), flag="loss", signal_name="loss/CE")
+    accuracy  = wl.watch_or_edit(
+        MulticlassAccuracy(num_classes=10).to(device),
+        flag="metric", signal_name="acc")
+
+    wl.serve(serving_grpc=True)

     for epoch in range(1, args.epochs + 1):
         model.train()
         accuracy.reset()
         for x, y in train_loader:
+            with guard_training_context:
                 x, y = x.to(device), y.to(device)
                 logits = model(x)
                 loss = criterion(logits, y)
                 optimizer.zero_grad()
                 loss.backward()
                 optimizer.step()
                 accuracy.update(logits, y)
-            wandb.log({"train/loss": loss.item()})
-        wandb.log({"train/acc": accuracy.compute().item(), "epoch": epoch})
+            wl.save_signals(preds_raw=logits, targets=y,
+                            signals={"metric/accuracy": accuracy.compute().item()})

         model.eval()
         accuracy.reset()
         with torch.no_grad():
             for x, y in test_loader:
+                with guard_testing_context:
                     x, y = x.to(device), y.to(device)
                     accuracy.update(model(x), y)
-        wandb.log({"test/acc": accuracy.compute().item(), "epoch": epoch})
+                wl.save_signals(preds_raw=logits, targets=y,
+                                signals={"metric/accuracy": accuracy.compute().item()})

-    wandb.finish()
+    wl.keep_serving()
```
</details>



### What WeightsLab Does Automatically

<!-- - 📊 **Tracks model parameters** and optimizer state in real-time -->
<!-- - 🔄 **Enables model editing (incoming feature)** and hyperparameter tuning through the UI -->
- 📊 **Experiment tracking for reproducibility**
- 📈 **Provides live metrics** and visualization in the web UI
- 🔄 **Enables data supervision during training** and experiment hyperparameter tuning through the UI


## Examples

### Local examples
After starting the UI, launch a local experiment with the command:
```bash
weightslab start example            # classification (default)
# weightslab start example --cls    # classification
# weightslab start example --seg    # segmentation
# weightslab start example --det    # detection
# weightslab start example --clus   # clustering
# weightslab start example --gen    # generation
```

###  Cloud examples
Find our demos online. The password is **graybx**.
<a href="https://preview.graybx.com/">
  <p style="text-indent:20px;">DEMOS</p>
</a>


## AI Agent

WeightsLab can run its data agent in two modes:

- Local provider with Ollama
- Cloud provider with OpenRouter

Use local Ollama when you want a fully local setup and do not need cloud-hosted models (see more details installation from the documentation).
Use OpenRouter when you want larger hosted models and model selection directly from Weightslab UI.


### Cloud OpenRouter
You can either preconfigure OpenRouter in `agent_config.yaml` / `.env`, or initialize it interactively from Weightslab UI.

#### SDK Configuration

Example static configuration:

```yaml
agent:
  provider: openrouter
  openrouter_model: meta-llama/llama-3.3-70b-instruct
  fallback_to_local: false
  # openrouter_api_key: ${OPENROUTER_API_KEY}
```

Environment variable:

```bash
export OPENROUTER_API_KEY=your_key_here
```

#### Interactive setup from Weightslab UI

OpenRouter models can be initialized and set directly from the UI:
1. Click in the agent bar or double-click to expand the agent window.
2. Type `/init`.
3. Choose either:
   - `A` Enter your OpenRouter API key manually
   - `B` Use the OpenRouter OAuth flow
4. Select a model from the fetched list, then confirm.

The default OpenRouter model, as recommended by Graybx, is `meta-llama/llama-3.3-70b-instruct`.

### Typical Usage Flow

1. Start your WeightsLab backend (e.g., "main.py").
2. Start Weightslab UI.
3. If you use Ollama, query the agent directly.
4. If you use OpenRouter and the agent is not configured yet, type `/init`.
5. Ask natural-language data operations such as sorting, filtering, slicing, and inspection requests. You can also ask questions about the data.
6. Use `/model` to try another cloud model without re-entering the key.
7. Use `/reset` if you want to clear the current connection and start over.


## Documentation (API + SDK)

Find our documentation [online](https://grayboxtech.github.io/weightslab/latest/index.html).


## Contributing & onboarding

New here (human or AI coding agent)? Start with [AGENTS.md](AGENTS.md) — it
captures the cross-repo architecture (weightslab backend ↔ weights_studio
frontend via the shared proto), the module maps, the `wl.watch_or_edit`
integration pattern, where tests live, and the gotchas that aren't obvious from
any single file. It's the fastest way to orient before a first change.


## Community

Graybx is building a wonderful community of AI researchers and engineers.
Are you interested in joining our project? Contact us at hello [at] graybx [dot] com
