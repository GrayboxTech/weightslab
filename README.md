<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/weightslab-banner-product-screen.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/weightslab-banner-product-screen.png" />
    <img alt="WeightsLab Banner" src="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/weightslab-banner-product-screen.png" width="100%" />
  </picture>
<br>

</div>
<div align="center">
  <h1>Built for ML Teams working with messy real-world data</h1>
  <p>Pause training, mine live loss signals to surface mislabels, class imbalance & outliers,<br>then curate your image, video, text & LiDAR data, without restarting.</p>
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
  ·
  <a href="https://youtu.be/WetZU_J7Tg8">Demo (LLM/Coming soon)</a>

</p>

<br>

## Overview

WeightsLab hooks into your existing PyTorch training loop and exposes a live UI where you can inspect per-sample signals, edit the dataset, and steer training. Without restarting.

## WeightsLab in Motion
<div align="center">
  <img width="960" height="540" alt="Image" src="https://github.com/user-attachments/assets/2e8f8846-efcd-4188-9d60-fee0674d1105" />
  <!-- <sub><a href="https://youtu.be/GBBDDaJQLWk">▶ Watch full demo</a></sub> -->
</div>

## Quickstart
For a detailed installation guide and advanced configuration: [Documentation](https://grayboxtech.github.io/weightslab/latest/quickstart.html).

**1. Install & Launch**
```bash
pip install weightslab
```

```bash
weightslab start ./my_experiment_directory # launch the UI
```

> [!TIP]
> Studio with HTTP Secure Deployment. Run before starting this command to set certificates.
> ```bash
> weightslab se  # set certificates for encryption
> ```

**2. Start**

**In the cloud directly**

[![Start coding with GCollab](https://img.shields.io/badge/Start_coding_with_GCollab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Colab/wl-colab-quickstart.ipynb)


**Or Wrap your training script locally**
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



## How can you use it ?

<h4>1. Find bad data fast</h4> Pause training mid-run before bad data are learnt, sort samples by loss, spot mislabels and outliers before they impact your model.

<h4>2. Fix and resume without re-starting</h4> Relabel or drop samples live, then continue training from the same checkpoint.

<h4>3. Catch model regressions early on</h4> Analyze per-sample loss trajectories to see exactly where the model is struggling.

## Resources & Community
<details>
<summary><b>Training script with Weightslab - Step-by-Step Integration</b></summary>

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
<summary><b>Migrating ?</b></summary>

See more details at https://grayboxtech.github.io/weightslab/latest/quickstart.html#local-integration-in-your-own-python-script-mnist.

</details>

<details open>
<summary><b>Agent: chat with your training run (OpenCode)</b></summary>

WeightsLab ships two distinct agent surfaces: the backend SDK agent for data-manipulation
queries, and a local [OpenCode](https://opencode.ai)-backed agent with a full bash/file
toolset that can restart training, edit your code, and run recurring `/loop` monitoring
jobs. See the [Agent docs](https://grayboxtech.github.io/weightslab/latest/agent.html) for
how the two connect, how to point either one at a local model, and the full `/loop`
reference.

</details>


## Contributing and OnBoarding

New here (human or AI coding agent)? Start with [AGENTS.md](AGENTS.md); it
captures the cross-repo architecture (weightslab backend ↔ weights_studio
frontend via the shared proto), the module maps, the integration pattern, where tests live, and the gotchas that aren't obvious from
any single file. It's the fastest way to orient before a first change and contribution.

## Community

We're building a community of ML engineers around data-centric training tooling.
Interested in contributing or just want to say hi? → hello [at] graybx [dot] com


## Citation
If WeightsLab is part of your development process / project / publication, please cite us ❤️:

```latex
@misc{wl2024,
  title = {WeightsLab - A monitoring platform for ML teams building real‑world AI at scale},
  year = {2024},
  note = {Software available from http://github.com/GrayboTech/weightslab},
  url={https://graybx.com},
  author = {GrayBx}
}
```
</details>
