<div align="center">
  <img
    src="https://raw.githubusercontent.com/GrayboxTech/.github/main/profile/GitHub_banner_WL.png"
    alt="Graybox Logo"
    height="250"
  />

  <p>
    <a href="https://graybx.com/">Website</a> |
    <a href="https://grayboxtech.github.io/weightslab/">Docs</a> |
    <a href="https://join.slack.com/t/grayboxcommunity/shared_invite/zt-3gtjg2p4y-UmSQC9pgAs8ZNE_gy4D~5A">Slack</a> |
    <a href="https://www.linkedin.com/company/graybx-com/">Linkedin</a>
  </p>
</div>

</pred>
</pred style="font-style: italic;">
WeightsLab — Inspect, Edit and Evolve Neural Networks
By Graybx.
</pre>
</div>

## About WeightsLab
WeightsLab is a powerful tool for editing and inspecting data & AI model weights, during training.

### What Problems Does It Solve?
WeightsLab addresses critical training challenges:

* Overfitting and training plateau
* Dataset insights & optimization
* Over/Under parameterization

### Key Capabilities
The granular statistics and interactive paradigm enables powerful workflows:

* Monitor granular insights on data samples and weights parameters
* Discard low quality samples by click or query
* Create slices of data and discard them during training
* Iterative pruning or growing of the architectures by click or query


## Getting Started
### Watch our demo below:

<div>
  <a href="https://www.loom.com/share/5d04822a0933427d971d320f64687730">
    <p>Demo Video - Watch Video</p>
  </a>
  <!-- <a href="https://www.loom.com/share/5d04822a0933427d971d320f64687730">
    <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/5d04822a0933427d971d320f64687730-00001.gif">
  </a> -->
</div>

### Installation
Define a python environment
```bash
python -m venv weightslab_venv
./weightslab_venv/Scripts/activate
```
or install and use conda.

Clone and install the framework (CLI based interaction):

```bash
git clone https://github.com/GrayboxTech/weightslab.git
cd weightslab
pip install -e .
```

Clone the UI repository (UI based interaction; cf. loom video):
```bash
git clone git@github.com:GrayboxTech/weightslab_ui.git
cd weightslab_ui
pip install -r ./requirements.txt
```

### Documentation (API + SDK)

WeightsLab includes a Sphinx documentation site in `docs/`.

Build once:
```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Serve on localhost with auto-reload:
```bash
sphinx-autobuild docs docs/_build/html --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`


### Cookbook

Check out our materials, with examples from toy to more complex models.

Quickstart examples:
- [WeightsLab - Classification toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch/ws-classification)
- [WeightsLab - Segmentation toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch/ws-segmentation)
- [WeightsLab - Detection toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch/ws-detection)
- [WeightsLab - Classification toy (PyTorch Lightning)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch_Lightning/ws-classification)

### New docs: practical use-case + Lightning

- Use-case walkthrough (commented, end-to-end): `docs/usecases.rst`
- PyTorch Lightning integration (including multi-GPU): `docs/pytorch_lightning.rst`

#### PyTorch Lightning + multi-GPU (quick snippet)

```python
import torch
import pytorch_lightning as pl

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if use_gpu else 0
multi_gpu = gpu_count > 1

trainer = pl.Trainer(
  max_epochs=max_epochs,
  accelerator="gpu" if use_gpu else "cpu",
  devices=gpu_count if multi_gpu else 1,
  strategy="ddp" if multi_gpu else "auto",
  sync_batchnorm=multi_gpu,
  use_distributed_sampler=multi_gpu,
  logger=False,
  enable_checkpointing=False,
)
```

When using WeightsLab in Lightning steps, keep passing `batch_ids` to tracked losses/signals to preserve per-sample traceability.

<!-- ### Documentation -->

### Community

Graybx is building a wonderful community of AI researchers and engineers.
Are you interested in joining our project ? Contact us at hello [at] graybx [dot] com


### Citation

If you publish work that uses Graybx, please cite Graybx as follows:

```bibtex
@article{graybx2025,
  title={Graybox: A Friendly BlackBox interactive approach},
  author={Luigi, Alex, Marc, And Guillaume},
  year={2025}
}
```
