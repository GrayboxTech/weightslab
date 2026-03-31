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
[![Version](https://img.shields.io/badge/pypi-v1.0.4-5865F2?style=flat&logoColor=white)](https://pypi.org/project/weightslab/)
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


## Play our demo below:

<ul class="tabbed">
  <li>
      <a href="https://sandbox.graybx.com/vla">
      <p style="text-indent:20px;">MNIST</p>
    </a>
  </li>
  <li>
      <a href="https://sandbox.graybx.com/vla">
      <p style="text-indent:20px;">VLA</p>
    </a>
  </li>
  <li>
      <a href="https://sandbox.graybx.com/bdd8k/clean">
      <p style="text-indent:20px;">BDD</p>
    </a>
  </li>
</ul>

## Getting Started


## Getting Started
### Installation

Define a Python environment (Python >= 3.10, <3.15)
```bash
python -m venv weightslab_venv
./weightslab_venv/Scripts/activate
```
Or install directly on your machine.

Install our framework:
```bash
python -m pip install weightslab
```

Deploy our interface with Docker:
```bash
cd ./ui
docker compose up -d
```

> [!IMPORTANT]
> For a detailed installation guide, please see the [Installation Documentation](https://grayboxtech.github.io/weightslab/latest/quickstart.html).


## Cookbook

Check out our materials, which include examples ranging from toys to more complex models and experiments.

Quickstart examples:
- [WeightsLab - Classification toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch/ws-classification)
- [WeightsLab - Segmentation toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch/ws-segmentation)
- [WeightsLab - Detection toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch/ws-detection)
- [WeightsLab - Classification toy (PyTorch Lightning)](https://github.com/GrayboxTech/weightslab/tree/dev/weightslab/examples/PyTorch_Lightning/ws-classification)


## Documentation (API + SDK)

* <div>
  <a href="https://grayboxtech.github.io/weightslab/latest/index.html">
    <p>Documentation</p>
  </a>
</div>


## Community

Graybx is building a wonderful community of AI researchers and engineers.
Are you interested in joining our project? Contact us at hello [at] graybx [dot] com

<!--
### Citation

If you publish work that uses Graybx, please cite Graybx as follows:

```bibtex
@article{graybx2025,
  title={Graybox: A Friendly BlackBox interactive approach},
  author={Luigi, Alex, Marc, And Guillaume},
  year={2025}
}
```
-->
