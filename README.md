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
      <a href="https://sandbox.graybx.com/mnist">
      <p style="text-indent:20px;">MNIST Demo</p>
    </a>
  </li>
  <li>
      <a href="https://sandbox.graybx.com/vla">
      <p style="text-indent:20px;">VLA Demo</p>
    </a>
  </li>
  <li>
      <a href="https://sandbox.graybx.com/bdd8k/clean">
      <p style="text-indent:20px;">BDD Demo</p>
    </a>
  </li>
</ul>


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
- [WeightsLab - Classification toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch/ws-classification)
- [WeightsLab - Segmentation toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch/ws-segmentation)
- [WeightsLab - Detection toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch/ws-detection)
- [WeightsLab - Classification toy (PyTorch Lightning)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch_Lightning/ws-classification)


## Configuration

WeightsLab and Weights Studio are configured through environment variables.
A fully-commented `.env` template is included at the repository root — copy it and adjust for your setup:

```bash
cp .env .env.local   # or edit .env directly
```

| Category | Key variables |
|---|---|
| **Logging** | `WEIGHTSLAB_LOG_LEVEL`, `WEIGHTSLAB_LOG_TO_FILE`, `WEIGHTSLAB_ROOT_LOG_DIR` |
| **gRPC server** | `GRPC_BACKEND_HOST`, `GRPC_BACKEND_PORT`, `GRPC_MAX_MESSAGE_BYTES` |
| **Watchdog** | `GRPC_WATCHDOG_STUCK_SECONDS`, `GRPC_WATCHDOG_INTERVAL_SECONDS`, `GRPC_WATCHDOG_RESTART_THRESHOLD`, `GRPC_WATCHDOG_EXIT_ON_STUCK` |
| **Data / cache** | `WL_MAX_PREVIEW_CACHE_SIZE`, `WL_PREVIEW_CACHE_WARMUP_WAIT_MS`, `WL_DEFAULT_THUMBNAIL_SIZE`, `WEIGHTSLAB_SAVE_PREDICTIONS_IN_H5` |
| **AI keys** | `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY` |
| **Agent config** | `AGENT_CONFIG_PATH` |
| **Weights Studio** | `VITE_SERVER_HOST`, `VITE_SERVER_PORT`, `VITE_HISTOGRAM_MAX_BINS`, `ENVOY_HOST`, `ENVOY_PORT` |

`AGENT_CONFIG_PATH` lets you point the data agent to a custom directory that contains `agent_config.yaml`.
If set, WeightsLab looks for `<AGENT_CONFIG_PATH>/agent_config.yaml` before fallback locations.

> Full documentation with all variables and their descriptions: [docs/configuration.rst](docs/configuration.rst)


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
