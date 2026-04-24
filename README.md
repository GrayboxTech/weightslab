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


## Find our demos:
The password is **graybx**. More demo to come in the future!

  <a href="https://preview.graybx.com/">
    <p style="text-indent:20px;">DEMOS</p>
  </a>


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
> For a detailed installation guide and more advanced features, please see the [Installation Documentation](https://grayboxtech.github.io/weightslab/latest/quickstart.html).


## Quick Training Example

Here's a complete example showing how to integrate WeightsLab into a basic PyTorch training script:

```python
#!/usr/bin/env python3
"""
Basic PyTorch training script with WeightsLab integration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import weightslab as wl  # ← Import WeightsLab (auto-creates secure certs!)

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

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

    # Wrap your model and optimizer with WeightsLab
    model = wl.watch_or_edit(SimpleModel())  # ← WeightsLab tracks your model
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=0.01))  # ← WeightsLab tracks optimizer

    # Create data and dataloader
    dataset = create_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    print("🏃 Starting training...")
    for epoch in range(5):  # Train for 5 epochs
        total_loss = 0

        for batch_X, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = nn.functional.mse_loss(predictions, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

    print("✅ Training complete!")
    print("💡 Launch the UI with: weightslab ui docker launch")
    print("🌐 Open browser to: https://localhost:8080")

if __name__ == "__main__":
    main()
```

### Step-by-Step Integration

1. **Add the import** at the top of your script:
   ```python
   import weightslab as wl  # ← This line creates certificates automatically!
   ```

2. **Wrap your model** with WeightsLab tracking:
   ```python
   model = wl.watch_or_edit(SimpleModel())  # ← Now WeightsLab monitors your model
   ```

3. **Wrap your optimizer** with WeightsLab tracking:
   ```python
   optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=0.01))  # ← Tracks optimizer state
   ```

4. **Run your training script** as usual:
   ```bash
   python train.py
   ```

5. **Launch the UI** in another terminal:
   ```bash
   weightslab ui docker launch
   ```

6. **Open your browser** to `https://localhost:8080` to see live training metrics!

### What WeightsLab Does Automatically

- 🔐 **Creates TLS certificates** in `~/.weightslab-certs/`
- 🎫 **Generates secure auth tokens** for gRPC communication
- 📊 **Tracks model parameters** and optimizer state in real-time
- 📈 **Provides live metrics** and visualization in the web UI
- 🔄 **Enables model editing** and hyperparameter tuning through the UI


## Security

By default, WeightsLab runs in **unsecured mode** (NO TLS, HTTP, no gRPC authentification). To enable TLS encryption and gRPC authentication:

### Quick Setup + Launch
Setup secure environment AND launch Docker in one command:
```bash
weightslab ui docker se
export WEIGHTSLAB_CERTS_DIR='~/.weightslab-certs/'
```

This will:
1. Generate TLS certificates and gRPC auth token in `~/.weightslab-certs/`
2. Launch the Docker stack with security enabled
3. If setup fails, Docker still launches (unsecured fallback)

### Separate Steps (If Needed)
Setup secure environment only:
```bash
weightslab se
```

Then launch Docker later:
```bash
weightslab ui docker launch
```

### Custom Certificates Directory
Store certificates in a custom location using the `WEIGHTSLAB_CERTS_DIR` environment variable:
```bash
weightslab se "/path/to/my/certs"
export WEIGHTSLAB_CERTS_DIR=/path/to/my/certs
weightslab ui docker launch
```

### Options for Secure Environment Setup

**For `weightslab se` or `weightslab ui docker se`:**

| Flag | Effect |
|---|---|
| `--force-certs` | Regenerate certificates even if they already exist |
| `--no-auth` | Skip gRPC auth token generation (TLS only) |

**Examples:**
```bash
weightslab se --force-certs              # Regenerate certs only
weightslab se --no-auth                  # TLS only, no gRPC token
weightslab ui docker se                  # Setup + launch in one
weightslab ui docker se --force-certs    # Setup (regenerate) + launch
```

### What Gets Created?
After running `weightslab se` or `weightslab ui docker se`, your `~/.weightslab-certs/` directory contains:
- `backend-server.crt` — Backend TLS certificate
- `backend-server.key` — Backend TLS private key
- `ca.crt` — CA certificate
- `.grpc_auth_token` — gRPC authentication token (if not using `--no-auth`)

### Running Without Security
To explicitly run in unsecured mode, delete the certificates directory:
```bash
rm -r ~/.weightslab-certs
weightslab ui docker launch  # Falls back to unsecured HTTP
```

### Testing the Setup
**Secured environment:**
```bash
weightslab se                      # Setup certs and token
weightslab ui docker launch        # Launch Docker
python main.py                     # Backend finds certs, runs secured
```

**Unsecured environment:**
```bash
weightslab ui docker launch        # Launch Docker (no certs)
python main.py                     # Backend runs unsecured
```


## Cookbook

Check out our materials, which include examples ranging from toys to more complex models and experiments.

Quickstart examples:
- [WeightsLab - Classification toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch/ws-classification)
- [WeightsLab - Segmentation toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch/ws-segmentation)
- [WeightsLab - Detection toy (PyTorch)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch/ws-detection)
- [WeightsLab - Classification toy (PyTorch Lightning)](https://github.com/GrayboxTech/weightslab/tree/main/weightslab/examples/PyTorch_Lightning/ws-classification)


## Configuration

WeightsLab and Weightslab UI are configured through environment variables.
A fully commented .env template is included at the repository root — copy it and adjust for your setup:

```bash
cp .env .env.local   # or edit .env directly
```

| Category | Key variables |
|---|---|
| **Logging** | `WEIGHTSLAB_LOG_LEVEL`, `WEIGHTSLAB_LOG_TO_FILE`, `WEIGHTSLAB_ROOT_LOG_DIR` |
| **gRPC server** | `GRPC_BACKEND_HOST`, `GRPC_BACKEND_PORT`, `GRPC_MAX_MESSAGE_BYTES`, `GRPC_TLS_ENABLED`, `GRPC_TLS_CERT_DIR`, `GRPC_TLS_CERT_FILE`, `GRPC_TLS_KEY_FILE`, `GRPC_TLS_CA_FILE`, `GRPC_TLS_REQUIRE_CLIENT_AUTH`, `GRPC_AUTH_TOKEN`, `GRPC_AUTH_TOKENS` |
| **Watchdog** | `GRPC_WATCHDOG_STUCK_SECONDS`, `GRPC_WATCHDOG_INTERVAL_SECONDS`, `GRPC_WATCHDOG_RESTART_THRESHOLD`, `GRPC_WATCHDOG_EXIT_ON_STUCK` |
| **Data / cache** | `WL_MAX_PREVIEW_CACHE_SIZE`, `WL_PREVIEW_CACHE_WARMUP_WAIT_MS`, `WL_DEFAULT_THUMBNAIL_SIZE`, `WEIGHTSLAB_SAVE_PREDICTIONS_IN_H5` |
| **AI keys** | `OPENROUTER_API_KEY` |
| **Agent config** | `AGENT_CONFIG_PATH` |
| **Weightslab UI** | `VITE_SERVER_HOST`, `VITE_SERVER_PORT`, `VITE_HISTOGRAM_MAX_BINS`, `ENVOY_HOST`, `ENVOY_PORT` |

`AGENT_CONFIG_PATH` lets you point the data agent to a custom directory that contains `agent_config.yaml`.
If set, WeightsLab looks for `<AGENT_CONFIG_PATH>/agent_config.yaml` before fallback locations.

WeightsLab also reads TLS settings from registered runtime config (hyperparameters),
using config-first precedence over environment variables.

If TLS is enabled (`grpc_tls_enabled` in config or `GRPC_TLS_ENABLED` in env),
certificate path resolution is:
1. config file paths (`grpc_tls_cert_file`, `grpc_tls_key_file`, `grpc_tls_ca_file`)
2. env file paths (`GRPC_TLS_CERT_FILE`, `GRPC_TLS_KEY_FILE`, `GRPC_TLS_CA_FILE`)
3. config directory (`grpc_tls_cert_dir`)
4. env directory (`GRPC_TLS_CERT_DIR`)
5. default `~/certs` (`backend-server.crt`, `backend-server.key`, `ca.crt`)

TLS flags also follow config-first precedence:
`grpc_tls_enabled` then `GRPC_TLS_ENABLED`, and
`grpc_tls_require_client_auth` then `GRPC_TLS_REQUIRE_CLIENT_AUTH`.

> Full documentation with all variables and their descriptions: [docs/configuration.rst](docs/configuration.rst)


## AI Agent

WeightsLab can run its data agent in two modes:

- Local provider with Ollama
- Cloud provider with OpenRouter

Use local Ollama when you want a fully local setup and do not need cloud-hosted models.
Use OpenRouter when you want larger hosted models and model selection directly from Weightslab UI.

### Option A: Local Ollama

Start Ollama on the same machine as the WeightsLab backend and pull the model first with Docker:

```bash
ollama pull llama3.2:3b
ollama serve
```

Then configure the agent provider in `agent_config.yaml`:

```yaml
agent:
  provider: ollama
  ollama_model: llama3.2:3b
  ollama_host: localhost
  ollama_port: 11435
  fallback_to_local: false
```

In this mode, the agent is ready when the backend starts. You can open Weightslab UI and query the agent directly from the chat bar.

### Option B: Cloud OpenRouter

You can either preconfigure OpenRouter in `agent_config.yaml` / `.env`, or initialize it interactively from Weightslab UI.

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

Interactive setup from Weightslab UI:

1. Click in the agent bar or double-click to expand the agent window.
2. Type `/init`.
3. Choose either:
   - `A` Enter your OpenRouter API key manually
   - `B` Use the OpenRouter OAuth flow
4. Select a model from the fetched list, then confirm.

The default OpenRouter model, as recommended by Graybx, is `meta-llama/llama-3.3-70b-instruct`.

### Agent Commands in Weightslab UI

The agent input supports these commands:

- `/init` initializes the cloud provider flow for OpenRouter
- `/model` opens the model chooser and switches the active OpenRouter model
- `/reset` clears the current runtime agent connection and status

The agent history also records setup and model-change events as log-style entries, separate from normal agent responses.

### Agent Commands in the Backend CLI

The local WeightsLab CLI also exposes agent control and querying commands.

Available commands:

- `agent status` checks whether the agent is attached and ready.
- `agent init --api-key KEY [--model MODEL] [--timeout SEC]` initializes OpenRouter from the CLI.
- `agent models` lists models available for the configured OpenRouter key.
- `agent model MODEL_NAME` switches the active OpenRouter model.
- `agent reset` clears the current agent connection.
- `agent query <prompt>` sends a natural-language request through the agent.
- `query <prompt>` and `ask <prompt>` are shortcuts for `agent query`.

Examples:

```bash
agent status
agent init --api-key sk-or-... --model openai/gpt-4o-mini --timeout 20
agent models
agent model meta-llama/llama-3.3-70b-instruct
agent query discard all samples with loss > 5 and tag them as hard_examples
ask tag validation samples as goldset
```

### CLI Sample-ID Commands

The backend CLI supports editing data directly from `sample_id` values.

- `discard <sample_id> [sample_id2 ...]` marks samples as discarded.
- `undiscard <sample_id> [sample_id2 ...]` restores samples.
- `add_tag <sample_id> <tag> [sample_id2 ...]` applies the same tag to one or more samples.

Examples:

```bash
discard sample_001 sample_002 sample_003
undiscard sample_002
add_tag sample_001 goldset sample_002 sample_003
```

### Typical Usage Flow

1. Start your WeightsLab backend (e.g., "main.py").
2. Start Weightslab UI.
3. If you use Ollama, query the agent directly.
4. If you use OpenRouter and the agent is not configured yet, type `/init`.
5. Ask natural-language data operations such as sorting, filtering, slicing, and inspection requests. You can also ask questions about the data.
6. Use `/model` to try another cloud model without re-entering the key.
7. Use `/reset` if you want to clear the current connection and start over.


## Documentation (API + SDK)

* <div>
  <a href="https://grayboxtech.github.io/weightslab/latest/index.html">
    <p>Documentation</p>
  </a>
</div>


## Community

Graybx is building a wonderful community of AI researchers and engineers.
Are you interested in joining our project? Contact us at hello [at] graybx [dot] com
