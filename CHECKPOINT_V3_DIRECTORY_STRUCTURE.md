# Checkpoint System V3 - Directory Structure & Hash Ordering

## Hash Format: HP_MODEL_DATA

The 24-byte experiment hash is composed of three 8-byte components in this order:
```
HP(8 bytes) + MODEL(8 bytes) + DATA(8 bytes) = 24 bytes total
```

Example hash: `a1b2c3d4e5f6g7h8i9j0k1l2`
- **HP hash**: `a1b2c3d4` (bytes 0-7)
- **Model hash**: `e5f6g7h8` (bytes 8-15)
- **Data hash**: `i9j0k1l2` (bytes 16-23)

## Directory Structure

```
root_log_dir/
├── data/                    # Global data files
├── logs/                    # Training logs
└── checkpoints/
    ├── manifest.yaml        # Tracks all hashes with timestamps
    ├── models/              # Model checkpoints
    │   └── {hash}/
    │       ├── {hash}_step_000100.pt
    │       ├── {hash}_step_000200.pt
    │       └── {hash}_architecture.pkl
    ├── HP/                  # Hyperparameter configs
    │   └── {hash}/
    │       └── {hash}_config.yaml
    └── data/                # Data state files
        └── {hash}/
            └── {hash}_data_state.yaml
```

Where `{hash}` is the 24-byte experiment hash like `a1b2c3d4e5f6g7h8i9j0k1l2`.

## Manifest File

The `manifest.yaml` file tracks all experiment hashes with metadata:

```yaml
latest_hash: a1b2c3d4e5f6g7h8i9j0k1l2
last_updated: '2026-01-21T10:30:00.123456'
experiments:
  a1b2c3d4e5f6g7h8i9j0k1l2:
    hp_hash: a1b2c3d4
    model_hash: e5f6g7h8
    data_hash: i9j0k1l2
    created: '2026-01-21T09:00:00.000000'
    last_used: '2026-01-21T10:30:00.123456'
  f0f0f0f0e5f6g7h8i9j0k1l2:
    hp_hash: f0f0f0f0
    model_hash: e5f6g7h8
    data_hash: i9j0k1l2
    created: '2026-01-21T08:00:00.000000'
    last_used: '2026-01-21T08:45:00.000000'
```

## Loading Most Recent Checkpoints

### Get Latest Hash

```python
from weightslab.components import CheckpointManagerV2

manager = CheckpointManagerV2('experiments')

# Get the most recently used hash
latest_hash = manager.get_latest_hash()
print(f"Latest hash: {latest_hash}")
```

### Get All Hashes Sorted

```python
# Get all hashes sorted by creation time (newest first)
hashes = manager.get_all_hashes(sort_by='created')
for entry in hashes:
    print(f"Hash: {entry['hash']}")
    print(f"  Created: {entry['created']}")
    print(f"  Last used: {entry['last_used']}")
    print(f"  HP: {entry['hp_hash']}, Model: {entry['model_hash']}, Data: {entry['data_hash']}")

# Or sort by last used
hashes = manager.get_all_hashes(sort_by='last_used')
```

### Find Hashes by Component

```python
# Find all experiments with the same HP hash
hp_hash = "a1b2c3d4"
matching_hashes = manager.get_hashes_by_component(hp_hash=hp_hash)
print(f"Experiments with HP hash {hp_hash}: {matching_hashes}")

# Find experiments with specific model and data, any HP
model_hash = "e5f6g7h8"
data_hash = "i9j0k1l2"
matching = manager.get_hashes_by_component(
    model_hash=model_hash,
    data_hash=data_hash
)
print(f"Experiments with model={model_hash}, data={data_hash}: {matching}")

# Find exact match for all three components
matching = manager.get_hashes_by_component(
    hp_hash="a1b2c3d4",
    model_hash="e5f6g7h8",
    data_hash="i9j0k1l2"
)
```

### Load Latest Checkpoint

```python
import torch as th
from pathlib import Path

manager = CheckpointManagerV2('experiments')

# Get latest hash
latest_hash = manager.get_latest_hash()

# Find latest model checkpoint in models/{hash}/
model_dir = manager.models_dir / latest_hash
checkpoint_files = sorted(model_dir.glob(f"{latest_hash}_step_*.pt"))

if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]  # Last one
    checkpoint = th.load(latest_checkpoint)

    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load config
config_file = manager.hp_dir / latest_hash / f"{latest_hash}_config.yaml"
if config_file.exists():
    import yaml
    with open(config_file) as f:
        config_data = yaml.safe_load(f)
        config = config_data['hyperparameters']

# Load data state
data_file = manager.data_checkpoint_dir / latest_hash / f"{latest_hash}_data_state.yaml"
if data_file.exists():
    with open(data_file) as f:
        data_state = yaml.safe_load(f)
```

## Comparing Hashes

### What Changed Between Two Experiments?

```python
from weightslab.components import ExperimentHashGenerator

gen = ExperimentHashGenerator()

hash1 = "a1b2c3d4e5f6g7h8i9j0k1l2"
hash2 = "f0f0f0f0e5f6g7h8i9j0k1l2"

changed = gen.compare_hashes(hash1, hash2)
# Returns: {'hp'}  # Only HP changed

# Breaking down the hashes:
# hash1: a1b2c3d4 | e5f6g7h8 | i9j0k1l2
#        HP        | Model    | Data
# hash2: f0f0f0f0 | e5f6g7h8 | i9j0k1l2
#        HP (diff) | Model ✓  | Data ✓
```

### Find Related Experiments

```python
# Current hash
current_hash = "a1b2c3d4e5f6g7h8i9j0k1l2"

# Extract component hashes
hp_hash = current_hash[0:8]      # "a1b2c3d4"
model_hash = current_hash[8:16]  # "e5f6g7h8"
data_hash = current_hash[16:24]  # "i9j0k1l2"

# Find all experiments with same model architecture
same_model_exps = manager.get_hashes_by_component(model_hash=model_hash)

# Find all experiments with same HP
same_hp_exps = manager.get_hashes_by_component(hp_hash=hp_hash)

# Find all experiments with same data
same_data_exps = manager.get_hashes_by_component(data_hash=data_hash)
```

## Complete Example: Loading and Resuming

```python
from weightslab.components import CheckpointManagerV2, get_checkpoint_system
import torch as th
import yaml

# Initialize manager
manager = CheckpointManagerV2('experiments')

# Get the most recent experiment
latest_hash = manager.get_latest_hash()
print(f"Resuming from hash: {latest_hash}")

# Analyze what's in this hash
hp_hash = latest_hash[0:8]
model_hash = latest_hash[8:16]
data_hash = latest_hash[16:24]
print(f"  HP: {hp_hash}, Model: {model_hash}, Data: {data_hash}")

# Load hyperparameters
config_file = manager.hp_dir / latest_hash / f"{latest_hash}_config.yaml"
with open(config_file) as f:
    config_data = yaml.safe_load(f)
    config = config_data['hyperparameters']

# Load model architecture
arch_file = manager.models_dir / latest_hash / f"{latest_hash}_architecture.pkl"
if arch_file.exists():
    import dill
    with open(arch_file, 'rb') as f:
        model = dill.load(f)
else:
    # Create model from config
    model = create_model(config)

# Load latest checkpoint
model_dir = manager.models_dir / latest_hash
checkpoints = sorted(model_dir.glob(f"{latest_hash}_step_*.pt"))
if checkpoints:
    latest_ckpt = checkpoints[-1]
    checkpoint = th.load(latest_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    resume_step = checkpoint['step']
    print(f"Resuming from step {resume_step}")

# Load data state
data_file = manager.data_checkpoint_dir / latest_hash / f"{latest_hash}_data_state.yaml"
with open(data_file) as f:
    data_state = yaml.safe_load(f)
    discarded_uids = set(data_state.get('discarded', []))
    tags = data_state.get('tags', {})

# Resume training with checkpoint system
checkpoint_system = get_checkpoint_system(
    root_log_dir='experiments',
    checkpoint_frequency=100
)

for step in range(resume_step + 1, 10000):
    # Training step
    loss = train_step(model, data_loader)

    # Automatic checkpointing
    checkpoint_system.on_training_step(step)
```

## Benefits of This Structure

### 1. **Component Isolation**
- Models in `checkpoints/models/{hash}/`
- Configs in `checkpoints/HP/{hash}/`
- Data states in `checkpoints/data/{hash}/`

### 2. **Easy Filtering**
```python
# Find all experiments with a specific model architecture
model_hash = "e5f6g7h8"
experiments = manager.get_hashes_by_component(model_hash=model_hash)

# Load each experiment's HP to compare
for exp_hash in experiments:
    config_file = manager.hp_dir / exp_hash / f"{exp_hash}_config.yaml"
    # Load and analyze...
```

### 3. **Chronological Tracking**
The manifest maintains creation and last-used timestamps, making it easy to:
- Find most recent experiments
- Clean up old experiments
- Track experiment history

### 4. **Component Reuse**
If you change only HP:
```
Old hash: a1b2c3d4_e5f6g7h8_i9j0k1l2
New hash: f0f0f0f0_e5f6g7h8_i9j0k1l2
          ^^^^^^^^  (HP changed)
                    ^^^^^^^^^^^^^^^^ (model and data same)
```

The model and data directories for the old hash can be referenced/reused since those components haven't changed.

## API Reference

### CheckpointManagerV2 Manifest Methods

```python
# Get latest hash
latest: Optional[str] = manager.get_latest_hash()

# Get all hashes sorted
hashes: List[Dict] = manager.get_all_hashes(sort_by='created')  # or 'last_used'

# Each dict in hashes has:
# {
#     'hash': '24-byte hash',
#     'hp_hash': '8-byte HP hash',
#     'model_hash': '8-byte model hash',
#     'data_hash': '8-byte data hash',
#     'created': 'ISO timestamp',
#     'last_used': 'ISO timestamp'
# }

# Find by component
matches: List[str] = manager.get_hashes_by_component(
    hp_hash="a1b2c3d4",      # Optional
    model_hash="e5f6g7h8",   # Optional
    data_hash="i9j0k1l2"     # Optional
)
```

### ExperimentHashGenerator

```python
from weightslab.components import ExperimentHashGenerator

gen = ExperimentHashGenerator()

# Compare hashes
changed: Set[str] = gen.compare_hashes(hash1, hash2)
# Returns: {'hp'}, {'model'}, {'data'}, or combinations

# Get component hashes from current state
hashes: Dict = gen.get_component_hashes()
# Returns: {'hp': '...', 'model': '...', 'data': '...', 'combined': '...'}
```
