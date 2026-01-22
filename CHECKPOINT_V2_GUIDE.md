# Checkpoint System V2 - Migration Guide

## Overview

The new checkpoint system provides better organization, provenance tracking, and flexibility for managing experiment artifacts. This guide explains the new structure, how to use it, and how to migrate from the old system.

## New Directory Structure

```
root_log_dir/
├── data/                           # Global data directory
│   └── (array h5 files, datasets, etc.)
├── logs/                           # Global logs directory
│   └── (training logs, metrics, etc.)
└── checkpoints/                    # Experiment checkpoints
    ├── {exp_hash_A}/              # Unique experiment configuration
    │   ├── model/                  # Model checkpoints
    │   │   ├── {hash}_step_000100.pt
    │   │   ├── {hash}_step_000200.pt
    │   │   ├── {hash}_architecture.pkl
    │   │   └── {hash}_architecture.txt
    │   ├── hp/                     # Hyperparameter configs
    │   │   └── {hash}_config.yaml
    │   └── data/                   # Data sample backups
    │       └── {hash}_data.h5
    └── {exp_hash_B}/              # Another experiment
        └── ...
```

## What is an Experiment Hash?

An **experiment hash** is a unique 16-character identifier generated from:

1. **Model Architecture**: Layer types, connections, parameters (not weights)
2. **Hyperparameters**: Learning rate, batch size, optimizer settings, etc.
3. **Active Data Samples**: UIDs of non-discarded training samples

When **any** of these change, a new hash is generated, creating a new checkpoint directory. This ensures complete provenance tracking.

### Hash Change Triggers

- **Model Architecture Changes**: Adding/removing layers, changing layer sizes
- **Hyperparameter Changes**: Modifying learning rate, batch size, etc.
- **Data Changes**: Discarding samples, adding new data

## Key Features

### 1. Model Checkpoints
- **Saved every N steps** (e.g., every 100 training steps)
- **Weights only** for fast checkpointing
- **Architecture saved once** per hash when first needed

### 2. Hyperparameter Tracking
- **Config saved when hash changes** (architecture, hp, or data change)
- **Updated when parameters change** during training (e.g., batch_size adjustment)
- **YAML format** for human readability

### 3. Data Backups
- **H5 files backed up** when hash changes
- **Only main data files**, not large array h5 files
- **Automatic on hash generation**

## Usage Examples

### Basic Setup

```python
from weightslab.components import CheckpointManagerV2

# Initialize manager
checkpoint_manager = CheckpointManagerV2(root_log_dir="./experiments")

# Set up your model and config
model = MyModel()
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'adam',
}

# Generate initial hash
exp_hash, is_new = checkpoint_manager.update_experiment_hash(
    model=model,
    config=config,
    active_data_uids=data_loader.get_sample_uids()
)

print(f"Experiment hash: {exp_hash}")
```

### During Training

```python
# Training loop
for step in range(num_steps):
    # Training code...
    loss = train_step(model, data)

    # Save checkpoint every 100 steps
    if step % 100 == 0:
        checkpoint_manager.save_model_checkpoint(
            model=model,
            step=step,
            save_optimizer=True,
            metadata={'loss': loss, 'epoch': epoch}
        )

    # If you change hyperparameters during training
    if step == 5000:
        config['batch_size'] = 64  # Increase batch size
        checkpoint_manager.save_config(config)  # Update config
```

### Handling Architecture Changes

```python
# If you modify the model architecture
model = MyNewImprovedModel()  # Different architecture

# Update hash - this creates a new checkpoint directory
exp_hash, is_new = checkpoint_manager.update_experiment_hash(
    model=model,
    config=config,
    active_data_uids=data_loader.get_sample_uids()
)

if is_new:
    print(f"New experiment hash: {exp_hash}")
    print("New checkpoint directory created!")
```

### Handling Data Discarding

```python
# When you discard samples
data_loader.discard_samples(['sample_123', 'sample_456'])

# Update hash - data composition changed
exp_hash, is_new = checkpoint_manager.update_experiment_hash(
    model=model,
    config=config,
    active_data_uids=data_loader.get_active_sample_uids()  # Excludes discarded
)

# Optionally backup the new data state
if is_new:
    checkpoint_manager.save_data_backup(
        data_h5_path=data_loader.h5_file_path
    )
```

### Loading Checkpoints

```python
# Load latest checkpoint for current experiment
checkpoint_data = checkpoint_manager.load_latest_checkpoint(
    model=model,
    load_optimizer=True
)

if checkpoint_data:
    step = checkpoint_data['step']
    print(f"Resumed from step {step}")

# Load config
config = checkpoint_manager.load_config()
```

### Listing Experiments

```python
# Get all experiment hashes
hashes = checkpoint_manager.list_experiment_hashes()
print(f"Found {len(hashes)} experiments")

# Get info about specific experiment
info = checkpoint_manager.get_checkpoint_info(hashes[0])
print(f"Checkpoints: {info['model_checkpoints']}")
print(f"Architecture saved: {info['architecture_saved']}")
print(f"Configs: {info['configs']}")
print(f"Data backups: {info['data_backups']}")
```

## Integration with Weightslab

### Using with Ledgers

The checkpoint manager integrates with Weightslab's ledger system:

```python
from weightslab.backend.ledgers import register_model, register_optimizer

# Register model and optimizer
register_model('my_model', model)
register_optimizer('my_optimizer', optimizer)

# Checkpoint manager will use ledgers if model/optimizer not provided
checkpoint_manager.save_model_checkpoint(
    model_name='my_model',  # Will get from ledger
    optimizer_name='my_optimizer',
    step=100
)
```

### Integration Points

1. **Trainer Integration**: Add checkpoint manager to your trainer class
2. **Config Updates**: Hook config changes to `save_config()`
3. **Data Discard**: Hook sample discarding to `update_experiment_hash()`
4. **Periodic Saves**: Add checkpoint saves to training loop

## Migration from Old System

### Old System (`CheckpointManager`)

```python
from weightslab.components.checkpoint import CheckpointManager

# Old way
old_manager = CheckpointManager(root_directory="./experiments")
checkpoint_id = old_manager.dump(
    model_name='model',
    save_full_model=True
)
```

### New System (`CheckpointManagerV2`)

```python
from weightslab.components import CheckpointManagerV2

# New way
new_manager = CheckpointManagerV2(root_log_dir="./experiments")

# Set up hash first
exp_hash, _ = new_manager.update_experiment_hash(
    model=model,
    config=config,
    active_data_uids=sample_uids
)

# Save checkpoint
checkpoint_path = new_manager.save_model_checkpoint(
    model=model,
    step=step
)
```

### Key Differences

| Feature | Old System | New System |
|---------|-----------|------------|
| **Directory Structure** | Flat, numbered checkpoints | Organized by experiment hash |
| **Hash-based** | No | Yes |
| **Config Tracking** | Limited | Full YAML config saved |
| **Data Provenance** | No | Yes (hash includes data) |
| **Architecture Saving** | Optional per checkpoint | Once per hash |
| **Step Tracking** | Checkpoint ID | Explicit step numbers |

## Best Practices

### 1. Hash Updates
- Call `update_experiment_hash()` at the start of training
- Re-call when architecture, config, or data changes
- The system handles directory creation automatically

### 2. Checkpoint Frequency
- Save model checkpoints every 50-100 steps for small models
- Every 500-1000 steps for large models
- Adjust based on training time per step

### 3. Config Management
- Save config immediately after hash update
- Re-save when parameters change during training
- Include all relevant hyperparameters in config dict

### 4. Data Backups
- Only backup main data h5 files (not large arrays)
- Call `save_data_backup()` after hash changes
- Consider data size before backing up

### 5. Cleanup
- Periodically clean old experiment hashes you don't need
- Keep at least the last 2-3 hashes for safety
- Use `list_experiment_hashes()` to find old experiments

## Advanced Usage

### Custom Metadata

```python
# Add custom metadata to checkpoints
checkpoint_manager.save_model_checkpoint(
    model=model,
    step=step,
    metadata={
        'loss': current_loss,
        'accuracy': current_accuracy,
        'epoch': epoch,
        'notes': 'After learning rate adjustment'
    }
)
```

### Multiple Configs

```python
# Save different config variants
checkpoint_manager.save_config(config, config_name="config")
checkpoint_manager.save_config(optimizer_config, config_name="optimizer_config")
checkpoint_manager.save_config(data_config, config_name="data_config")
```

### Manual Hash Generation

```python
from weightslab.components import ExperimentHashGenerator

# Create hash generator separately
hash_gen = ExperimentHashGenerator()

# Generate hash
exp_hash = hash_gen.generate_hash(
    model=model,
    config=config,
    active_data_uids=sample_uids
)

# Check what changed
has_changed, components = hash_gen.has_changed(
    model=new_model,
    config=config,
    active_data_uids=sample_uids
)

if has_changed:
    print(f"Changed components: {components}")  # e.g., {'model', 'config'}
```

## Troubleshooting

### Hash Not Changing When Expected

**Problem**: You changed the model but hash didn't update.

**Solution**: Make sure you're passing the updated model to `update_experiment_hash()`:

```python
exp_hash, is_new = checkpoint_manager.update_experiment_hash(
    model=new_model,  # Pass new model
    config=config,
    active_data_uids=sample_uids
)
```

### Too Many Checkpoint Directories

**Problem**: New directory created too frequently.

**Cause**: Config or data UIDs changing when they shouldn't.

**Solution**:
- Ensure config dict is stable (no timestamps, random values)
- Ensure data UIDs list is sorted and consistent
- Use `has_changed()` to debug what's changing

### Checkpoint Loading Fails

**Problem**: `load_latest_checkpoint()` returns None.

**Causes**:
1. No checkpoints saved yet
2. Wrong exp_hash specified
3. Corrupted checkpoint file

**Solution**:
```python
# Check what exists
info = checkpoint_manager.get_checkpoint_info()
print(info)

# List all hashes
hashes = checkpoint_manager.list_experiment_hashes()
print(f"Available hashes: {hashes}")
```

## API Reference

### `CheckpointManagerV2`

#### `__init__(root_log_dir='root_experiment')`
Initialize checkpoint manager with root directory.

#### `update_experiment_hash(model=None, config=None, active_data_uids=None, force=False)`
Generate/update experiment hash. Returns `(exp_hash, is_new)`.

#### `save_model_checkpoint(model=None, model_name=None, step=None, save_optimizer=True, optimizer_name=None, metadata=None)`
Save model weights checkpoint. Returns checkpoint file path.

#### `save_model_architecture(model, model_name=None)`
Save full model architecture. Returns architecture file path.

#### `save_config(config, config_name='config')`
Save hyperparameter configuration. Returns config file path.

#### `save_data_backup(data_h5_path, backup_name=None)`
Backup data h5 file. Returns backup file path.

#### `load_latest_checkpoint(model=None, model_name=None, load_optimizer=True, optimizer_name=None, exp_hash=None)`
Load latest checkpoint. Returns checkpoint data dict.

#### `load_config(exp_hash=None)`
Load configuration. Returns config dict.

#### `list_experiment_hashes()`
List all experiment hashes. Returns list of hash strings.

#### `get_checkpoint_info(exp_hash=None)`
Get info about experiment checkpoints. Returns info dict.

### `ExperimentHashGenerator`

#### `generate_hash(model=None, config=None, active_data_uids=None)`
Generate experiment hash. Returns 16-char hash string.

#### `has_changed(model=None, config=None, active_data_uids=None)`
Check if config changed. Returns `(has_changed, changed_components)`.

#### `get_last_hash()`
Get last generated hash. Returns hash string or None.

#### `get_component_hashes()`
Get individual component hashes. Returns dict with 'model', 'config', 'data', 'combined'.

## Future Enhancements

Planned improvements for the checkpoint system:

1. **Automatic cleanup**: Remove old checkpoints based on age/count
2. **Compression**: Compress old checkpoints to save space
3. **Remote storage**: Support for S3, GCS, etc.
4. **Checkpoint comparison**: Tools to diff checkpoints
5. **Resume helpers**: Automatic resume from latest checkpoint
6. **Visualization**: Web UI to browse experiments

## Support

For issues, questions, or suggestions:
- Check the [Weightslab documentation](../readme.md)
- Open an issue on GitHub
- Contact the Weightslab team

---

**Last Updated**: January 2026
**Version**: 2.0.0
