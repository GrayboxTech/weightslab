# Checkpoint System V3 - Comprehensive Guide

## Overview

The checkpoint system now uses a **24-byte hash** composed of three 8-byte components:
- **Model Architecture Hash** (8 bytes)
- **Hyperparameters Hash** (8 bytes)
- **Data State Hash** (8 bytes - UIDs, discard status, tags)

This allows you to **identify what changed** between experiment versions by comparing hash segments.

## Key Features

### 1. Separate Hash Components

```python
# Hash structure: MMMMMMMM CCCCCCCC DDDDDDDD
#                 Model(8) Config(8) Data(8)

# Example hash: a1b2c3d4 e5f6g7h8 i9j0k1l2
#               ^^^^^^^^ ^^^^^^^^ ^^^^^^^^
#               Model    Config   Data
```

### 2. Pending Changes System

Changes are **pending** (not immediately dumped) until:
- Training resumes → automatic dump
- Manual dump requested → `force_dump=True`

This prevents unnecessary disk writes when making configuration changes.

### 3. Component Change Detection

```python
from weightslab.components import ExperimentHashGenerator

gen = ExperimentHashGenerator()

# Compare two hashes
hash1 = "a1b2c3d4e5f6g7h8i9j0k1l2"  # Old hash
hash2 = "a1b2c3d4f0f0f0f0i9j0k1l2"  # New hash

changed = gen.compare_hashes(hash1, hash2)
# Returns: {'config'}  # Only config changed
```

## Usage Examples

### Automatic System (Recommended)

```python
from weightslab.components import checkpoint_on_step

# Training loop
for step in range(1000):
    # Your training code
    loss = train_step()

    # Option 1: Regular checkpoint (every N steps)
    checkpoint_on_step(step)

    # Option 2: Force dump pending changes
    if step % 100 == 0:
        checkpoint_on_step(step, force_dump=True)
```

### Tracking Changes

```python
from weightslab.components import (
    checkpoint_on_model_change,
    checkpoint_on_config_change,
    checkpoint_on_data_change
)

# Model architecture changed (e.g., added layers)
checkpoint_on_model_change()  # Marked as pending

# Hyperparameters changed
new_config = {'lr': 0.001, 'batch_size': 64}
checkpoint_on_config_change(new_config)  # Marked as pending

# Data changed (discarded samples, added tags)
data_state = {
    'uids': ['sample_001', 'sample_002', 'sample_003'],
    'discarded': {'sample_002'},  # sample_002 is discarded
    'tags': {'sample_001': ['validated', 'clean']}
}
checkpoint_on_data_change(data_state)  # Marked as pending

# Resume training → all pending changes dumped automatically
for step in range(100):
    loss = train_step()
    checkpoint_on_step(step)  # First call dumps pending changes
```

### Immediate Dumps

```python
# Force immediate dump instead of marking as pending
checkpoint_on_model_change(dump_immediately=True)
checkpoint_on_config_change(config, dump_immediately=True)
checkpoint_on_data_change(data_state, dump_immediately=True)
```

### Manual Checkpoint Manager

```python
from weightslab.components import CheckpointManagerV2

manager = CheckpointManagerV2(root_log_dir='experiments')

# Update hash with all components
model = my_model
config = {'lr': 0.001, 'batch_size': 32}
data_state = {
    'uids': [...],
    'discarded': {...},
    'tags': {...}
}

# Mark as pending (default)
exp_hash, is_new, changed = manager.update_experiment_hash(
    model=model,
    config=config,
    data_state=data_state,
    dump_immediately=False  # Pending
)

print(f"New hash: {exp_hash}")
print(f"Changed components: {changed}")  # e.g., {'model', 'data'}

# Check pending changes
has_pending, pending_comps = manager.has_pending_changes()
if has_pending:
    print(f"Pending: {pending_comps}")

# Dump pending changes manually
manager.dump_pending_changes(force=True)

# Save model checkpoint (with optional force dump)
manager.save_model_checkpoint(
    model=model,
    step=100,
    force_dump_pending=True  # Dump pending before saving checkpoint
)
```

## Directory Structure

```
root_log_dir/
  checkpoints/
    {24-byte-hash}/          # e.g., a1b2c3d4e5f6g7h8i9j0k1l2
      model/
        {hash}_step_000100.pt
        {hash}_step_000200.pt
        {hash}_architecture.pkl
      hp/
        {hash}_config.yaml
      data/
        {hash}_data_state.yaml
```

## Hash Breakdown

When you have a hash like `a1b2c3d4e5f6g7h8i9j0k1l2`:

```python
model_hash = "a1b2c3d4"     # Characters 0-7
config_hash = "e5f6g7h8"    # Characters 8-15
data_hash = "i9j0k1l2"      # Characters 16-23
```

To identify changes between two hashes:

```python
# Hash 1: a1b2c3d4e5f6g7h8i9j0k1l2
# Hash 2: a1b2c3d4e5f6g7h8ffffffff

# Compare segments:
# Model:  a1b2c3d4 == a1b2c3d4  ✓ No change
# Config: e5f6g7h8 == e5f6g7h8  ✓ No change
# Data:   i9j0k1l2 != ffffffff  ✗ CHANGED

# Conclusion: Only data changed
```

## Data State Format

```python
data_state = {
    'uids': [
        'sample_001',
        'sample_002',
        'sample_003'
    ],
    'discarded': {
        'sample_002'  # Set of discarded UIDs
    },
    'tags': {
        'sample_001': ['validated', 'augmented'],
        'sample_003': ['test']
    }
}
```

The data hash is computed from:
- UIDs (sorted)
- Discard status per UID
- Tags per UID (sorted)

Example internal representation:
```
sample_001:d0:taugmented,validated
sample_002:d1:t
sample_003:d0:ttest
```

## API Reference

### ExperimentHashGenerator

```python
from weightslab.components import ExperimentHashGenerator

gen = ExperimentHashGenerator()

# Generate hash
hash_str = gen.generate_hash(model, config, data_state)
# Returns: "a1b2c3d4e5f6g7h8i9j0k1l2" (24 chars)

# Check if changed
has_changed, changed_comps = gen.has_changed(model, config, data_state)
# Returns: (True, {'model', 'data'})

# Get component hashes
hashes = gen.get_component_hashes()
# Returns: {
#     'model': 'a1b2c3d4',
#     'config': 'e5f6g7h8',
#     'data': 'i9j0k1l2',
#     'combined': 'a1b2c3d4e5f6g7h8i9j0k1l2'
# }

# Compare two hashes
changed = gen.compare_hashes(hash1, hash2)
# Returns: {'config'}
```

### CheckpointManagerV2

```python
from weightslab.components import CheckpointManagerV2

manager = CheckpointManagerV2('experiments')

# Update hash
exp_hash, is_new, changed = manager.update_experiment_hash(
    model=model,
    config=config,
    data_state=data_state,
    dump_immediately=False  # Pending by default
)

# Check pending
has_pending, components = manager.has_pending_changes()

# Dump pending
manager.dump_pending_changes(force=True)

# Save checkpoint
manager.save_model_checkpoint(
    model=model,
    step=100,
    force_dump_pending=True
)

# Save individual components
manager.save_config(config)
manager.save_data_state(data_state)
manager.save_model_architecture(model)
```

### AutomaticCheckpointSystem

```python
from weightslab.components import (
    get_checkpoint_system,
    checkpoint_on_step,
    checkpoint_on_model_change,
    checkpoint_on_config_change,
    checkpoint_on_data_change
)

# Get global system
system = get_checkpoint_system(
    root_log_dir='experiments',
    checkpoint_frequency=100
)

# Training loop
for step in range(1000):
    loss = train_step()
    checkpoint_on_step(step, force_dump=False)

# Change notifications
checkpoint_on_model_change(model, dump_immediately=False)
checkpoint_on_config_change(config, dump_immediately=False)
checkpoint_on_data_change(data_state, dump_immediately=False)
```

## Migration from V2

If you were using the old system without data hashing:

```python
# Old V2 (16-byte hash, no data)
exp_hash = gen.generate_hash(model, config)
# Returns: "a1b2c3d4e5f6g7h8"

# New V3 (24-byte hash, with data)
exp_hash = gen.generate_hash(model, config, data_state)
# Returns: "a1b2c3d4e5f6g7h800000000"  # Default 00000000 if data_state=None
```

Old hashes are **not compatible**. You'll need to regenerate hashes with the new system.

## Best Practices

1. **Use pending changes** for configuration experiments
   ```python
   checkpoint_on_config_change(config)  # Pending
   # Make more config changes...
   checkpoint_on_model_change(model)  # Still pending
   # Resume training → dumps all pending
   checkpoint_on_step(0)  # Dumps on first step
   ```

2. **Force dumps** for critical changes
   ```python
   checkpoint_on_model_change(model, dump_immediately=True)
   ```

3. **Regular checkpoints** with force_dump on milestones
   ```python
   for step in range(1000):
       train_step()

       # Regular checkpoint
       if step % checkpoint_freq == 0:
           checkpoint_on_step(step)

       # Milestone with forced dump
       if step % 500 == 0:
           checkpoint_on_step(step, force_dump=True)
   ```

4. **Compare hashes** to understand changes
   ```python
   from weightslab.components import ExperimentHashGenerator

   gen = ExperimentHashGenerator()
   changes = gen.compare_hashes(old_hash, new_hash)

   if 'model' in changes:
       print("Model architecture changed")
   if 'config' in changes:
       print("Hyperparameters changed")
   if 'data' in changes:
       print("Data state changed (discard/tags)")
   ```

## Troubleshooting

### Hashes keep changing unexpectedly

Check which component is changing:
```python
has_changed, changed_comps = manager.has_pending_changes()
print(f"Pending: {changed_comps}")
```

### Pending changes not dumping

Ensure training resumes:
```python
checkpoint_on_step(step)  # First call after changes dumps pending
```

Or force dump:
```python
checkpoint_on_step(step, force_dump=True)
```

### Data state not tracking

Ensure dataset has required methods:
```python
# Required for data state tracking
dataset.get_sample_uids()  # or .sample_ids or .uids
dataset.is_discarded(uid)
dataset.get_tags(uid)  # or .tags dict
```

## Summary

- **24-byte hash** = 8 bytes model + 8 bytes config + 8 bytes data
- **Pending changes** = marked but not dumped until training resumes
- **Force dump** = immediately write pending changes to disk
- **Component comparison** = identify what changed between hashes
- **Automatic system** = transparent, ledger-integrated checkpointing
