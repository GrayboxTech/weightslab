# Weightslab Checkpoint System V2 - Summary

## What Was Created

I've implemented a complete restructuring of the weightslab checkpoint system with the following new components:

### 1. **New Modules**

#### `weightslab/components/experiment_hash.py`
- `ExperimentHashGenerator`: Generates deterministic hashes from:
  - Model architecture (layers, structure, parameters)
  - Hyperparameters configuration
  - Active data sample UIDs (non-discarded)
- Hash changes automatically when any component changes
- Provides `has_changed()` to detect what changed

#### `weightslab/components/checkpoint_manager_v2.py`
- `CheckpointManagerV2`: New structured checkpoint management
- Organized directory structure by experiment hash
- Separate handling for:
  - Model weights (saved every N steps)
  - Model architecture (saved once per hash)
  - Hyperparameter configs (saved/updated on changes)
  - Data backups (saved on hash changes)

### 2. **Directory Structure**

```
root_log_dir/
├── data/                    # Global data files
├── logs/                    # Global training logs
└── checkpoints/
    ├── {exp_hash_A}/
    │   ├── model/          # Model checkpoints & architecture
    │   ├── hp/             # Config YAML files
    │   └── data/           # Data sample backups (h5)
    └── {exp_hash_B}/
        └── ...
```

### 3. **Documentation**

- **`CHECKPOINT_V2_GUIDE.md`**: Complete migration guide with:
  - Usage examples
  - API reference
  - Migration instructions from old system
  - Best practices
  - Troubleshooting

- **`examples/checkpoint_v2_example.py`**: Runnable examples demonstrating:
  - Basic usage
  - Training loop integration
  - Architecture changes
  - Config changes
  - Data discarding
  - Loading checkpoints
  - Experiment info retrieval

### 4. **Updated Files**

- **`weightslab/components/__init__.py`**: Exports new classes
  - `CheckpointManagerV2`
  - `ExperimentHashGenerator`

## Key Features

### 1. Hash-Based Experiment Tracking
- **Automatic hash generation** from model, config, and data
- **New directory created** when any component changes
- **Complete provenance** tracking

### 2. Structured Checkpointing
- **Model weights**: Every N steps (fast, weights only)
- **Architecture**: Once per hash (full model structure)
- **Config**: Saved/updated on changes (YAML format)
- **Data**: Backed up on hash changes (h5 files)

### 3. Flexible Integration
- Works with **existing ledger system**
- Can pass objects directly or use ledger names
- **Metadata support** for additional tracking
- **Step-based** checkpoint naming

### 4. Robust Loading
- Load latest checkpoint by hash
- Load specific step checkpoint
- Load config separately
- List all experiments

## Usage Example

```python
from weightslab.components import CheckpointManagerV2

# Initialize
ckpt_manager = CheckpointManagerV2(root_log_dir="./experiments")

# Set up experiment
model = MyModel()
config = {'learning_rate': 0.001, 'batch_size': 32}
sample_uids = get_active_sample_uids()

# Generate hash (creates directories)
exp_hash, is_new = ckpt_manager.update_experiment_hash(
    model=model,
    config=config,
    active_data_uids=sample_uids
)

# Training loop
for step in range(num_steps):
    train_step(model)

    # Save checkpoint every 100 steps
    if step % 100 == 0:
        ckpt_manager.save_model_checkpoint(
            model=model,
            step=step,
            save_optimizer=True
        )

# Load latest checkpoint
checkpoint = ckpt_manager.load_latest_checkpoint(model=model)
```

## Integration Points

### Where to Integrate in Weightslab:

1. **Trainer Initialization**
   - Replace `CheckpointManager` with `CheckpointManagerV2`
   - Call `update_experiment_hash()` at start

2. **Training Loop**
   - Add periodic `save_model_checkpoint()` calls
   - Update config with `save_config()` when params change

3. **Data Discarding**
   - Call `update_experiment_hash()` after discarding samples
   - Optionally backup data with `save_data_backup()`

4. **Model Changes**
   - Call `update_experiment_hash()` when architecture changes
   - New directory automatically created

## Benefits Over Old System

| Feature | Old System | New System |
|---------|-----------|------------|
| **Organization** | Flat, numbered | Hash-based hierarchy |
| **Provenance** | Limited | Complete (model+config+data) |
| **Config Tracking** | No | Full YAML configs |
| **Data Tracking** | No | Hash includes data UIDs |
| **Architecture** | Optional per ckpt | Once per hash |
| **Finding Experiments** | Hard | Easy (list hashes) |
| **Resume Training** | By ID | By hash or latest |

## Next Steps

### To Start Using:

1. **Import the new system**:
   ```python
   from weightslab.components import CheckpointManagerV2
   ```

2. **Replace old checkpoint manager** in your trainer/main files

3. **Add hash updates** at appropriate points:
   - Start of training
   - After model changes
   - After data discarding
   - After config changes

4. **Test with examples**:
   ```bash
   python weightslab/examples/checkpoint_v2_example.py
   ```

### Migration Strategy:

1. **Keep old system** for backward compatibility (it's still there)
2. **Use new system** for new experiments
3. **Gradual migration** of existing code
4. **Old system can be deprecated** later

### Future Enhancements:

- Automatic checkpoint cleanup (remove old checkpoints)
- Compression for old checkpoints
- Remote storage support (S3, GCS)
- Web UI for browsing experiments
- Checkpoint comparison tools
- Automatic resume functionality

## Files Created

```
weightslab/
├── components/
│   ├── __init__.py                    (updated)
│   ├── experiment_hash.py             (new)
│   └── checkpoint_manager_v2.py       (new)
├── examples/
│   └── checkpoint_v2_example.py       (new)
└── CHECKPOINT_V2_GUIDE.md             (new)
```

## Summary

This checkpoint system provides:
- ✅ **Organized structure** with hash-based directories
- ✅ **Complete provenance** tracking (model, config, data)
- ✅ **Flexible checkpointing** (weights vs architecture)
- ✅ **Easy experiment management** (list, load, compare)
- ✅ **Backward compatible** (old system still works)
- ✅ **Well documented** (guide + examples)
- ✅ **Production ready** (error handling, logging)

The system is ready to use! Check the guide and examples for detailed usage instructions.
