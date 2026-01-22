"""
Automatic Checkpoint System - Integration Guide

This guide shows how the automatic checkpoint system works and how to
integrate it with existing weightslab code.

The system is designed to be completely transparent - once enabled, it
automatically handles all checkpointing with no manual intervention required.
"""

# =============================================================================
# 1. AUTOMATIC INITIALIZATION
# =============================================================================

# The checkpoint system auto-initializes when the first model or dataloader
# is registered in the ledger.

import weightslab as wl
from weightslab.components import get_checkpoint_system

# Just get the checkpoint system - it initializes automatically
checkpoint_system = get_checkpoint_system(
    root_log_dir="./experiments",
    checkpoint_frequency=100  # Save every 100 steps
)

# That's it! The system is now active and monitoring the ledger.


# =============================================================================
# 2. IN YOUR TRAINING LOOP
# =============================================================================

# Option A: Manual step tracking (simple)
from weightslab.components import checkpoint_on_step

for step in range(num_steps):
    # Your training code
    loss = train_step(model, data)

    # Just call this after each step - it handles everything
    checkpoint_on_step(step)


# Option B: Automatic step tracking (even simpler)
# The system can auto-increment steps if you don't pass a step number
checkpoint_on_step()  # Auto-increments internally


# Option C: Hook into existing trainer
# If you have a trainer class, add the checkpoint call:

class MyTrainer:
    def train_step(self):
        # ... training code ...
        loss = self.compute_loss()

        # Automatically checkpoint
        checkpoint_on_step(self.current_step)

        return loss


# =============================================================================
# 3. MODEL ARCHITECTURE CHANGES
# =============================================================================

# When you modify model architecture (add/prune layers), just notify the system:
from weightslab.components import checkpoint_on_model_change

# Original model
model = MyModel(hidden_size=128)
wl.register_model('model', model)

# ... train for a while ...

# Change architecture
model = MyModel(hidden_size=256)  # Different architecture
wl.register_model('model', model)

# Notify checkpoint system (auto-detects from ledger if model not provided)
checkpoint_on_model_change()

# A new checkpoint directory with a new hash is automatically created!


# =============================================================================
# 4. HYPERPARAMETER CHANGES
# =============================================================================

# When hyperparameters change, notify the system:
from weightslab.components import checkpoint_on_config_change

# Initial config
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'adam'
}
wl.register_hyperparams('config', config)

# ... train for a while ...

# Change config
config['learning_rate'] = 0.0001  # Decrease learning rate
config['batch_size'] = 64          # Increase batch size
wl.register_hyperparams('config', config)

# Notify checkpoint system
checkpoint_on_config_change()

# Config is automatically saved and potentially new hash created!


# =============================================================================
# 5. MODEL STATE CHANGES (FREEZE/RESET)
# =============================================================================

# When you freeze or reset model components, notify the system:
from weightslab.components import checkpoint_on_state_change

# Freeze some layers
for param in model.encoder.parameters():
    param.requires_grad = False

checkpoint_on_state_change('freeze')

# Later, reset some layers
model.decoder.reset_parameters()

checkpoint_on_state_change('reset')

# Checkpoints are saved with metadata about the state change!


# =============================================================================
# 6. INTEGRATION WITH DATALOADER
# =============================================================================

# The system automatically initializes when you register a dataloader:

train_loader = MyDataLoader(...)
wl.register_dataloader('train', train_loader)

# That's it! The checkpoint system now knows about your data.

# Note: Data changes (discarded samples, tags) do NOT trigger new checkpoints.
# Only model and hyperparameter changes trigger new checkpoint directories.


# =============================================================================
# 7. COMPLETE TRAINING EXAMPLE
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import weightslab as wl
from weightslab.components import checkpoint_on_step, checkpoint_on_config_change

# Setup
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
config = {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10}

# Register in ledger
wl.register_model('model', model)
wl.register_optimizer('optimizer', optimizer)
wl.register_hyperparams('config', config)

# Checkpoint system auto-initializes on registration!

# Training loop
global_step = 0
for epoch in range(config['epochs']):
    for batch in train_loader:
        # Training step
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

        # Automatic checkpoint (every 100 steps by default)
        checkpoint_on_step(global_step)
        global_step += 1

    # Adjust learning rate
    if epoch == 5:
        config['learning_rate'] = 0.0001
        wl.register_hyperparams('config', config)
        checkpoint_on_config_change()

# That's it! All checkpoints saved automatically!


# =============================================================================
# 8. MONITORING CHECKPOINT STATUS
# =============================================================================

# Check checkpoint system status
from weightslab.components import get_checkpoint_system

system = get_checkpoint_system()
status = system.get_status()

print(f"Initialized: {status['initialized']}")
print(f"Current step: {status['current_step']}")
print(f"Last checkpoint: {status['last_checkpoint_step']}")
print(f"Experiment hash: {status['current_exp_hash']}")
print(f"Root dir: {status['root_log_dir']}")


# =============================================================================
# 9. LOADING CHECKPOINTS
# =============================================================================

# Loading is also automatic:
from weightslab.components import get_checkpoint_system

system = get_checkpoint_system()

# Load latest checkpoint for current experiment
checkpoint_data = system.checkpoint_manager.load_latest_checkpoint(
    model=model,
    load_optimizer=True
)

if checkpoint_data:
    print(f"Resumed from step {checkpoint_data['step']}")


# =============================================================================
# 10. ADVANCED: CUSTOM CHECKPOINT FREQUENCY
# =============================================================================

# Change checkpoint frequency at runtime:
system = get_checkpoint_system()
system.checkpoint_frequency = 50  # Save every 50 steps instead of 100


# =============================================================================
# 11. INTEGRATION WITH MODEL INTERFACE
# =============================================================================

# If you're using weightslab's ModelInterface, the checkpoint system
# can hook into its callbacks:

class MyModelWithAutoCheckpoint(wl.ModelInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Checkpoint system is already active from ledger registration
        # No additional setup needed!

    def train_step(self, batch):
        loss = super().train_step(batch)

        # Auto-checkpoint
        checkpoint_on_step()

        return loss


# =============================================================================
# 12. WHAT GETS CHECKPOINTED AUTOMATICALLY
# =============================================================================

"""
The automatic checkpoint system handles:

EVERY N STEPS (default 100):
    ✓ Model weights (state_dict)
    ✓ Optimizer state
    ✓ Training step number
    ✓ Metadata (loss, metrics, etc.)

ON MODEL ARCHITECTURE CHANGE:
    ✓ New experiment hash generated
    ✓ New checkpoint directory created
    ✓ Full model architecture saved
    ✓ Current weights saved

ON HYPERPARAMETER CHANGE:
    ✓ New experiment hash generated (if significant change)
    ✓ Updated config saved to YAML
    ✓ Checkpoint saved with new config

ON MODEL STATE CHANGE (freeze/reset):
    ✓ Checkpoint saved with state change metadata
    ✓ Current state preserved

NOT TRIGGERED BY:
    ✗ Data discarding (samples removed)
    ✗ Tag changes
    ✗ Data augmentation changes

These are considered data transformations, not experiment changes,
so they don't trigger new checkpoint directories.
"""


# =============================================================================
# 13. TIPS AND BEST PRACTICES
# =============================================================================

"""
1. Register model/optimizer/config EARLY in your script
   - This initializes the checkpoint system automatically

2. Call checkpoint_on_step() at the END of each training step
   - After optimizer.step() and any metric computation

3. Call checkpoint_on_config_change() AFTER updating hyperparams
   - Not before, so the new values are captured

4. Use checkpoint_on_model_change() AFTER architecture modifications
   - After changing layers, not before

5. Use checkpoint_on_state_change() for freeze/reset operations
   - Helps track when model behavior changes

6. Don't worry about checkpoint frequency
   - The system handles it automatically
   - Change checkpoint_frequency if needed

7. Check status periodically during development
   - Use get_status() to verify checkpoints are being saved

8. The system is thread-safe
   - Multiple workers can safely trigger checkpoints
   - Internal locking prevents race conditions
"""
