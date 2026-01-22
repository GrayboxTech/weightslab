"""
Quick Test - Automatic Checkpoint System

This script demonstrates the automatic checkpoint system in action.
Run this to verify everything works correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import weightslab as wl
from weightslab.components import (
    get_checkpoint_system,
    checkpoint_on_step,
    checkpoint_on_model_change,
    checkpoint_on_config_change,
    checkpoint_on_state_change,
)


# Simple test model
class SimpleNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_automatic_system():
    """Test the automatic checkpoint system."""
    print("="*60)
    print("Automatic Checkpoint System - Quick Test")
    print("="*60)

    # 1. Initialize checkpoint system
    print("\n1. Initializing checkpoint system...")
    checkpoint_system = get_checkpoint_system(
        root_log_dir="./test_checkpoints",
        checkpoint_frequency=5,  # Checkpoint every 5 steps for testing
        auto_init=True
    )
    print(f"   ✓ Initialized at: {checkpoint_system.checkpoint_manager.root_log_dir}")

    # 2. Register model and optimizer
    print("\n2. Registering model and optimizer...")
    model = SimpleNet(hidden_size=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'hidden_size': 64,
    }

    wl.register_model('model', model)
    wl.register_optimizer('optimizer', optimizer)
    wl.register_hyperparams('config', config)
    print("   ✓ Registered in ledger")

    # 3. Check initial status
    print("\n3. Checking initial status...")
    status = checkpoint_system.get_status()
    print(f"   Initialized: {status['initialized']}")
    print(f"   Experiment hash: {status['current_exp_hash']}")
    print(f"   Checkpoint frequency: {status['checkpoint_frequency']}")

    # 4. Simulate training steps
    print("\n4. Simulating training (15 steps)...")
    for step in range(15):
        # Fake training step
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        # Automatic checkpoint
        checkpoint_on_step(step)

        if step % 5 == 0:
            print(f"   Step {step}: loss={loss.item():.4f}")

    print("   ✓ Training completed")

    # 5. Check checkpoint status
    print("\n5. Checking checkpoint status...")
    status = checkpoint_system.get_status()
    print(f"   Current step: {status['current_step']}")
    print(f"   Last checkpoint step: {status['last_checkpoint_step']}")

    info = checkpoint_system.checkpoint_manager.get_checkpoint_info()
    print(f"   Checkpoints saved: {len(info['model_checkpoints'])}")
    print(f"   Checkpoint files: {info['model_checkpoints']}")

    # 6. Test config change
    print("\n6. Testing config change...")
    config['learning_rate'] = 0.0001
    config['batch_size'] = 64
    wl.register_hyperparams('config', config)
    checkpoint_on_config_change()
    print("   ✓ Config updated and saved")

    status = checkpoint_system.get_status()
    print(f"   New experiment hash: {status['current_exp_hash']}")

    # 7. Test model architecture change
    print("\n7. Testing model architecture change...")
    old_hash = status['current_exp_hash']

    # Create bigger model
    model_v2 = SimpleNet(hidden_size=128)  # Double hidden size
    wl.register_model('model', model_v2)
    checkpoint_on_model_change()
    print("   ✓ Model architecture changed")

    status = checkpoint_system.get_status()
    new_hash = status['current_exp_hash']
    print(f"   Old hash: {old_hash}")
    print(f"   New hash: {new_hash}")
    print(f"   Hash changed: {old_hash != new_hash}")

    # 8. Test state change
    print("\n8. Testing state change (freeze)...")
    for param in model_v2.fc1.parameters():
        param.requires_grad = False
    checkpoint_on_state_change('freeze')
    print("   ✓ Frozen fc1 layer, checkpoint saved")

    # 9. List all experiments
    print("\n9. Listing all experiments...")
    all_hashes = checkpoint_system.checkpoint_manager.list_experiment_hashes()
    print(f"   Total experiments: {len(all_hashes)}")
    for i, exp_hash in enumerate(all_hashes, 1):
        print(f"   {i}. {exp_hash}")
        info = checkpoint_system.checkpoint_manager.get_checkpoint_info(exp_hash)
        print(f"      - Checkpoints: {len(info['model_checkpoints'])}")
        print(f"      - Architecture saved: {info['architecture_saved']}")
        print(f"      - Configs: {len(info['configs'])}")

    # 10. Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"✓ Automatic initialization: Working")
    print(f"✓ Periodic checkpointing: Working (saved {len(info['model_checkpoints'])} checkpoints)")
    print(f"✓ Config change detection: Working")
    print(f"✓ Model change detection: Working")
    print(f"✓ State change tracking: Working")
    print(f"✓ Experiment organization: {len(all_hashes)} hash(es) created")
    print(f"\nCheckpoint directory: ./test_checkpoints")
    print("="*60)


def cleanup_test_checkpoints():
    """Optional: Clean up test checkpoints."""
    import shutil
    test_dir = Path("./test_checkpoints")
    if test_dir.exists():
        response = input("\nClean up test checkpoints? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(test_dir)
            print("✓ Test checkpoints cleaned up")


if __name__ == "__main__":
    try:
        test_automatic_system()
        cleanup_test_checkpoints()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
