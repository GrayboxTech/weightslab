"""
Example: Using CheckpointManagerV2

This script demonstrates how to use the new checkpoint system
with a simple training example.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import new checkpoint system
from weightslab.components import CheckpointManagerV2


# Simple example model
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def example_basic_usage():
    """Example 1: Basic checkpoint usage"""
    print("\n=== Example 1: Basic Usage ===")

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManagerV2(root_log_dir="./example_experiments")

    # Create model and config
    model = SimpleModel()
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam',
        'epochs': 10
    }

    # Simulate data sample UIDs
    sample_uids = [f"sample_{i:04d}" for i in range(1000)]

    # Generate initial experiment hash
    exp_hash, is_new = ckpt_manager.update_experiment_hash(
        model=model,
        config=config,
        active_data_uids=sample_uids
    )

    print(f"Experiment hash: {exp_hash}")
    print(f"Is new experiment: {is_new}")

    # Save a checkpoint
    checkpoint_path = ckpt_manager.save_model_checkpoint(
        model=model,
        step=0,
        metadata={'loss': 1.5, 'epoch': 0}
    )
    print(f"Saved checkpoint: {checkpoint_path.name if checkpoint_path else 'Failed'}")

    return ckpt_manager, model, config, sample_uids


def example_training_loop():
    """Example 2: Simulated training loop with periodic checkpointing"""
    print("\n=== Example 2: Training Loop ===")

    # Setup
    ckpt_manager = CheckpointManagerV2(root_log_dir="./example_experiments")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam',
    }

    sample_uids = [f"sample_{i:04d}" for i in range(1000)]

    # Initialize hash
    exp_hash, _ = ckpt_manager.update_experiment_hash(
        model=model,
        config=config,
        active_data_uids=sample_uids
    )

    print(f"Training with experiment hash: {exp_hash}")

    # Simulate training
    num_steps = 500
    checkpoint_frequency = 100

    for step in range(num_steps):
        # Simulate training step
        fake_loss = 2.0 - (step / num_steps) * 1.5  # Decreasing loss

        # Save checkpoint every N steps
        if step % checkpoint_frequency == 0:
            ckpt_manager.save_model_checkpoint(
                model=model,
                step=step,
                save_optimizer=True,
                metadata={'loss': fake_loss}
            )
            print(f"Step {step:04d}: Loss={fake_loss:.4f} (checkpoint saved)")

    # Get checkpoint info
    info = ckpt_manager.get_checkpoint_info()
    print(f"\nTotal checkpoints saved: {len(info['model_checkpoints'])}")
    print(f"Checkpoint files: {info['model_checkpoints']}")


def example_architecture_change():
    """Example 3: Changing model architecture (new hash)"""
    print("\n=== Example 3: Architecture Change ===")

    ckpt_manager = CheckpointManagerV2(root_log_dir="./example_experiments")

    config = {'learning_rate': 0.001, 'batch_size': 32}
    sample_uids = [f"sample_{i:04d}" for i in range(1000)]

    # Original model
    model_v1 = SimpleModel(hidden_size=128)
    hash_v1, _ = ckpt_manager.update_experiment_hash(
        model=model_v1,
        config=config,
        active_data_uids=sample_uids
    )
    print(f"Model V1 hash: {hash_v1}")

    # Save checkpoint
    ckpt_manager.save_model_checkpoint(model=model_v1, step=100)

    # Change architecture (different hidden size)
    model_v2 = SimpleModel(hidden_size=256)  # Bigger model
    hash_v2, is_new = ckpt_manager.update_experiment_hash(
        model=model_v2,
        config=config,
        active_data_uids=sample_uids
    )

    print(f"Model V2 hash: {hash_v2}")
    print(f"Is new experiment: {is_new}")
    print(f"Hashes match: {hash_v1 == hash_v2}")

    # Save checkpoint for new architecture
    ckpt_manager.save_model_checkpoint(model=model_v2, step=100)

    # List all experiments
    all_hashes = ckpt_manager.list_experiment_hashes()
    print(f"\nAll experiment hashes: {all_hashes}")


def example_config_change():
    """Example 4: Changing hyperparameters (new hash)"""
    print("\n=== Example 4: Config Change ===")

    ckpt_manager = CheckpointManagerV2(root_log_dir="./example_experiments")
    model = SimpleModel()
    sample_uids = [f"sample_{i:04d}" for i in range(1000)]

    # Original config
    config_v1 = {'learning_rate': 0.001, 'batch_size': 32}
    hash_v1, _ = ckpt_manager.update_experiment_hash(
        model=model,
        config=config_v1,
        active_data_uids=sample_uids
    )
    print(f"Config V1 hash: {hash_v1}")
    print(f"Config V1: {config_v1}")

    # Change config
    config_v2 = {'learning_rate': 0.01, 'batch_size': 64}  # Different params
    hash_v2, is_new = ckpt_manager.update_experiment_hash(
        model=model,
        config=config_v2,
        active_data_uids=sample_uids
    )

    print(f"Config V2 hash: {hash_v2}")
    print(f"Config V2: {config_v2}")
    print(f"Is new experiment: {is_new}")
    print(f"Hashes match: {hash_v1 == hash_v2}")


def example_data_discard():
    """Example 5: Discarding data samples (new hash)"""
    print("\n=== Example 5: Data Discarding ===")

    ckpt_manager = CheckpointManagerV2(root_log_dir="./example_experiments")
    model = SimpleModel()
    config = {'learning_rate': 0.001, 'batch_size': 32}

    # Original data
    all_samples = [f"sample_{i:04d}" for i in range(1000)]
    hash_v1, _ = ckpt_manager.update_experiment_hash(
        model=model,
        config=config,
        active_data_uids=all_samples
    )
    print(f"Full dataset hash: {hash_v1}")
    print(f"Sample count: {len(all_samples)}")

    # Discard some samples
    discarded_samples = all_samples[:100]  # Discard first 100
    active_samples = all_samples[100:]      # Keep the rest

    hash_v2, is_new = ckpt_manager.update_experiment_hash(
        model=model,
        config=config,
        active_data_uids=active_samples
    )

    print(f"Filtered dataset hash: {hash_v2}")
    print(f"Sample count: {len(active_samples)}")
    print(f"Is new experiment: {is_new}")
    print(f"Hashes match: {hash_v1 == hash_v2}")


def example_checkpoint_loading():
    """Example 6: Loading checkpoints"""
    print("\n=== Example 6: Checkpoint Loading ===")

    ckpt_manager = CheckpointManagerV2(root_log_dir="./example_experiments")
    model = SimpleModel()
    config = {'learning_rate': 0.001, 'batch_size': 32}
    sample_uids = [f"sample_{i:04d}" for i in range(1000)]

    # Create and save checkpoint
    exp_hash, _ = ckpt_manager.update_experiment_hash(
        model=model,
        config=config,
        active_data_uids=sample_uids
    )

    # Save multiple checkpoints
    for step in [100, 200, 300]:
        ckpt_manager.save_model_checkpoint(
            model=model,
            step=step,
            metadata={'step': step}
        )

    # Load latest checkpoint
    checkpoint_data = ckpt_manager.load_latest_checkpoint(model=model)

    if checkpoint_data:
        print(f"Loaded checkpoint from step: {checkpoint_data['step']}")
        print(f"Timestamp: {checkpoint_data['timestamp']}")
        print(f"Metadata: {checkpoint_data.get('metadata', {})}")

    # Load config
    loaded_config = ckpt_manager.load_config()
    print(f"Loaded config: {loaded_config}")


def example_experiment_info():
    """Example 7: Getting experiment information"""
    print("\n=== Example 7: Experiment Info ===")

    ckpt_manager = CheckpointManagerV2(root_log_dir="./example_experiments")

    # List all experiments
    all_hashes = ckpt_manager.list_experiment_hashes()
    print(f"Found {len(all_hashes)} experiments:")

    for i, exp_hash in enumerate(all_hashes, 1):
        print(f"\n{i}. Experiment: {exp_hash}")

        info = ckpt_manager.get_checkpoint_info(exp_hash)
        print(f"   Model checkpoints: {len(info['model_checkpoints'])}")
        print(f"   Architecture saved: {info['architecture_saved']}")
        print(f"   Configs: {len(info['configs'])}")
        print(f"   Data backups: {len(info['data_backups'])}")

        if info['model_checkpoints']:
            print(f"   Latest checkpoint: {info['model_checkpoints'][-1]}")


def main():
    """Run all examples"""
    print("=" * 60)
    print("CheckpointManagerV2 Examples")
    print("=" * 60)

    # Run examples
    example_basic_usage()
    example_training_loop()
    example_architecture_change()
    example_config_change()
    example_data_discard()
    example_checkpoint_loading()
    example_experiment_info()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check './example_experiments' directory for results")
    print("=" * 60)


if __name__ == "__main__":
    main()
