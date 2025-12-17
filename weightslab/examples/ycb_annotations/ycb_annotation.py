import os
import logging
import tempfile

import yaml
import weightslab as wl

from torchvision import datasets, transforms


# Setup logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "ycb_training_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Set Defaults Parameters Values
    parameters.setdefault("experiment_name", "ycb_cnn")
    parameters.setdefault("number_of_workers", 4)

    # Get experiment name
    exp_name = parameters["experiment_name"]

    # Logging directory
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = os.path.join(tmp_dir, "logs")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    tqdm_display = parameters.get('tqdm_display', True)
    verbose = parameters.get('verbose', True)
    log_dir = parameters["root_log_dir"]

    # --- 4) Hyperparameters ---
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    # ------------------------ DATA ------------------------
    # ------------------------------------------------------
    data_root = parameters.get("data", {}).get(
        "data_dir",
        os.path.join(parameters["root_log_dir"], "data", "ycb_datasets"),
    )
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    image_size = parameters.get("image_size", 128)

    IM_MEAN = (0.6312528, 0.4949005, 0.3298562)
    IM_STD  = (0.0721354, 0.0712461, 0.0598827)
    common_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IM_MEAN, IM_STD),
        ]
    )

    # Load subsample of datasets for quick testing
    _train_dataset = datasets.ImageFolder(root=train_dir, transform=common_transform)
    _test_dataset = datasets.ImageFolder(root=val_dir, transform=common_transform)

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        name="train_loader",
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=train_cfg.get("train_shuffle", True),
        is_training=True,
        compute_hash=False
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        name="test_loader",
        batch_size=test_cfg.get("batch_size", 16),
        shuffle=test_cfg.get("test_shuffle", False),
        compute_hash=False
    )

    # --- 7) Start WeightsLab services (UI + gRPC) ---
    wl.serve(
        serving_ui=False,
        
        serving_cli=True,

        serving_grpc=True,
        n_workers_grpc=parameters.get("number_of_workers"),
    )

    print("\n" + "=" * 60)
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)
    
    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()
