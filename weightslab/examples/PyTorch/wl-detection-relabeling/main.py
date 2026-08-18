import os
import tempfile

import yaml

import weightslab as wl

from utils.data import PennFudanDetectionDataset, det_collate


# =============================================================================
# Main -- data exploration / inspection / tagging / export-for-relabeling.
#
# No model, optimizer, loss, or training loop: `watch_or_edit(..., flag="data")`
# already preloads every sample's ground-truth boxes and metadata into the
# dataframe at registration time (`preload_labels`/`preload_metadata` default
# to True), so the grid is fully browsable the moment this script starts
# serving -- there is nothing to train towards, only data to look at.
# =============================================================================
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    parameters.setdefault("experiment_name", "pennfudan_detection_relabeling")
    parameters.setdefault("num_classes", 1)  # Penn-Fudan: single class (person)
    parameters.setdefault("image_size", 256)
    parameters.setdefault("seed_review_tag_count", 10)

    exp_name = parameters["experiment_name"]

    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    num_classes = int(parameters["num_classes"])
    image_size = int(parameters["image_size"])

    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    # --- Data (Penn-Fudan pedestrians, downloaded on first run) ---
    default_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    data_root = parameters.get("data_root", default_data_root)
    data_cfg = parameters.get("data", {}).get("data_loader", {})

    dataset = PennFudanDetectionDataset(
        root=data_root,
        split="train",  # only the train split is loaded here, for simplicity
        num_classes=num_classes,
        image_size=image_size,
        max_samples=data_cfg.get("max_samples", None),
    )

    loader = wl.watch_or_edit(
        dataset,
        flag="data",
        loader_name="data_loader",
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=False,
        is_training=False,
        compute_hash=False,
        array_autoload_arrays=False,
        array_return_proxies=True,
        array_use_cache=True,
        collate_fn=det_collate,
    )

    # Seed a few samples with a "ToReview" tag purely so the tag-filtered
    # export workflow has something to demo on first launch. Real usage: tag
    # whatever you actually flag while browsing (grid right-click, or just
    # ask the in-app chat agent), then export that subset -- see README.md.
    # Sample ids are filename-derived (e.g. "FudanPed00001"), not 0..N indices.
    seed_count = min(int(parameters.get("seed_review_tag_count", 10)), len(dataset))
    seed_sample_ids = [
        os.path.splitext(os.path.basename(p))[0] for p in dataset.images[:seed_count]
    ]
    if seed_sample_ids:
        wl.tag_samples(seed_sample_ids, "ToReview")

    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", True),
    )

    print("=" * 60)
    print(" PENN-FUDAN DETECTION -- DATA EXPLORATION / TAGGING / RELABELING")
    print(f" {len(dataset)} samples registered ({seed_count} seeded with tag:ToReview)")
    print(f" Data root: {data_root}")
    print(" Open Weights Studio, browse/tag samples, then export a tagged")
    print(" subset for relabeling, e.g.:")
    print("   weightslab export -f cvat --tag ToReview")
    print("=" * 60 + "\n")

    # No wl.start_training()/guard_training_context/guard_testing_context: this
    # usecase never runs a training step, so there is nothing to gate. See
    # README.md ("Why no training loop") for the reasoning.
    wl.keep_serving()
