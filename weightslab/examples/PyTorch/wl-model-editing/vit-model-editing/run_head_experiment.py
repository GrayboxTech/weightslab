#!/usr/bin/env python3
"""End-to-end ViT feature extractor + editable MLP head experiment."""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

import torch
from common import (
    build_vit_b_16,
    environment_info,
    extract_embeddings,
    make_pattern_images,
    optimizer_parameter_ids,
    resolve_device,
    seed_everything,
    write_report,
)
from torch import nn

import weightslab as wl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ViT-B/16 feature pipeline, edit its MLP head, and resume training."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vit_model_editing"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--add-neurons", type=int, default=2)
    parser.add_argument("--before-steps", type=int, default=4)
    parser.add_argument("--after-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def train_steps(
    model,
    optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    capture_layer=None,
    capture_neuron: int = 0,
) -> tuple[list[float], float | None]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    captured_gradient_norm = None

    for step in range(steps):
        start = (step * batch_size) % len(features)
        indices = torch.arange(start, start + batch_size) % len(features)
        with wl.guard_training_context:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indices])
            loss = criterion(logits, labels[indices])
            loss.backward()
            if capture_layer is not None and step == 0:
                gradient = capture_layer.weight.grad
                if gradient is None:
                    raise AssertionError("Expected a weight gradient, but found None.")
                captured_gradient_norm = float(gradient[capture_neuron].norm().item())
            optimizer.step()
        losses.append(float(loss.detach()))
    return losses, captured_gradient_norm


def active_neuron_for_batch(
    model,
    optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    layer,
) -> tuple[int, float]:
    """Run one ordinary step and return a neuron proven active on that batch."""
    model.train()
    with wl.guard_training_context:
        optimizer.zero_grad(set_to_none=True)
        logits = model(features[:batch_size])
        loss = nn.functional.cross_entropy(logits, labels[:batch_size])
        loss.backward()
        gradient = layer.weight.grad
        if gradient is None:
            raise AssertionError("Expected hidden-layer gradients, but found None.")
        row_norms = gradient.norm(dim=1)
        neuron = int(row_norms.argmax().item())
        norm = float(row_norms[neuron].item())
        if norm <= 1e-12:
            raise AssertionError("No active hidden neuron was found for the test batch.")
        optimizer.step()
    return neuron, norm


@torch.no_grad()
def evaluate(model, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    model.eval()
    with wl.guard_testing_context:
        logits = model(features)
        loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return {"loss": float(loss), "accuracy": float(accuracy)}


def find_edit_target(graph: dict, hidden_size: int) -> tuple[int, int]:
    linear_layers = [layer for layer in graph["layers"] if layer["type"] == "Linear"]
    source = next(
        (layer for layer in linear_layers if layer["output_neurons"] == hidden_size),
        None,
    )
    if source is None:
        raise AssertionError("Could not identify the editable hidden Linear layer.")

    downstream = next(
        (
            layer
            for layer in linear_layers
            if layer["id"] != source["id"] and layer["input_neurons"] == hidden_size
        ),
        None,
    )
    if downstream is None:
        raise AssertionError("Could not identify the downstream classifier Linear layer.")
    return source["id"], downstream["id"]


def main() -> int:
    args = parse_args()
    print("[1/7] Resolving environment and synthetic data", flush=True)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    report_path = args.output_dir / "vit_head_editing_report.json"
    report = {
        "status": "running",
        "scope": "ViT-B/16 frozen feature extractor with a WeightsLab-editable MLP head",
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "environment": environment_info(device),
        "checks": {},
    }

    try:
        train_images, train_labels = make_pattern_images(
            samples=args.train_samples,
            image_size=args.image_size,
            num_classes=args.num_classes,
            seed=args.seed,
            normalize=args.pretrained,
        )
        eval_images, eval_labels = make_pattern_images(
            samples=args.eval_samples,
            image_size=args.image_size,
            num_classes=args.num_classes,
            seed=args.seed + 1,
            normalize=args.pretrained,
        )

        backbone, block_count = build_vit_b_16(
            image_size=args.image_size,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
        )
        embedding_size = backbone.hidden_dim
        print(f"[2/7] Extracting embeddings with ViT-B/16 ({block_count} blocks)", flush=True)
        backbone.heads = nn.Identity()
        backbone.requires_grad_(False)
        backbone.to(device)
        train_features = extract_embeddings(
            backbone,
            train_images,
            batch_size=args.batch_size,
            device=device,
        )
        eval_features = extract_embeddings(
            backbone,
            eval_images,
            batch_size=args.batch_size,
            device=device,
        )
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("[3/7] Wrapping the editable MLP head", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        wl.clear_all()
        wl.watch_or_edit(
            {
                "root_log_dir": str(args.output_dir / "weightslab_state"),
                "skip_checkpoint_load": True,
                "experiment_dump_to_train_steps_ratio": 0,
            },
            flag="hyperparameters",
        )
        raw_head = nn.Sequential(
            nn.Linear(embedding_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.num_classes),
        )
        model = wl.watch_or_edit(
            raw_head,
            flag="model",
            device="cpu",
            dummy_input=train_features[:1],
            compute_dependencies=True,
            forced_model_wrapping=True,
            skip_previous_auto_load=True,
        )
        optimizer = wl.watch_or_edit(
            torch.optim.SGD(model.parameters(), lr=args.learning_rate),
            flag="optimizer",
        )
        # This is a finite, headless tensor experiment, so no Studio/serve flow
        # exists to resume the training guard for us.
        wl.start_training()

        graph_before = model.get_model_graph(include_neurons=True)
        print("[4/7] Training the baseline head", flush=True)
        source_id, downstream_id = find_edit_target(graph_before, args.hidden_size)
        before_metrics = evaluate(model, eval_features, eval_labels)
        before_losses, _ = train_steps(
            model,
            optimizer,
            train_features,
            train_labels,
            steps=args.before_steps,
            batch_size=args.batch_size,
        )

        model.add_neurons(source_id, count=args.add_neurons)
        print(f"[5/7] Added {args.add_neurons} hidden neurons; validating propagation", flush=True)
        graph_after_add = model.get_model_graph(include_neurons=True)
        source_after = model.get_layer_info(source_id)
        downstream_after = model.get_layer_info(downstream_id)
        expected_hidden = args.hidden_size + args.add_neurons
        if source_after["output_neurons"] != expected_hidden:
            raise AssertionError("The hidden layer did not gain the requested neurons.")
        if downstream_after["input_neurons"] != expected_hidden:
            raise AssertionError("The downstream classifier input was not propagated.")

        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_ids_after_add = optimizer_parameter_ids(optimizer)
        if model_parameter_ids != optimizer_ids_after_add:
            raise AssertionError("The optimizer does not reference every post-edit parameter.")

        after_add_losses, _ = train_steps(
            model,
            optimizer,
            train_features,
            train_labels,
            steps=args.after_steps,
            batch_size=args.batch_size,
        )

        source_layer = model.get_layer_by_id(source_id)
        freeze_neuron, pre_freeze_gradient_norm = active_neuron_for_batch(
            model,
            optimizer,
            train_features,
            train_labels,
            batch_size=args.batch_size,
            layer=source_layer,
        )
        model.freeze_neurons(source_id, [freeze_neuron])
        print("[6/7] Validating frozen and unfrozen gradients", flush=True)
        if not model.get_layer_info(source_id)["neurons"][freeze_neuron]["frozen"]:
            raise AssertionError(f"Neuron {freeze_neuron} was not reported frozen.")
        source_layer = model.get_layer_by_id(source_id)
        _, frozen_gradient_norm = train_steps(
            model,
            optimizer,
            train_features,
            train_labels,
            steps=1,
            batch_size=args.batch_size,
            capture_layer=source_layer,
            capture_neuron=freeze_neuron,
        )
        if frozen_gradient_norm is None or frozen_gradient_norm > 1e-12:
            raise AssertionError(
                f"Frozen neuron gradient should be zero, got {frozen_gradient_norm}."
            )

        model.unfreeze_neurons(source_id, [freeze_neuron])
        if model.get_layer_info(source_id)["neurons"][freeze_neuron]["frozen"]:
            raise AssertionError(f"Neuron {freeze_neuron} was not reported unfrozen.")
        source_layer = model.get_layer_by_id(source_id)
        _, unfrozen_gradient_norm = train_steps(
            model,
            optimizer,
            train_features,
            train_labels,
            steps=1,
            batch_size=args.batch_size,
            capture_layer=source_layer,
            capture_neuron=freeze_neuron,
        )
        if unfrozen_gradient_norm is None or unfrozen_gradient_norm <= 1e-12:
            raise AssertionError(
                f"Unfrozen neuron gradient should be non-zero, got {unfrozen_gradient_norm}."
            )

        final_metrics = evaluate(model, eval_features, eval_labels)
        print("[7/7] Writing experiment report", flush=True)
        all_losses = before_losses + after_add_losses
        if not all(math.isfinite(loss) for loss in all_losses):
            raise AssertionError("Training produced a non-finite loss.")

        report.update(
            {
                "status": "passed",
                "architecture": {
                    "name": "torchvision vit_b_16",
                    "encoder_blocks": block_count,
                    "embedding_size": embedding_size,
                    "editable_scope": "MLP classification head",
                },
                "graph_before": graph_before,
                "graph_after_add": graph_after_add,
                "metrics": {
                    "before_training": before_metrics,
                    "after_training_and_edits": final_metrics,
                    "training_losses": all_losses,
                },
                "checks": {
                    "add_propagated": True,
                    "optimizer_rebuilt": True,
                    "tested_neuron": freeze_neuron,
                    "pre_freeze_gradient_norm": pre_freeze_gradient_norm,
                    "frozen_gradient_norm": frozen_gradient_norm,
                    "unfrozen_gradient_norm": unfrozen_gradient_norm,
                    "forward_after_edit": True,
                    "training_resumed": True,
                },
            }
        )
        print(f"PASS: report written to {report_path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - experiment reports must capture any failure
        report.update(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
            }
        )
        print(f"FAIL: {type(exc).__name__}: {exc}")
        print(f"Report written to {report_path}")
        return 1
    finally:
        write_report(report_path, report)
        wl.clear_all()


if __name__ == "__main__":
    raise SystemExit(main())
