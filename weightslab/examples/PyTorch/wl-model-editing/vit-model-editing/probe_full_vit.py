#!/usr/bin/env python3
"""Compatibility gate for editing the complete torchvision ViT-B/16 model."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch
from common import (
    build_vit_b_16,
    environment_info,
    make_pattern_images,
    resolve_device,
    seed_everything,
    write_report,
)
from torch import nn

import weightslab as wl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether WeightsLab can train and edit a complete ViT-B/16."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vit_model_editing"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Return success while still recording an unsupported result.",
    )
    return parser.parse_args()


def one_raw_training_step(model, images, labels, optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    report_path = args.output_dir / "full_vit_compatibility_report.json"
    report = {
        "status": "running",
        "scope": "Complete torchvision ViT-B/16 model",
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "environment": environment_info(device),
        "phases": {},
    }

    try:
        images, labels = make_pattern_images(
            samples=max(args.batch_size, args.num_classes),
            image_size=args.image_size,
            num_classes=args.num_classes,
            seed=args.seed,
            normalize=args.pretrained,
        )
        images = images[: args.batch_size].to(device)
        labels = labels[: args.batch_size].to(device)
        raw_model, block_count = build_vit_b_16(
            image_size=args.image_size,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
        )
        raw_model.to(device)
        baseline_optimizer = torch.optim.SGD(raw_model.parameters(), lr=1e-3)
        baseline_loss = one_raw_training_step(
            raw_model, images, labels, baseline_optimizer
        )
        report["phases"]["baseline_training"] = {
            "status": "passed",
            "loss": baseline_loss,
            "encoder_blocks": block_count,
        }

        wl.clear_all()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        wl.watch_or_edit(
            {
                "root_log_dir": str(args.output_dir / "full_vit_weightslab_state"),
                "skip_checkpoint_load": True,
                "experiment_dump_to_train_steps_ratio": 0,
            },
            flag="hyperparameters",
        )
        model = wl.watch_or_edit(
            raw_model,
            flag="model",
            device=str(device),
            dummy_input=images[:1],
            compute_dependencies=True,
            forced_model_wrapping=True,
            skip_previous_auto_load=True,
        )
        graph = model.get_model_graph()
        report["phases"]["weightslab_graph"] = {
            "status": "passed",
            "layers": len(graph["layers"]),
            "dependencies": len(graph["dependencies"]),
        }

        head = next(layer for layer in graph["layers"] if layer["name"] == "heads.head")
        optimizer = wl.watch_or_edit(
            torch.optim.SGD(model.parameters(), lr=1e-3),
            flag="optimizer",
        )
        wl.start_training()
        model.freeze_neurons(head["id"], [0])
        with wl.guard_training_context:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        report["phases"]["weightslab_edit_and_resume"] = {
            "status": "passed",
            "loss": float(loss.detach()),
        }
        report["status"] = "passed"
        print(f"PASS: complete ViT editing is supported; report written to {report_path}")
        return 0
    except Exception as exc:  # noqa: BLE001 - compatibility probes must report any failure
        report.update(
            {
                "status": "unsupported",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": traceback.format_exc(),
            }
        )
        print(f"UNSUPPORTED: {type(exc).__name__}: {exc}")
        print(f"Report written to {report_path}")
        return 0 if args.allow_unsupported else 1
    finally:
        write_report(report_path, report)
        wl.clear_all()


if __name__ == "__main__":
    raise SystemExit(main())
