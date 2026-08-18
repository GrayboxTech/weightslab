"""wl-video-generation — train a small video (and image) generator under
full WeightsLab supervision, with sound.

Three conditioning modes, selected by `mode` in config.yaml:

    text        prompt                -> clip      (text-to-video)
    text_video  prompt + source clip  -> clip      (instructed video editing)
    video       source clip           -> clip      (a learned "anime" filter)

Set `frames: 1` and the exact same script becomes an IMAGE generator — a clip
of length one is just a picture, and every path below is length-agnostic.

What WeightsLab buys you here
-----------------------------
Every clip is a ROW keyed by its uid. FlowMatchingLoss is a reduction="none"
per-sample criterion wrapped with wl.watch_or_edit(flag="loss"), so passing
batch_ids=uids makes the wrapper log a per-clip loss trajectory under
"train/fm_loss" and auto-enrol it in the loss-shape classifier.

That turns "watch the samples, not the loss" into an instrument:
  * a clip whose loss stays flat and high  -> bad caption / off-style -> tag and
    remove it live from the studio.
  * a clip whose loss collapses to ~0      -> the model is memorizing it ->
    down-weight it.

In the studio, video samples show a poster frame in grid and list mode; opening
one and pressing "Play" streams the muxed clip (H.264 + AAC) through the
GetMedia RPC, where you can scrub frames and hear the soundtrack.
"""
import itertools
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
import tqdm
import yaml

import weightslab as wl
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import VideoGenerationDataset, AUDIO_SAMPLE_RATE
from utils.model import FlowMatchingLoss, VideoFlowUNet, sample_clip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wl-video-generation")


def to_model_space(clip_uint8):
    """[B, T, H, W, C] uint8  ->  [B, C, T, H, W] float in [-1, 1]."""
    x = clip_uint8.float() / 127.5 - 1.0
    return x.permute(0, 4, 1, 2, 3).contiguous()


def to_clip_space(x):
    """[B, C, T, H, W] float in [-1, 1]  ->  [B, T, H, W, C] uint8."""
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 4, 1).contiguous()


def unpack_batch(inputs, mode, device):
    """Split the tracked loader's batch into (target, source) model tensors."""
    target = to_model_space(inputs[0]).to(device)
    source = None
    if mode in ("text_video", "video"):
        source = to_model_space(inputs[1]).to(device)
    return target, source


def flow_matching_step(model, criterion, target, source, captions, uids, device):
    """One rectified-flow training step; returns the per-sample loss [B]."""
    noise = torch.randn_like(target)
    # Uniform t in (0, 1); x_t interpolates from the clip (t=0) to noise (t=1).
    t = torch.rand(target.shape[0], device=device)
    t_b = t[:, None, None, None, None]
    noisy = (1.0 - t_b) * target + t_b * noise
    velocity = noise - target  # the field the model must predict

    pred = model(noisy, t * 1000.0, captions=captions, source=source)
    # target= is a KEYWORD on purpose (see FlowMatchingLoss docstring).
    return criterion(pred, target=velocity, batch_ids=uids)


@torch.no_grad()
def render_previews(model, dataset, cfg, device, step):
    """Sample a few clips and write them next to the logs as MP4s.

    Best-effort: a preview failure must never kill a training run.
    """
    try:
        from weightslab.data import video_utils as vu

        model.eval()
        # cfg values are live ValueProxy objects (see the note at the call to
        # watch_or_edit) — coerce anything that feeds a shape or a slice.
        mode = str(cfg["mode"])
        frames, res = int(cfg["frames"]), int(cfg["resolution"])
        prompts = list(cfg.get("sample_prompts") or [dataset.items[0]["caption"]])
        prompts = prompts[:int(cfg.get("num_preview_samples", 2))]

        source = None
        if mode in ("text_video", "video"):
            stack = np.stack([dataset.items[i % len(dataset.items)]["frames"]
                              for i in range(len(prompts))])
            source = to_model_space(torch.from_numpy(stack)).to(device)

        generated = sample_clip(
            model, (len(prompts), 3, frames, res, res), device,
            steps=int(cfg.get("sample_steps", 25)),
            captions=prompts if mode in ("text", "text_video") else None,
            source=source)

        out_dir = os.path.join(str(cfg["root_log_dir"]), "samples", f"step_{step}")
        os.makedirs(out_dir, exist_ok=True)
        clips = to_clip_space(generated).cpu().numpy()
        for i, clip in enumerate(clips):
            audio, _ = dataset.get_audio(i)
            data, mime, _ = vu.encode_clip(
                clip, float(cfg["fps"]), audio=audio, sample_rate=AUDIO_SAMPLE_RATE)
            if not data:
                continue
            ext = "mp4" if mime == "video/mp4" else "gif"
            with open(os.path.join(out_dir, f"sample_{i}.{ext}"), "wb") as handle:
                handle.write(data)
        logger.info("[step %s] previews -> %s", step, out_dir)
    except Exception as exc:
        logger.warning("preview render skipped: %r", exc)
    finally:
        model.train()


@torch.no_grad()
def attach_generated_media(model, inputs, uids, captions, mode, cfg, device,
                           dataset):
    """Generate a clip per sample and attach it to the studio as metadata.

    This is what makes the run reviewable: ``pred_video`` sits next to
    ``target_video`` as its own column in list mode, one thumbnail per row, and
    clicking either opens the player. Without it the generations would only ever
    exist as files on disk.

    Sampling is far more expensive than a training step, so this runs on the
    eval cadence, not every step.
    """
    try:
        model.eval()
        target, source = unpack_batch(inputs, mode, device)
        frames_n, res = int(cfg["frames"]), int(cfg["resolution"])
        generated = sample_clip(
            model, (len(uids), 3, frames_n, res, res), device,
            steps=int(cfg.get("sample_steps", 25)),
            captions=captions if mode in ("text", "text_video") else None,
            source=source)

        gen_clips = to_clip_space(generated).cpu().numpy()
        tgt_clips = to_clip_space(target).cpu().numpy()
        waveforms = [dataset.get_audio(i)[0] for i in range(len(uids))]
        fps_v = float(cfg["fps"])

        wl.save_media("pred_video", batch_ids=uids, media=gen_clips,
                      kind="video", fps=fps_v, audio=waveforms,
                      sample_rate=AUDIO_SAMPLE_RATE, dataset=dataset)
        wl.save_media("target_video", batch_ids=uids, media=tgt_clips,
                      kind="video", fps=fps_v, audio=waveforms,
                      sample_rate=AUDIO_SAMPLE_RATE, dataset=dataset)
        # For the conditioned modes the input clip is itself worth seeing, so
        # input -> output can be compared on one row.
        if source is not None:
            wl.save_media("source_video", batch_ids=uids,
                          media=to_clip_space(source).cpu().numpy(),
                          kind="video", fps=fps_v, dataset=dataset)
    except Exception as exc:
        logger.warning("attaching generated media skipped: %r", exc)
    finally:
        model.train()


def evaluate(loader, model, criterion, mode, device, cfg=None, attach=False,
             dataset=None):
    """Full pass over the eval loader, logging per-sample reconstruction error.

    When ``attach`` is set, also generates and attaches a clip per sample so the
    studio can show prediction vs target side by side. ``dataset`` must be the
    RAW dataset, not the loader: the tracked loader wraps it in a
    DataSampleTrackingWrapper that does not proxy ``get_audio``.
    """
    losses, count = 0.0, 0
    for inputs, ids, labels, metadata in loader:
        with guard_testing_context:
            uids = list(metadata["uid"])
            captions = list(metadata["caption"])
            target, source = unpack_batch(inputs, mode, device)
            per_sample = flow_matching_step(
                model, criterion, target, source, captions, uids, device)
            losses += float(per_sample.mean())
            count += 1
        if attach and cfg is not None and dataset is not None:
            attach_generated_media(model, inputs, uids, captions, mode, cfg,
                                   device, dataset)
    return losses / max(1, count)


if __name__ == "__main__":
    start_time = time.time()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "config.yaml"), "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    if cfg.get("device", "auto") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg["device"])
    cfg["root_log_dir"] = os.path.abspath(os.path.join(here, cfg["root_log_dir"]))
    os.makedirs(cfg["root_log_dir"], exist_ok=True)

    mode = str(cfg["mode"])
    if mode not in ("text", "text_video", "video"):
        raise SystemExit(
            f"mode must be one of text | text_video | video, got {mode!r}")

    # ---- 1. Hyperparameters (editable live from the studio) ----
    # NOTE: this replaces the scalar values in `cfg` with live ValueProxy
    # objects so the studio can edit them mid-run. Anything that feeds a tensor
    # shape, a numpy call or a range() must therefore be coerced with
    # int()/float() at the point of use — a proxy is not a number.
    wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)

    frames = int(cfg["frames"])
    resolution = int(cfg["resolution"])
    fps = float(cfg["fps"])

    # ---- 2. Data ----
    data_cfg = cfg.get("data", {})
    cache_dir = cfg.get("hf_cache_dir") or None
    train_dataset = VideoGenerationDataset(
        source=str(data_cfg.get("source", "synthetic")),
        mode=mode,
        num_frames=frames,
        resolution=resolution,
        fps=fps,
        max_samples=int(data_cfg.get("train_loader", {}).get("max_samples", 64)),
        cache_dir=cache_dir,
        split="train",
    )
    eval_dataset = VideoGenerationDataset(
        source=str(data_cfg.get("source", "synthetic")),
        mode=mode,
        num_frames=frames,
        resolution=resolution,
        fps=fps,
        max_samples=int(data_cfg.get("test_loader", {}).get("max_samples", 16)),
        cache_dir=cache_dir,
        split="eval",
    )
    logger.info("Loaded %d train / %d eval clips (%s, mode=%s)",
                len(train_dataset), len(eval_dataset),
                data_cfg.get("source", "synthetic"), mode)

    train_loader = wl.watch_or_edit(
        train_dataset, flag="data", loader_name="train_loader",
        batch_size=int(data_cfg.get("train_loader", {}).get("batch_size", 2)),
        shuffle=True, is_training=True, compute_hash=False,
        preload_labels=False, preload_metadata=True,
        enable_h5_persistence=cfg.get("enable_h5_persistence", True),
    )
    eval_loader = wl.watch_or_edit(
        eval_dataset, flag="data", loader_name="test_loader",
        batch_size=int(data_cfg.get("test_loader", {}).get("batch_size", 2)),
        shuffle=False, is_training=False, compute_hash=False,
        preload_labels=False, preload_metadata=True,
        enable_h5_persistence=cfg.get("enable_h5_persistence", True),
    )

    # ---- 3. Model ----
    _model = VideoFlowUNet(
        mode=mode,
        base_channels=int(cfg.get("base_channels", 48)),
        channel_mults=tuple(int(m) for m in cfg.get("channel_mults", [1, 2, 4])),
    ).to(device)
    logger.info("Model parameters: %s",
                f"{sum(p.numel() for p in _model.parameters()):,}")
    # compute_dependencies=False: the neuron dependency graph is built for 2D
    # conv classifiers and is not meaningful for a 3D generative U-Net.
    model = wl.watch_or_edit(
        _model, flag="model", device=device, compute_dependencies=False)

    # ---- 4. Loss (per-sample, watched) ----
    train_criterion = wl.watch_or_edit(
        FlowMatchingLoss(), flag="loss", signal_name="train/fm_loss", log=True)
    eval_criterion = wl.watch_or_edit(
        FlowMatchingLoss(), flag="loss", signal_name="test/fm_loss", log=True)

    # ---- 5. Optimizer ----
    _optimizer = optim.AdamW(
        _model.parameters(), lr=float(cfg.get("optimizer", {}).get("lr", 2e-4)),
        weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 0.0)))
    optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")

    # ---- 6. Services ----
    wl.serve(serving_grpc=cfg.get("serving_grpc", True),
             serving_cli=cfg.get("serving_cli", False))

    # ================= Training loop =================
    wl.start_training(timeout=3)

    steps = cfg.get("training_steps_to_do")
    steps = int(steps) if steps is not None else None
    eval_ratio = int(cfg.get("eval_full_to_train_steps_ratio", 100))
    sample_every = int(cfg.get("sample_every", 250))
    grad_clip = float(cfg.get("max_grad_norm", 1.0))

    train_range = tqdm.tqdm(
        range(steps) if steps is not None else itertools.count(),
        desc=f"video-gen[{mode}]", ncols=120)

    _model.train()
    eval_loss = None
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step

        with guard_training_context:
            try:
                inputs, ids, labels, metadata = next(train_loader)
            except StopIteration:
                continue

            uids = list(metadata["uid"])
            captions = list(metadata["caption"])
            target, source = unpack_batch(inputs, mode, device)

            optimizer.zero_grad()
            per_sample = flow_matching_step(
                model, train_criterion, target, source, captions, uids, device)
            loss = per_sample.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), grad_clip)
            optimizer.step()

        if age > 0 and age % eval_ratio == 0:
            eval_loss = evaluate(
                eval_loader, model, eval_criterion, mode, device,
                cfg=cfg, attach=bool(cfg.get("attach_generated_media", True)),
                dataset=eval_dataset)
            _model.train()

        if age > 0 and sample_every and age % sample_every == 0:
            render_previews(model, train_dataset, cfg, device, age)

        postfix = {"loss": f"{float(loss):.4f}"}
        if eval_loss is not None:
            postfix["eval"] = f"{eval_loss:.4f}"
        train_range.set_postfix(postfix)

    logger.info("Training finished in %.1fs", time.time() - start_time)
    render_previews(model, train_dataset, cfg, device, "final")
    wl.keep_serving()
