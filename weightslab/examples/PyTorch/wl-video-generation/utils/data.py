"""Datasets for the wl-video-generation example.

Three conditioning modes share one dataset class, because they differ only in
which conditioning tensors each sample carries:

  * ``text``       prompt -> clip                (pure text-to-video)
  * ``text_video`` prompt + source clip -> clip  (instructed video editing)
  * ``video``      source clip -> clip           (a learned "anime" style filter)

Sources
-------
``disney``  ``sayakpaul/video-dataset-disney-organized`` on the HF Hub — 24 MB,
            69 clips, each with a long descriptive prompt joined through
            ``metadata.csv``. Downloaded file-by-file with ``huggingface_hub``
            and decoded with OpenCV; deliberately NOT via ``datasets``, whose
            v4 ``Video`` feature pulls in ``torchcodec`` (an FFmpeg- and
            torch-version-matched build that routinely fails on Windows).
            These clips are silent, so an audio track is synthesized.
``kinetics`` ``nateraw/kinetics-mini`` — 136 MB, 100 clips that DO carry real
            AAC soundtracks. Captions are templated from the class label.
``synthetic`` procedural moving shapes with phase-locked audio, no download.
            This is the default: it always runs, and its audio is genuinely
            correlated with the motion (a bounce makes a click, velocity sets
            the pitch), so the audio path is actually learnable — unlike real
            YouTube-sourced ambience.

Every sample returns ``(inputs, idx, label, metadata)``, the 4-tuple form the
tracked loader expects, where ``inputs`` is a dict of tensors and ``label`` is
the caption string. The clip tensor is ``[T, H, W, C]`` uint8 so that
WeightsLab's video router previews it directly; the model normalizes.
"""
import csv
import logging
import subprocess
import shutil
import wave

import numpy as np
import torch

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DISNEY_REPO = "sayakpaul/video-dataset-disney-organized"
KINETICS_REPO = "nateraw/kinetics-mini"
AUDIO_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Decoding helpers (OpenCV for frames, ffmpeg CLI for audio)
# ---------------------------------------------------------------------------
def decode_frames(path, num_frames, resolution):
    """Decode ``num_frames`` evenly spaced RGB frames from a video file."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Could not read any frames from {path}")

    wanted = np.linspace(0, max(0, total - 1), num_frames).astype(int)
    frames, next_wanted, index = [], 0, 0
    while next_wanted < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        # Emit the same decoded frame as many times as it was requested, so a
        # clip shorter than num_frames still yields a full-length tensor.
        while next_wanted < len(wanted) and wanted[next_wanted] == index:
            resized = cv2.resize(frame, (resolution, resolution),
                                 interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            next_wanted += 1
        index += 1
    capture.release()

    if not frames:
        raise RuntimeError(f"Could not decode {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames).astype(np.uint8)


def decode_audio(path, duration_seconds):
    """Extract mono 16 kHz audio via the ffmpeg CLI. Returns None when silent."""
    binary = shutil.which("ffmpeg")
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
             "-t", f"{duration_seconds:.3f}", "-f", "wav", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not proc.stdout:
            return None
        import io
        with wave.open(io.BytesIO(proc.stdout), "rb") as handle:
            raw = handle.readframes(handle.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples if samples.size and np.abs(samples).max() > 1e-4 else None
    except Exception as exc:
        logger.debug("Audio extraction failed for %s: %r", path, exc)
        return None


def synthesize_audio(frames, duration_seconds):
    """Derive a soundtrack from the clip's own motion.

    Frame-to-frame difference drives amplitude and mean brightness drives
    pitch, so the track is phase-locked to the visuals. For the synthetic
    source this is the ground truth the model can actually learn; for the
    silent Disney clips it at least keeps the audio path exercised.
    """
    length = int(AUDIO_SAMPLE_RATE * duration_seconds)
    if length <= 0 or frames.shape[0] < 2:
        return np.zeros(max(1, length), dtype=np.float32)

    gray = frames.astype(np.float32).mean(axis=3).mean(axis=(1, 2)) / 255.0
    motion = np.abs(np.diff(gray, prepend=gray[:1]))
    peak = motion.max()
    motion = motion / peak if peak > 1e-6 else motion

    t = np.arange(length) / AUDIO_SAMPLE_RATE
    # Per-sample envelope / pitch interpolated from the per-frame series.
    frame_pos = np.linspace(0, frames.shape[0] - 1, length)
    envelope = np.interp(frame_pos, np.arange(frames.shape[0]), motion)
    pitch = 220.0 + 440.0 * np.interp(frame_pos, np.arange(frames.shape[0]), gray)
    wave_out = 0.6 * envelope * np.sin(2 * np.pi * pitch * t)
    return wave_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Source loaders -> list of {"frames", "caption", "audio"}
# ---------------------------------------------------------------------------
def _load_disney(cache_dir, num_frames, resolution, max_samples):
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(DISNEY_REPO, repo_type="dataset")
    meta_name = next((f for f in files if f.endswith("metadata.csv")), None)
    if meta_name is None:
        raise RuntimeError(f"{DISNEY_REPO} has no metadata.csv")

    meta_path = hf_hub_download(
        DISNEY_REPO, meta_name, repo_type="dataset", cache_dir=cache_dir)
    prefix = meta_name.rsplit("/", 1)[0] + "/" if "/" in meta_name else ""

    rows = []
    with open(meta_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name, caption = row.get("file_name"), row.get("prompt")
            if name and caption:
                rows.append((prefix + name, caption.strip()))
    if max_samples:
        rows = rows[:max_samples]

    items = []
    for name, caption in rows:
        try:
            path = hf_hub_download(
                DISNEY_REPO, name, repo_type="dataset", cache_dir=cache_dir)
            frames = decode_frames(path, num_frames, resolution)
        except Exception as exc:
            logger.warning("Skipping %s: %r", name, exc)
            continue
        items.append({"frames": frames, "caption": caption, "audio": None})
    return items


def _load_kinetics(cache_dir, num_frames, resolution, max_samples, fps):
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(KINETICS_REPO, repo_type="dataset")
             if f.endswith(".mp4")]
    files.sort()
    if max_samples:
        files = files[:max_samples]

    duration = num_frames / float(fps)
    items = []
    for name in files:
        # Path layout is train/<class>/<clip>.mp4 — the class is the caption.
        parts = name.split("/")
        label = parts[-2].replace("_", " ") if len(parts) >= 2 else "a video"
        try:
            path = hf_hub_download(
                KINETICS_REPO, name, repo_type="dataset", cache_dir=cache_dir)
            frames = decode_frames(path, num_frames, resolution)
            audio = decode_audio(path, duration)
        except Exception as exc:
            logger.warning("Skipping %s: %r", name, exc)
            continue
        items.append({
            "frames": frames,
            "caption": f"a video of a person {label}",
            "audio": audio,
        })
    return items


def _load_synthetic(count, num_frames, resolution, seed=0):
    """Procedural clips: a coloured shape bouncing on a coloured ground."""
    rng = np.random.default_rng(seed)
    palette = [
        ("red", (220, 60, 60)), ("green", (60, 200, 90)),
        ("blue", (70, 110, 230)), ("yellow", (235, 205, 60)),
    ]
    motions = ["bouncing", "drifting left", "drifting right", "spinning in place"]

    items = []
    for i in range(count):
        colour_name, colour = palette[i % len(palette)]
        motion = motions[(i // len(palette)) % len(motions)]
        frames = np.zeros((num_frames, resolution, resolution, 3), dtype=np.uint8)
        # Background gradient keeps the frames from being trivially constant.
        frames[:, :, :, 2] = np.linspace(30, 90, resolution).astype(np.uint8)[None, :, None]

        radius = max(3, resolution // 8)
        phase = rng.uniform(0, 2 * np.pi)
        for t in range(num_frames):
            u = t / max(1, num_frames - 1)
            if motion == "bouncing":
                cx = resolution // 2
                cy = int(radius + (resolution - 2 * radius) * abs(np.sin(np.pi * 2 * u + phase)))
            elif motion == "drifting left":
                cx = int(resolution - radius - (resolution - 2 * radius) * u)
                cy = resolution // 2
            elif motion == "drifting right":
                cx = int(radius + (resolution - 2 * radius) * u)
                cy = resolution // 2
            else:
                cx = int(resolution / 2 + (resolution / 4) * np.cos(2 * np.pi * u + phase))
                cy = int(resolution / 2 + (resolution / 4) * np.sin(2 * np.pi * u + phase))

            ys, xs = np.ogrid[:resolution, :resolution]
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
            for c in range(3):
                frames[t, :, :, c] = np.where(mask, colour[c], frames[t, :, :, c])

        items.append({
            "frames": frames,
            "caption": f"a {colour_name} ball {motion}",
            "audio": None,  # synthesized below, from this clip's own motion
        })
    return items


# ---------------------------------------------------------------------------
# Style filter (the "video-only" mode target)
# ---------------------------------------------------------------------------
def anime_stylize(frames):
    """A cheap, deterministic 'anime' look: posterize + edge darkening.

    This is the supervision target for ``mode: video`` — the model learns to
    reproduce the filter, which makes the mode meaningful without needing a
    second, style-paired dataset.
    """
    quantized = (frames.astype(np.int16) // 48) * 48 + 24
    quantized = np.clip(quantized, 0, 255)

    gray = frames.astype(np.float32).mean(axis=3)
    # Cross-shaped gradient magnitude; strong edges become ink lines.
    dy = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1, :]))
    dx = np.abs(np.diff(gray, axis=2, prepend=gray[:, :, :1]))
    edges = (dx + dy) > 28.0
    quantized[edges] = 20
    # Slightly boost saturation so the palette reads as illustrated.
    mean = quantized.mean(axis=3, keepdims=True)
    quantized = np.clip(mean + (quantized - mean) * 1.35, 0, 255)
    return quantized.astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VideoGenerationDataset(Dataset):
    """Clips + captions + audio, in one of three conditioning modes.

    WeightsLab reads three attributes off this object:
      * ``task_type = "video_generation"`` routes previews to the poster-frame
        renderer and the modal to the GetMedia player.
      * ``fps`` / ``num_frames`` let the grid badge a duration without decoding.
      * ``get_audio(index)`` supplies the soundtrack the player muxes in.
    """

    task_type = "video_generation"

    def __init__(self, source="synthetic", mode="text", num_frames=16,
                 resolution=64, fps=8, max_samples=64, cache_dir=None,
                 split="train"):
        self.mode = mode
        self.num_frames = num_frames
        self.resolution = resolution
        self.fps = fps
        self.split = split

        if source == "disney":
            items = _load_disney(cache_dir, num_frames, resolution, max_samples)
        elif source == "kinetics":
            items = _load_kinetics(cache_dir, num_frames, resolution, max_samples, fps)
        elif source == "synthetic":
            items = _load_synthetic(max_samples, num_frames, resolution)
        else:
            raise ValueError(f"Unknown source {source!r}")

        if not items:
            raise SystemExit(
                f"No clips loaded from source {source!r}. Check connectivity, or "
                "set `data.source: synthetic` in config.yaml to run offline.")

        duration = num_frames / float(fps)
        for item in items:
            if item["audio"] is None:
                item["audio"] = synthesize_audio(item["frames"], duration)
            # The video-only mode learns the filter, so its target is the
            # stylized clip and its conditioning is the original.
            item["styled"] = anime_stylize(item["frames"]) if mode == "video" else None

        self.items = items
        self.num_frames = num_frames

    # -- WeightsLab hooks ---------------------------------------------------
    def get_audio(self, index):
        """Soundtrack for sample ``index`` (WeightsLab GetMedia hook)."""
        item = self.items[int(index) % len(self.items)]
        return item["audio"], AUDIO_SAMPLE_RATE

    # -- Dataset protocol ---------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        target = item["styled"] if self.mode == "video" else item["frames"]

        # `inputs` MUST be a tuple whose FIRST element is the clip to preview:
        # WeightsLab plucks element [0] of a nested tuple for the poster frame
        # (a dict here would break preview extraction entirely). Conditioning
        # rides along in the remaining slots.
        target_t = torch.from_numpy(np.ascontiguousarray(target))
        if self.mode == "text":
            inputs = (target_t,)
        else:
            source_t = torch.from_numpy(np.ascontiguousarray(item["frames"]))
            inputs = (target_t, source_t)

        metadata = {
            "uid": f"{self.split}_{idx:05d}",
            "caption": item["caption"],
            "mode": self.mode,
        }
        # The label is the caption: it is what a human reviewer needs in order
        # to judge whether a generation matches what was asked for.
        return inputs, idx, item["caption"], metadata
