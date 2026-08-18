"""Video / audio preview utilities (generative media support).

Video samples (task_type "video_generation", or any dataset declaring
``media_kind = "video"``) cannot be PIL-encoded as a single image, so the
studio pipeline previews them the same way point clouds are previewed — with a
server-rendered still — and ships the playable media over a separate stream:

  * thumbnails / preview cache / grid / list  -> ``video_poster_frame``
    (one representative frame, encoded by the normal WebP path). The grid and
    list therefore stay cheap: they never carry video bytes.
  * the modal player fetches the muxed clip via the GetMedia RPC
    (``encode_video_mp4`` does frame + audio muxing), and can page through
    individual frames via ``video_frame_at``.
  * audio-only samples ("audio_generation") stream a WAV/AAC track through the
    same RPC with ``kind = "audio"``.

Image-generation samples (task_type "image_generation") need none of this —
they are ordinary 2D arrays and flow through the existing image path. They are
covered here only by ``is_generation_task`` so the label/prediction gates in
data_service treat generated output as free-form rather than as a mask.

Datasets can replace or supply any part of this by defining attributes:

  * ``media_kind`` -> "video" | "audio" | "image"   (disambiguates 4-D arrays,
    which would otherwise be indistinguishable from volumetric medical data)
  * ``fps`` / ``frame_rate`` -> float, defaults to WL_VIDEO_FPS (8)
  * ``get_audio(index) -> (np.ndarray [S] or [S, C], sample_rate)``
  * ``render_video_poster(frames) -> PIL.Image | np.ndarray``

The frame array layout accepted everywhere below is ``[T, H, W, C]`` uint8 or
float. ``_get_image_array_and_metadata`` in data_utils already normalizes the
common PyTorch ``[T, C, H, W]`` layout into that form.
"""
import io
import logging
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

from PIL import Image

logger = logging.getLogger(__name__)

# Canonical task types for generative media. TEXT_GENERATION_TASK lives here
# too (not just video/audio/image) because this module is the shared registry
# data_service.py checks before treating a target/prediction as a mask — text
# generation needs that exclusion exactly like the others, even though its
# own preview path (a plain string, no pixels) is handled in data_utils.py.
VIDEO_GENERATION_TASK = "video_generation"
IMAGE_GENERATION_TASK = "image_generation"
AUDIO_GENERATION_TASK = "audio_generation"
TEXT_GENERATION_TASK = "text_generation"

_GENERATION_TASKS = (
    VIDEO_GENERATION_TASK,
    IMAGE_GENERATION_TASK,
    AUDIO_GENERATION_TASK,
    TEXT_GENERATION_TASK,
)

# Dataset attribute names.
MEDIA_KIND_ATTRS = ("media_kind", "wl_media_kind")
FPS_ATTRS = ("fps", "frame_rate")
AUDIO_HOOK = "get_audio"
POSTER_HOOK = "render_video_poster"

# Global decorator registry (set via wl.video_poster).
_REGISTERED_POSTER_FN = None

# A clip must have at least this many frames before a 4-D array is even
# considered video. Below it, the array is far more likely to be a volumetric
# stack that the existing middle-slice path already handles well.
MIN_VIDEO_FRAMES = 2
# Channel counts that make sense as displayable frames.
_VIDEO_CHANNELS = (1, 3, 4)


def register_poster_fn(fn):
    """Register a global video poster renderer (see wl.video_poster)."""
    global _REGISTERED_POSTER_FN
    _REGISTERED_POSTER_FN = fn
    return fn


# ---------------------------------------------------------------------------
# Task-type / media-kind predicates
# ---------------------------------------------------------------------------
def is_generation_task(task_type) -> bool:
    """True for any generative task type (video, image, or audio generation)."""
    return str(task_type or "").strip().lower() in _GENERATION_TASKS


def is_video_task(task_type) -> bool:
    """True when the task type denotes video output."""
    return str(task_type or "").strip().lower() == VIDEO_GENERATION_TASK


def is_audio_task(task_type) -> bool:
    """True when the task type denotes audio-only output."""
    return str(task_type or "").strip().lower() == AUDIO_GENERATION_TASK


def get_media_kind(dataset) -> str:
    """Return the dataset's declared media kind ("" when it declares none)."""
    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    for source in (wrapped, dataset):
        for attr in MEDIA_KIND_ATTRS:
            value = getattr(source, attr, None)
            if value:
                return str(value).strip().lower()
    return ""


def get_fps(dataset, default: float = None) -> float:
    """Frame rate for a video dataset (dataset attr -> WL_VIDEO_FPS -> 8)."""
    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    for source in (wrapped, dataset):
        for attr in FPS_ATTRS:
            value = getattr(source, attr, None)
            if value:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    return parsed
    if default is not None:
        return default
    try:
        return max(1.0, float(os.environ.get("WL_VIDEO_FPS", "8")))
    except (TypeError, ValueError):
        return 8.0


def looks_like_video(arr) -> bool:
    """Shape heuristic: does this array look like a [T, H, W, C] clip?

    Deliberately conservative — 4-D arrays are ambiguous (volumetric scans have
    the same rank), so callers must additionally confirm via the task type or
    the dataset's ``media_kind``. This function only rules out shapes that
    could not be a clip at all.
    """
    if arr is None:
        return False
    shape = getattr(arr, "shape", None)
    if shape is None or len(shape) != 4:
        return False
    frames, height, width, channels = shape
    return (
        frames >= MIN_VIDEO_FRAMES
        and height >= 2
        and width >= 2
        and channels in _VIDEO_CHANNELS
    )


def is_video_sample(dataset, arr, task_type=None) -> bool:
    """Full routing decision for the preview pipeline.

    A sample is treated as video when it has a clip-shaped array AND the
    dataset opts in — either through the task type or an explicit
    ``media_kind`` attribute. Without the opt-in, volumetric data keeps its
    existing middle-slice behaviour.
    """
    if not looks_like_video(arr):
        return False
    if is_video_task(task_type):
        return True
    if get_media_kind(dataset) == "video":
        return True
    raw_task = getattr(
        getattr(dataset, "wrapped_dataset", dataset), "task_type",
        getattr(dataset, "task_type", None))
    return is_video_task(raw_task)


# ---------------------------------------------------------------------------
# Frame handling
# ---------------------------------------------------------------------------
def _frames_to_uint8(frames) -> np.ndarray:
    """Normalize a [T, H, W, C] array to contiguous uint8 RGB/L frames."""
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        peak = float(finite.max()) if finite.size else 0.0
        low = float(finite.min()) if finite.size else 0.0
        # [-1, 1] generative output is the common case; then [0, 1]; then raw.
        if low < -0.01:
            arr = (arr + 1.0) * 127.5
        elif peak <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def video_frame_at(frames, index: int) -> Image.Image:
    """Return frame ``index`` of a clip as a PIL image (clamped to range)."""
    arr = _frames_to_uint8(frames)
    if arr.shape[0] == 0:
        raise ValueError("Cannot extract a frame from an empty clip.")
    idx = max(0, min(int(index), arr.shape[0] - 1))
    return _frame_to_pil(arr[idx])


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    """One [H, W, C] uint8 frame -> PIL image."""
    if frame.ndim == 2:
        return Image.fromarray(frame, mode="L")
    channels = frame.shape[-1]
    if channels == 1:
        return Image.fromarray(frame[..., 0], mode="L")
    if channels == 4:
        return Image.fromarray(frame, mode="RGBA")
    if channels == 3:
        return Image.fromarray(frame, mode="RGB")
    return Image.fromarray(frame[..., 0], mode="L")


def video_poster_frame(dataset, frames) -> Image.Image:
    """Render the still shown in the grid, list, and as the player poster.

    Resolution order mirrors the point-cloud thumbnail hook chain:
    dataset ``render_video_poster`` -> globally registered fn -> middle frame.
    The middle frame is used rather than the first because generative clips
    frequently start from noise or a black fade-in.
    """
    arr = _frames_to_uint8(frames)

    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    for source in (wrapped, dataset):
        hook = getattr(source, POSTER_HOOK, None)
        if callable(hook):
            try:
                rendered = hook(arr)
                if rendered is not None:
                    return rendered if isinstance(rendered, Image.Image) \
                        else _frame_to_pil(_frames_to_uint8(rendered[None])[0])
            except Exception as exc:
                logger.warning("Dataset %s hook failed: %r", POSTER_HOOK, exc)

    if _REGISTERED_POSTER_FN is not None:
        try:
            rendered = _REGISTERED_POSTER_FN(arr)
            if rendered is not None:
                return rendered if isinstance(rendered, Image.Image) \
                    else _frame_to_pil(_frames_to_uint8(rendered[None])[0])
        except Exception as exc:
            logger.warning("Registered video poster fn failed: %r", exc)

    return _frame_to_pil(arr[arr.shape[0] // 2])


# ---------------------------------------------------------------------------
# Audio handling
# ---------------------------------------------------------------------------
def load_sample_audio(dataset, index):
    """Fetch ``(samples, sample_rate)`` for one sample, or ``(None, 0)``.

    The dataset opts in by defining ``get_audio(index)``. Returned samples may
    be [S] mono or [S, C]; float in [-1, 1] or int16.
    """
    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    for source in (wrapped, dataset):
        hook = getattr(source, AUDIO_HOOK, None)
        if callable(hook):
            try:
                result = hook(index)
            except Exception as exc:
                logger.warning("Dataset %s(%s) failed: %r", AUDIO_HOOK, index, exc)
                return None, 0
            if result is None:
                return None, 0
            samples, sample_rate = result
            if samples is None:
                return None, 0
            return np.asarray(samples), int(sample_rate or 0)
    return None, 0


def _audio_to_int16(samples: np.ndarray) -> np.ndarray:
    """Normalize audio to interleaved int16, shape [S, C]."""
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        # [C, S] channel-first -> [S, C]; safe because real clips have far more
        # samples than channels.
        arr = arr.T
    if arr.dtype != np.int16:
        arr = np.asarray(arr, dtype=np.float32)
        peak = float(np.abs(arr).max()) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak
        arr = np.clip(arr, -1.0, 1.0) * 32767.0
        arr = arr.astype(np.int16)
    return np.ascontiguousarray(arr)


def encode_audio_wav(samples, sample_rate: int) -> bytes:
    """Encode audio to a WAV container (browser-playable, no ffmpeg needed)."""
    pcm = _audio_to_int16(samples)
    if pcm.size == 0 or sample_rate <= 0:
        return b""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Muxing
# ---------------------------------------------------------------------------
def _ffmpeg_binary() -> str:
    """Locate an ffmpeg executable, preferring the pip-installable bundle.

    Returns "" when none is available; callers degrade to an animated preview
    rather than failing the request.
    """
    override = os.environ.get("WL_FFMPEG_BINARY", "").strip()
    if override:
        return override
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg") or ""


def ffmpeg_available() -> bool:
    """True when video muxing to MP4 is possible in this process."""
    return bool(_ffmpeg_binary())


def encode_video_mp4(frames, fps: float, audio=None, sample_rate: int = 0,
                     crf: int = None) -> bytes:
    """Mux ``frames`` (+ optional ``audio``) into a browser-playable MP4.

    H.264 + AAC in an MP4 with the moov atom moved to the front, so the studio
    can play it straight from a Blob URL without waiting for the whole file.
    Returns b"" when ffmpeg is unavailable — callers fall back to ``encode_video_gif``.
    """
    binary = _ffmpeg_binary()
    if not binary:
        return b""

    arr = _frames_to_uint8(frames)
    if arr.shape[0] == 0:
        return b""
    # H.264 requires even dimensions; pad rather than crop so nothing is lost.
    if arr.shape[1] % 2 or arr.shape[2] % 2:
        arr = np.pad(
            arr,
            ((0, 0), (0, arr.shape[1] % 2), (0, arr.shape[2] % 2), (0, 0)),
            mode="edge")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    frame_count, height, width, _ = arr.shape
    if crf is None:
        try:
            crf = int(os.environ.get("WL_VIDEO_CRF", "23"))
        except (TypeError, ValueError):
            crf = 23

    wav_bytes = b""
    if audio is not None and sample_rate:
        wav_bytes = encode_audio_wav(audio, sample_rate)

    tmp_dir = tempfile.mkdtemp(prefix="wl_video_")
    out_path = os.path.join(tmp_dir, "clip.mp4")
    audio_path = os.path.join(tmp_dir, "audio.wav")
    try:
        cmd = [
            binary, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{max(1.0, float(fps))}",
            "-i", "-",
        ]
        if wav_bytes:
            with open(audio_path, "wb") as handle:
                handle.write(wav_bytes)
            cmd += ["-i", audio_path, "-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            # yuv420p + even dims is what browsers actually decode.
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.run(
            cmd, input=arr.tobytes(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not os.path.exists(out_path):
            logger.warning(
                "ffmpeg failed to encode %d frames: %s",
                frame_count, proc.stderr.decode("utf-8", "replace")[:400])
            return b""
        with open(out_path, "rb") as handle:
            return handle.read()
    except Exception as exc:
        logger.warning("MP4 encoding failed: %r", exc)
        return b""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def encode_video_gif(frames, fps: float) -> bytes:
    """Animated-GIF fallback used when ffmpeg is not installed.

    Silent and much larger than H.264, but it keeps the modal player useful on
    a bare ``pip install weightslab`` with no system ffmpeg.
    """
    arr = _frames_to_uint8(frames)
    if arr.shape[0] == 0:
        return b""
    images = [_frame_to_pil(frame).convert("P", palette=Image.ADAPTIVE)
              for frame in arr]
    buffer = io.BytesIO()
    images[0].save(
        buffer, format="GIF", save_all=True, append_images=images[1:],
        duration=max(1, int(1000.0 / max(1.0, float(fps)))), loop=0)
    return buffer.getvalue()


def describe_clip(dataset) -> dict:
    """Cheap clip descriptor for the grid, read from dataset attributes only.

    Called once per rendered row, so it must never decode a clip. Datasets that
    do not advertise ``num_frames`` simply omit it and the studio falls back to
    whatever the GetMedia stream reports when the player opens.
    """
    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    info = {"kind": "video", "fps": get_fps(dataset)}

    for attr in ("num_frames", "clip_length", "frames_per_clip"):
        for source in (wrapped, dataset):
            value = getattr(source, attr, None)
            if value:
                try:
                    info["frame_count"] = int(value)
                except (TypeError, ValueError):
                    continue
                break
        if "frame_count" in info:
            break

    info["has_audio"] = any(
        callable(getattr(source, AUDIO_HOOK, None)) for source in (wrapped, dataset))
    if info.get("frame_count"):
        info["duration_seconds"] = round(
            info["frame_count"] / max(1.0, info["fps"]), 3)
    return info


def encode_clip(frames, fps: float, audio=None, sample_rate: int = 0):
    """Encode a clip for transport, returning ``(bytes, mime, has_audio)``."""
    mp4 = encode_video_mp4(frames, fps, audio=audio, sample_rate=sample_rate)
    if mp4:
        return mp4, "video/mp4", bool(audio is not None and sample_rate)
    gif = encode_video_gif(frames, fps)
    if gif:
        return gif, "image/gif", False
    return b"", "", False
