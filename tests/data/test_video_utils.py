"""Unit tests for the video / audio preview utilities."""
import numpy as np
import pytest

from PIL import Image

from weightslab.data import video_utils as vu


def _clip(frames=8, height=32, width=32, channels=3, dtype=np.uint8):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (frames, height, width, channels))
    return arr.astype(dtype)


# ---------------------------------------------------------------------------
# Task-type predicates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task,expected", [
    ("video_generation", True),
    ("image_generation", True),
    ("audio_generation", True),
    ("  Video_Generation  ", True),
    ("classification", False),
    ("segmentation", False),
    ("detection_pointcloud", False),
    ("", False),
    (None, False),
])
def test_is_generation_task(task, expected):
    assert vu.is_generation_task(task) is expected


def test_is_video_task_is_narrower_than_is_generation_task():
    assert vu.is_video_task("video_generation")
    assert not vu.is_video_task("image_generation")
    assert not vu.is_video_task("audio_generation")


def test_generation_tasks_do_not_collide_with_point_cloud_routing():
    """The point-cloud router matches substrings "3d"/"point"/"lidar".

    A generative task name containing any of them would silently be sent to the
    BEV renderer, so guard the chosen names against that regression.
    """
    from weightslab.data.point_cloud_utils import is_point_cloud_task

    for task in (vu.VIDEO_GENERATION_TASK, vu.IMAGE_GENERATION_TASK,
                 vu.AUDIO_GENERATION_TASK):
        assert not is_point_cloud_task(task)


# ---------------------------------------------------------------------------
# Shape heuristics / routing
# ---------------------------------------------------------------------------
def test_looks_like_video_accepts_clip_shape():
    assert vu.looks_like_video(_clip())


@pytest.mark.parametrize("arr", [
    None,
    np.zeros((32, 32, 3), dtype=np.uint8),      # single image
    np.zeros((8, 32, 32, 7), dtype=np.uint8),   # implausible channel count
    np.zeros((1, 32, 32, 3), dtype=np.uint8),   # single frame
    np.zeros((100, 4), dtype=np.float32),       # point cloud
])
def test_looks_like_video_rejects_non_clips(arr):
    assert not vu.looks_like_video(arr)


def test_video_routing_requires_explicit_opt_in():
    """A bare 4-D array must NOT be treated as video.

    Volumetric scans have the same rank; without this guard every medical
    volume would start rendering as a clip.
    """
    class Dataset:
        pass

    assert not vu.is_video_sample(Dataset(), _clip())


def test_video_routing_via_task_type_and_media_kind():
    class Bare:
        pass

    class Declared:
        media_kind = "video"

    class Tasked:
        task_type = "video_generation"

    assert vu.is_video_sample(Bare(), _clip(), "video_generation")
    assert vu.is_video_sample(Declared(), _clip())
    assert vu.is_video_sample(Tasked(), _clip())


def test_volumetric_data_is_not_hijacked_by_media_kind_image():
    class Volume:
        media_kind = "image"

    assert not vu.is_video_sample(Volume(), _clip())


# ---------------------------------------------------------------------------
# Frames / poster
# ---------------------------------------------------------------------------
def test_poster_uses_middle_frame_by_default():
    clip = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    clip[2] = 200  # middle frame is the distinctive one

    poster = vu.video_poster_frame(object(), clip)
    assert np.asarray(poster).max() == 200


def test_poster_honours_dataset_hook():
    class Dataset:
        def render_video_poster(self, frames):
            return Image.new("RGB", (11, 7), (1, 2, 3))

    poster = vu.video_poster_frame(Dataset(), _clip())
    assert poster.size == (11, 7)


def test_poster_falls_back_when_dataset_hook_raises():
    class Dataset:
        def render_video_poster(self, frames):
            raise RuntimeError("boom")

    # A broken user hook must degrade to the built-in poster, never propagate.
    poster = vu.video_poster_frame(Dataset(), _clip())
    assert poster.size == (32, 32)


@pytest.mark.parametrize("scale,offset,label", [
    (255.0, 0.0, "raw 0-255"),
    (1.0, 0.0, "normalized 0-1"),
    (2.0, -1.0, "generative -1..1"),
])
def test_float_frames_are_rescaled_to_visible_range(scale, offset, label):
    # Every frame spans the full 0..1 range, so the poster (the middle frame)
    # is representative regardless of which frame gets picked.
    ramp = np.linspace(0, 1, 8 * 8 * 3).reshape(1, 8, 8, 3)
    frames = (np.tile(ramp, (4, 1, 1, 1)) * scale + offset).astype(np.float32)

    poster = np.asarray(vu.video_poster_frame(object(), frames))
    assert poster.dtype == np.uint8
    # A washed-out or all-black poster means the range detection misfired.
    assert poster.max() > 200, label


def test_low_contrast_frames_are_not_stretched_to_full_range():
    """Rescaling maps by convention, it does not auto-level.

    A dim clip must stay dim: min/max normalizing here would misrepresent a
    genuinely low-contrast generation as a healthy one.
    """
    frames = np.full((4, 8, 8, 3), 0.5, dtype=np.float32)  # mid-grey, 0..1
    poster = np.asarray(vu.video_poster_frame(object(), frames))
    assert 120 <= poster.min() <= 135
    assert 120 <= poster.max() <= 135


def test_video_frame_at_clamps_out_of_range_index():
    clip = _clip(frames=4)
    assert vu.video_frame_at(clip, -5).size == (32, 32)
    assert vu.video_frame_at(clip, 99).size == (32, 32)


def test_video_frame_at_selects_the_requested_frame():
    clip = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    clip[3] = 123
    assert np.asarray(vu.video_frame_at(clip, 3)).max() == 123


def test_grayscale_and_rgba_clips_are_previewable():
    assert vu.video_poster_frame(object(), _clip(channels=1)).mode == "L"
    assert vu.video_poster_frame(object(), _clip(channels=4)).mode == "RGBA"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def test_encode_audio_wav_roundtrip():
    import io
    import wave

    sample_rate = 16_000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate)

    data = vu.encode_audio_wav(tone, sample_rate)
    with wave.open(io.BytesIO(data), "rb") as handle:
        assert handle.getframerate() == sample_rate
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == sample_rate


def test_encode_audio_wav_handles_stereo_and_channel_first():
    sample_rate = 8_000
    stereo = np.zeros((2, sample_rate), dtype=np.float32)  # [C, S]
    data = vu.encode_audio_wav(stereo, sample_rate)

    import io
    import wave
    with wave.open(io.BytesIO(data), "rb") as handle:
        # [C, S] must be transposed to [S, C], not read as a 2-sample clip.
        assert handle.getnchannels() == 2
        assert handle.getnframes() == sample_rate


def test_encode_audio_wav_empty_is_safe():
    assert vu.encode_audio_wav(np.zeros(0), 16_000) == b""
    assert vu.encode_audio_wav(np.zeros(10), 0) == b""


def test_load_sample_audio_without_hook():
    assert vu.load_sample_audio(object(), 0) == (None, 0)


def test_load_sample_audio_survives_failing_hook():
    class Dataset:
        def get_audio(self, index):
            raise RuntimeError("no audio")

    assert vu.load_sample_audio(Dataset(), 0) == (None, 0)


# ---------------------------------------------------------------------------
# Descriptors / fps
# ---------------------------------------------------------------------------
def test_get_fps_prefers_dataset_attribute():
    class Dataset:
        fps = 24

    assert vu.get_fps(Dataset()) == 24.0


def test_get_fps_ignores_invalid_values(monkeypatch):
    class Dataset:
        fps = "not-a-number"

    monkeypatch.delenv("WL_VIDEO_FPS", raising=False)
    assert vu.get_fps(Dataset()) == 8.0


def test_describe_clip_reports_duration():
    class Dataset:
        num_frames = 24
        fps = 12

        def get_audio(self, index):
            return None

    info = vu.describe_clip(Dataset())
    assert info["frame_count"] == 24
    assert info["fps"] == 12.0
    assert info["has_audio"] is True
    assert info["duration_seconds"] == pytest.approx(2.0)


def test_describe_clip_omits_unknown_frame_count():
    info = vu.describe_clip(object())
    assert "frame_count" not in info
    assert info["has_audio"] is False


# ---------------------------------------------------------------------------
# Muxing
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not vu.ffmpeg_available(), reason="ffmpeg not installed")
def test_encode_video_mp4_produces_a_real_container():
    data = vu.encode_video_mp4(_clip(frames=6), fps=8)
    assert data[4:8] == b"ftyp"


@pytest.mark.skipif(not vu.ffmpeg_available(), reason="ffmpeg not installed")
def test_encode_video_mp4_pads_odd_dimensions():
    """H.264 rejects odd width/height; the encoder must pad, not fail."""
    data = vu.encode_video_mp4(_clip(frames=4, height=31, width=33), fps=8)
    assert data[4:8] == b"ftyp"


@pytest.mark.skipif(not vu.ffmpeg_available(), reason="ffmpeg not installed")
def test_encode_clip_muxes_audio():
    sample_rate = 16_000
    tone = np.sin(2 * np.pi * 440 * np.arange(sample_rate * 2) / sample_rate)

    data, mime, has_audio = vu.encode_clip(
        _clip(frames=16), fps=8, audio=tone, sample_rate=sample_rate)
    assert mime == "video/mp4"
    assert has_audio is True
    # A muxed AAC track makes the file substantially larger than the video-only
    # encode of the same frames.
    video_only, _, _ = vu.encode_clip(_clip(frames=16), fps=8)
    assert len(data) > len(video_only)


def test_encode_clip_falls_back_to_gif_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(vu, "_ffmpeg_binary", lambda: "")

    data, mime, has_audio = vu.encode_clip(_clip(frames=4), fps=8)
    assert mime == "image/gif"
    assert has_audio is False
    assert data[:6] in (b"GIF87a", b"GIF89a")


def test_encode_video_mp4_returns_empty_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(vu, "_ffmpeg_binary", lambda: "")
    assert vu.encode_video_mp4(_clip(), fps=8) == b""


def test_encode_empty_clip_is_safe():
    empty = np.zeros((0, 8, 8, 3), dtype=np.uint8)
    assert vu.encode_video_mp4(empty, fps=8) == b""
    assert vu.encode_video_gif(empty, fps=8) == b""
