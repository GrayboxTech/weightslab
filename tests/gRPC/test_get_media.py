"""Servicer-level tests for the GetMedia streaming RPC."""
import threading

import numpy as np
import pytest

import weightslab.proto.experiment_service_pb2 as pb2

from weightslab.data import video_utils as vu
from weightslab.trainer.services.data_service import (
    DataService,
    _build_media_stats,
    _DEFAULT_MEDIA_CHUNK_BYTES,
    _media_chunk_bytes,
)


SAMPLE_ID = 7
SAMPLE_RATE = 16_000


class _FakeVideoDataset:
    task_type = "video_generation"
    fps = 8
    num_frames = 12

    def __init__(self, frames=12, with_audio=True, height=32, width=32):
        rng = np.random.default_rng(0)
        self._clip = rng.integers(
            0, 255, (frames, height, width, 3)).astype(np.uint8)
        self._with_audio = with_audio
        self.num_frames = frames

    def get_index_from_sample_id(self, sample_id):
        if int(sample_id) != SAMPLE_ID:
            raise KeyError(sample_id)
        return 0

    def __getitem__(self, idx):
        return self._clip, f"uid_{SAMPLE_ID:06d}", None, None

    def get_items(self, idx, include_metadata=False, include_labels=False,
                  include_images=False):
        return self._clip, f"uid_{SAMPLE_ID:06d}", None, None

    def get_audio(self, index):
        if not self._with_audio:
            return None
        duration = self._clip.shape[0] / float(self.fps)
        t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
        return np.sin(2 * np.pi * 440 * t).astype(np.float32), SAMPLE_RATE


class _FakeImageDataset:
    """A non-video dataset, to prove GetMedia refuses it cleanly."""
    task_type = "classification"

    def get_index_from_sample_id(self, sample_id):
        return 0

    def __getitem__(self, idx):
        return np.zeros((32, 32, 3), dtype=np.uint8), "uid", None, None

    def get_items(self, idx, **kwargs):
        return np.zeros((32, 32, 3), dtype=np.uint8), "uid", None, None


class _StubService:
    """Minimal stand-in exposing only what GetMedia touches."""
    _MEDIA_CHUNK_BYTES = DataService._MEDIA_CHUNK_BYTES
    _MEDIA_CACHE_ENTRIES = DataService._MEDIA_CACHE_ENTRIES
    GetMedia = DataService.GetMedia
    _stream_media = DataService._stream_media
    _media_cache_get = DataService._media_cache_get
    _media_cache_put = DataService._media_cache_put
    _locate_sample = DataService._locate_sample

    def __init__(self, dataset):
        self._dataset = dataset
        self._media_cache = {}
        self._media_cache_lock = threading.Lock()

    def _get_dataset(self, origin):
        return self._dataset if origin == "train_loader" else None


def _collect(stub, **kwargs):
    kwargs.setdefault("sample_id", str(SAMPLE_ID))
    kwargs.setdefault("origin", "train_loader")
    return list(stub.GetMedia(pb2.MediaRequest(**kwargs), context=None))


ffmpeg_required = pytest.mark.skipif(
    not vu.ffmpeg_available(), reason="ffmpeg not installed")


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------
@ffmpeg_required
def test_get_media_streams_a_playable_clip():
    stub = _StubService(_FakeVideoDataset())
    chunks = _collect(stub)

    assert all(c.success for c in chunks)
    first = chunks[0]
    assert first.mime_type == "video/mp4"
    assert first.frame_count == 12
    assert first.fps == pytest.approx(8.0)
    assert first.has_audio is True
    assert (first.width, first.height) == (32, 32)
    assert first.duration_seconds == pytest.approx(1.5)
    assert first.total_chunks == len(chunks)

    payload = b"".join(c.data for c in chunks)
    assert len(payload) == first.total_bytes
    assert payload[4:8] == b"ftyp"


@ffmpeg_required
def test_only_the_first_chunk_carries_the_header():
    stub = _StubService(_FakeVideoDataset(frames=32, height=128, width=128))
    # Force many chunks so there is a meaningful tail to check.
    stub._MEDIA_CHUNK_BYTES = 1024
    chunks = _collect(stub)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks[1:], start=1):
        assert chunk.mime_type == ""
        assert chunk.frame_count == 0
        assert chunk.total_bytes == 0
        assert chunk.chunk_index == index


def test_get_media_audio_kind_returns_wav():
    stub = _StubService(_FakeVideoDataset())
    chunks = _collect(stub, kind="audio")

    assert all(c.success for c in chunks)
    first = chunks[0]
    assert first.mime_type == "audio/wav"
    assert first.sample_rate == SAMPLE_RATE
    assert first.has_audio is True
    assert first.duration_seconds == pytest.approx(1.5, abs=0.01)

    payload = b"".join(c.data for c in chunks)
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"


@ffmpeg_required
def test_max_frames_subsamples_across_the_whole_clip():
    """Capping must subsample, not truncate — the tail matters for video."""
    stub = _StubService(_FakeVideoDataset(frames=40))
    chunks = _collect(stub, max_frames=10)

    assert chunks[0].frame_count == 10
    # 40 source frames at 8 fps played back as 10 frames at 8 fps.
    assert chunks[0].duration_seconds == pytest.approx(10 / 8.0)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
@ffmpeg_required
def test_repeated_requests_reuse_the_encoded_clip():
    stub = _StubService(_FakeVideoDataset())
    first = _collect(stub)
    assert len(stub._media_cache) == 1

    # Re-encoding would be observable as a different byte payload only by luck,
    # so assert on the cache itself plus payload identity.
    second = _collect(stub)
    assert len(stub._media_cache) == 1
    assert b"".join(c.data for c in first) == b"".join(c.data for c in second)


@ffmpeg_required
def test_media_cache_evicts_oldest_entries():
    stub = _StubService(_FakeVideoDataset())
    for max_frames in range(1, DataService._MEDIA_CACHE_ENTRIES + 4):
        _collect(stub, max_frames=max_frames + 1)

    assert len(stub._media_cache) <= DataService._MEDIA_CACHE_ENTRIES


@ffmpeg_required
def test_video_and_audio_are_cached_separately():
    stub = _StubService(_FakeVideoDataset())
    _collect(stub)
    _collect(stub, kind="audio")
    assert len(stub._media_cache) == 2


def test_media_cache_is_safe_under_concurrent_access():
    """gRPC serves handlers from a thread pool, so put/evict must be atomic.

    Without the lock, `next(iter(...))` racing an insert raises
    "dictionary changed size during iteration" or over-evicts.
    """
    stub = _StubService(_FakeVideoDataset())
    errors = []

    def hammer(worker):
        try:
            for i in range(400):
                key = (worker, i % 8)
                if stub._media_cache_get(key) is None:
                    stub._media_cache_put(key, {"data": b"x"})
        except Exception as exc:  # pragma: no cover - only on a real race
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(stub._media_cache) <= DataService._MEDIA_CACHE_ENTRIES


# ---------------------------------------------------------------------------
# Failure modes — every one must yield a clean chunk, never raise
# ---------------------------------------------------------------------------
def test_unknown_sample_fails_gracefully():
    stub = _StubService(_FakeVideoDataset())
    chunks = _collect(stub, sample_id="404")

    assert len(chunks) == 1
    assert chunks[0].success is False
    assert "not found" in chunks[0].message


def test_unknown_origin_fails_gracefully():
    stub = _StubService(_FakeVideoDataset())
    chunks = _collect(stub, origin="nope_loader")

    assert len(chunks) == 1
    assert chunks[0].success is False


def test_non_video_sample_fails_gracefully():
    stub = _StubService(_FakeImageDataset())
    chunks = _collect(stub)

    assert len(chunks) == 1
    assert chunks[0].success is False
    assert "not a video" in chunks[0].message


def test_audio_request_without_audio_fails_gracefully():
    stub = _StubService(_FakeVideoDataset(with_audio=False))
    chunks = _collect(stub, kind="audio")

    assert len(chunks) == 1
    assert chunks[0].success is False
    assert "no audio" in chunks[0].message


def test_encode_failure_reports_actionable_message(monkeypatch):
    monkeypatch.setattr(vu, "_ffmpeg_binary", lambda: "")
    monkeypatch.setattr(vu, "encode_video_gif", lambda *a, **k: b"")

    stub = _StubService(_FakeVideoDataset())
    chunks = _collect(stub)

    assert chunks[0].success is False
    assert "ffmpeg" in chunks[0].message


def test_clip_without_ffmpeg_still_streams_as_gif(monkeypatch):
    monkeypatch.setattr(vu, "_ffmpeg_binary", lambda: "")

    stub = _StubService(_FakeVideoDataset(frames=4))
    chunks = _collect(stub)

    assert all(c.success for c in chunks)
    assert chunks[0].mime_type == "image/gif"
    assert chunks[0].has_audio is False
    payload = b"".join(c.data for c in chunks)
    assert payload[:6] in (b"GIF87a", b"GIF89a")


# ---------------------------------------------------------------------------
# Media attached to metadata fields (wl.save_media) — the generated-output path
# ---------------------------------------------------------------------------
def _store_clip(field="pred_video", sample_id=str(SAMPLE_ID), size=9000, **meta):
    from weightslab.data import media_store

    payload = {
        "frame_count": 16, "fps": 8.0, "has_audio": True,
        "width": 64, "height": 64, "duration_seconds": 2.0,
    }
    payload.update(meta)
    media_store.put(field, sample_id, b"\x00" * size, "video/mp4", "video",
                    poster=b"poster", meta=payload)
    return payload


def test_field_media_streams_from_the_store():
    from weightslab.data import media_store
    media_store.clear()
    _store_clip()

    stub = _StubService(_FakeVideoDataset())
    chunks = _collect(stub, field="pred_video")

    assert all(c.success for c in chunks)
    first = chunks[0]
    assert first.mime_type == "video/mp4"
    assert first.frame_count == 16
    assert first.has_audio is True
    assert first.duration_seconds == pytest.approx(2.0)
    assert sum(len(c.data) for c in chunks) == 9000


def test_field_media_does_not_touch_the_dataset():
    """A generated clip lives in the store, not in the dataset.

    The field path must therefore work even when the sample is not a video
    sample at all — otherwise attaching media to a classification dataset
    would be impossible.
    """
    from weightslab.data import media_store
    media_store.clear()
    _store_clip()

    stub = _StubService(_FakeImageDataset())
    chunks = _collect(stub, field="pred_video")

    assert all(c.success for c in chunks)
    assert chunks[0].mime_type == "video/mp4"


def test_missing_field_media_fails_gracefully():
    from weightslab.data import media_store
    media_store.clear()

    stub = _StubService(_FakeVideoDataset())
    chunks = _collect(stub, field="pred_video")

    assert len(chunks) == 1
    assert chunks[0].success is False
    assert "pred_video" in chunks[0].message


def test_field_media_is_not_confused_with_the_input_clip():
    """field="" and field="pred_video" must resolve to different payloads."""
    from weightslab.data import media_store
    media_store.clear()
    _store_clip(size=1234)

    stub = _StubService(_FakeVideoDataset())
    attached = b"".join(c.data for c in _collect(stub, field="pred_video"))
    assert len(attached) == 1234

    if vu.ffmpeg_available():
        own = b"".join(c.data for c in _collect(stub))
        assert own != attached
        assert own[4:8] == b"ftyp"


def test_media_stats_carry_descriptor_and_poster():
    import pandas as pd
    from weightslab.data import media_store
    media_store.clear()
    _store_clip()

    row = pd.Series({
        "sample_id": SAMPLE_ID,
        "media:pred_video": '{"kind":"video","fps":8}',
        "loss": 0.5,
    })
    stats = _build_media_stats(row)

    assert len(stats) == 1
    assert stats[0].name == "media:pred_video"
    assert stats[0].type == "media"
    assert stats[0].value_string == '{"kind":"video","fps":8}'
    assert stats[0].thumbnail == b"poster"


def test_media_stats_survive_an_evicted_payload():
    """The column must keep its shape even when the bytes are gone."""
    import pandas as pd
    from weightslab.data import media_store
    media_store.clear()

    row = pd.Series({
        "sample_id": SAMPLE_ID,
        "media:pred_video": '{"kind":"video"}',
    })
    stats = _build_media_stats(row)

    assert len(stats) == 1
    assert stats[0].thumbnail == b"" # no still, but the descriptor is intact


def test_rows_without_media_emit_no_media_stats():
    import pandas as pd

    row = pd.Series({"sample_id": SAMPLE_ID, "loss": 0.5, "label": "cat"})
    stats = _build_media_stats(row)
    assert stats == []


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_media_chunk_bytes_defaults(monkeypatch):
    monkeypatch.delenv("WL_MEDIA_CHUNK_BYTES", raising=False)
    assert _media_chunk_bytes() == _DEFAULT_MEDIA_CHUNK_BYTES


@pytest.mark.parametrize("raw", ["0", "-5", "not-an-int", ""])
def test_media_chunk_bytes_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv("WL_MEDIA_CHUNK_BYTES", raw)
    assert _media_chunk_bytes() == _DEFAULT_MEDIA_CHUNK_BYTES


def test_media_chunk_bytes_honours_override(monkeypatch):
    monkeypatch.setenv("WL_MEDIA_CHUNK_BYTES", "4096")
    assert _media_chunk_bytes() == 4096


@ffmpeg_required
def test_chunk_sizes_respect_the_configured_bound():
    stub = _StubService(_FakeVideoDataset(frames=24, height=96, width=96))
    stub._MEDIA_CHUNK_BYTES = 2048
    chunks = _collect(stub)

    assert all(len(c.data) <= 2048 for c in chunks)
    assert sum(len(c.data) for c in chunks) == chunks[0].total_bytes
