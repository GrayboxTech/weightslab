"""Servicer-level tests for the GetPointCloud streaming RPC."""
import numpy as np

import weightslab.proto.experiment_service_pb2 as pb2

from weightslab.trainer.services.data_service import (
    DataService,
    _DEFAULT_POINT_CLOUD_CHUNK_BYTES,
    _point_cloud_chunk_bytes,
)


PC_RANGE = (0.0, -32.0, -3.0, 64.0, 32.0, 1.0)


class _FakeLidarDataset:
    task_type = "detection_pointcloud"
    pc_range = PC_RANGE

    def __init__(self, n_points=50_000):
        rng = np.random.default_rng(0)
        self._cloud = np.stack([
            rng.uniform(0, 64, n_points),
            rng.uniform(-32, 32, n_points),
            rng.uniform(-2.0, 0.5, n_points),
            rng.uniform(0, 1, n_points),
        ], axis=1).astype(np.float32)

    def get_index_from_sample_id(self, sample_id):
        if int(sample_id) != 7:
            raise KeyError(sample_id)
        return 0

    def __getitem__(self, idx):
        return self._cloud, "uid_000007", None, None

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
        return self._cloud, "uid_000007", None, None


class _StubService:
    """Minimal stand-in exposing only what GetPointCloud touches."""
    _POINT_CLOUD_CHUNK_BYTES = DataService._POINT_CLOUD_CHUNK_BYTES
    GetPointCloud = DataService.GetPointCloud

    def __init__(self, dataset):
        self._dataset = dataset

    def _get_dataset(self, origin):
        return self._dataset if origin == "train_loader" else None


def _collect(stub, request):
    return list(stub.GetPointCloud(request, context=None))


def test_get_point_cloud_streams_full_cloud():
    stub = _StubService(_FakeLidarDataset())
    chunks = _collect(stub, pb2.PointCloudRequest(sample_id="7", origin="train_loader"))

    assert all(c.success for c in chunks)
    first = chunks[0]
    assert first.num_points == 50_000
    assert first.num_features == 4
    assert list(first.pc_range) == list(PC_RANGE)
    assert first.total_chunks == len(chunks)

    payload = b"".join(c.data for c in chunks)
    assert len(payload) == 50_000 * 4 * 4
    decoded = np.frombuffer(payload, dtype="<f4").reshape(50_000, 4)
    assert np.isfinite(decoded).all()


def test_get_point_cloud_respects_max_points():
    stub = _StubService(_FakeLidarDataset())
    chunks = _collect(stub, pb2.PointCloudRequest(
        sample_id="7", origin="train_loader", max_points=1234))
    assert chunks[0].num_points == 1234
    payload = b"".join(c.data for c in chunks)
    assert len(payload) == 1234 * 4 * 4


def test_get_point_cloud_unknown_sample_fails_gracefully():
    stub = _StubService(_FakeLidarDataset())
    chunks = _collect(stub, pb2.PointCloudRequest(sample_id="99", origin="train_loader"))
    assert len(chunks) == 1
    assert not chunks[0].success
    assert "not found" in chunks[0].message


def test_point_cloud_chunk_bytes_default(monkeypatch):
    monkeypatch.delenv("WL_POINT_CLOUD_CHUNK_BYTES", raising=False)
    assert _point_cloud_chunk_bytes() == _DEFAULT_POINT_CLOUD_CHUNK_BYTES == (1 << 20)


def test_point_cloud_chunk_bytes_env_override(monkeypatch):
    monkeypatch.setenv("WL_POINT_CLOUD_CHUNK_BYTES", "4096")
    assert _point_cloud_chunk_bytes() == 4096


def test_point_cloud_chunk_bytes_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("WL_POINT_CLOUD_CHUNK_BYTES", "not-a-number")
    assert _point_cloud_chunk_bytes() == _DEFAULT_POINT_CLOUD_CHUNK_BYTES
    monkeypatch.setenv("WL_POINT_CLOUD_CHUNK_BYTES", "0")
    assert _point_cloud_chunk_bytes() == _DEFAULT_POINT_CLOUD_CHUNK_BYTES
    monkeypatch.setenv("WL_POINT_CLOUD_CHUNK_BYTES", "-10")
    assert _point_cloud_chunk_bytes() == _DEFAULT_POINT_CLOUD_CHUNK_BYTES


def test_get_point_cloud_honours_configured_chunk_size():
    """A smaller chunk size splits the same cloud into more (correct) messages."""
    class _SmallChunkService(_StubService):
        _POINT_CLOUD_CHUNK_BYTES = 4096 # bytes

    stub = _SmallChunkService(_FakeLidarDataset())
    chunks = _collect(stub, pb2.PointCloudRequest(sample_id="7", origin="train_loader"))

    total_bytes = 50_000 * 4 * 4
    assert len(chunks) > 1
    assert all(len(c.data) <= 4096 for c in chunks)
    assert chunks[0].total_chunks == len(chunks)
    assert sum(len(c.data) for c in chunks) == total_bytes


def test_get_point_cloud_non_pointcloud_sample_fails_gracefully():
    class ImgDataset(_FakeLidarDataset):
        def get_items(self, idx, **kwargs):
            return np.zeros((64, 64, 3), np.float32), "uid", None, None

        def __getitem__(self, idx):
            return np.zeros((64, 64, 3), np.float32), "uid", None, None

    stub = _StubService(ImgDataset())
    chunks = _collect(stub, pb2.PointCloudRequest(sample_id="7", origin="train_loader"))
    assert len(chunks) == 1
    assert not chunks[0].success
    assert "not a point cloud" in chunks[0].message
