"""Tests for point-cloud preview utilities (BEV rendering / box projection /
binary packing) and the point-cloud branch of load_raw_image_array."""
import numpy as np
import pytest

from PIL import Image

from weightslab.data.point_cloud_utils import (
    boxes_dimensionality,
    is_point_cloud_detection_task,
    box_format_string,
    colorize_from_image,
    compute_point_normals,
    filter_valid_points,
    get_pc_range,
    get_point_feature_names,
    is_point_cloud_task,
    looks_like_point_cloud,
    pack_point_cloud,
    point_cloud_to_bev_image,
    point_cloud_to_range_image,
    point_distances,
    project_boxes_to_bev,
    register_boxes_fn,
    register_thumbnail_fn,
    render_bev_for_dataset,
    render_thumbnail_2d_for_dataset,
    serialize_pointcloud_box_payload,
    voxel_downsample,
)


PC_RANGE = (0.0, -32.0, -3.0, 64.0, 32.0, 1.0)


def _cloud(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.stack([
        rng.uniform(0, 64, n),
        rng.uniform(-32, 32, n),
        rng.uniform(-2.0, 0.5, n),
        rng.uniform(0, 1, n),
    ], axis=1).astype(np.float32)
    return pts


class _FakeDataset:
    task_type = "detection_pointcloud"
    pc_range = PC_RANGE


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------
def test_is_point_cloud_task():
    assert is_point_cloud_task("detection_pointcloud")
    assert is_point_cloud_task("detection_3d")
    assert is_point_cloud_task("Detection_3D")
    assert is_point_cloud_task("pointcloud_seg")
    assert not is_point_cloud_task("detection")
    assert not is_point_cloud_task("segmentation")
    assert not is_point_cloud_task(None)


def test_is_point_cloud_detection_task():
    assert is_point_cloud_detection_task("detection_pointcloud")
    assert is_point_cloud_detection_task("Detection_PointCloud")
    assert is_point_cloud_detection_task("detection_3d") # legacy alias
    assert not is_point_cloud_detection_task("detection")
    assert not is_point_cloud_detection_task("segmentation")
    assert not is_point_cloud_detection_task(None)


def test_looks_like_point_cloud():
    assert looks_like_point_cloud(_cloud())
    assert looks_like_point_cloud(_cloud()[:, :3])
    assert looks_like_point_cloud(_cloud()[:, :2])
    # Multi-channel clouds (xyz + intensity + normals + rgb = 10 cols) qualify.
    assert looks_like_point_cloud(np.zeros((100, 10), np.float32))
    assert not looks_like_point_cloud(_cloud()[:8]) # too few rows
    assert not looks_like_point_cloud(np.zeros((100, 20), np.float32)) # too many cols
    assert not looks_like_point_cloud(np.zeros((64, 64), np.uint8)) # int image
    assert not looks_like_point_cloud(np.zeros((64, 64, 3), np.float32)) # 3D array


def test_point_distances():
    pts = np.array([[3.0, 4.0, 0.0, 1.0], [0.0, 0.0, 5.0, 0.5]], np.float32)
    d = point_distances(pts)
    np.testing.assert_allclose(d, [5.0, 5.0], rtol=1e-5)


def test_compute_point_normals_planar():
    # Points on the z=0 plane -> normals should be ~+/-z.
    rng = np.random.default_rng(0)
    xy = rng.uniform(-5, 5, (500, 2)).astype(np.float32)
    pts = np.concatenate([xy, np.zeros((500, 1), np.float32)], axis=1)
    normals = compute_point_normals(pts, k=12)
    assert normals.shape == (500, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    assert np.abs(normals[:, 2]).mean() > 0.95 # mostly aligned with z


def test_voxel_downsample_reduces_points():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (5000, 4)).astype(np.float32)
    out = voxel_downsample(pts, voxel_size=0.25)
    assert out.shape[1] == 4
    assert out.shape[0] < pts.shape[0]
    assert out.shape[0] <= 4 ** 3 # at most one point per 0.25 voxel in the unit cube


def test_colorize_from_image():
    image = np.zeros((10, 20, 3), np.uint8)
    image[:, :, 0] = 255 # all red
    pts = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], np.float32)

    def project(p):
        uv = np.stack([np.full(len(p), 5.0), np.full(len(p), 5.0)], axis=1)
        return uv, np.array([True, False])

    rgb = colorize_from_image(pts, image, project)
    np.testing.assert_allclose(rgb[0], [1.0, 0.0, 0.0], atol=1e-5) # sampled red
    np.testing.assert_allclose(rgb[1], [0.5, 0.5, 0.5], atol=1e-5) # invalid -> grey


def test_range_image_shape():
    img = point_cloud_to_range_image(_cloud(2000), image_height=48, image_width=256)
    assert img.size == (256, 48)
    arr = np.asarray(img)
    assert (arr != arr[0, 0]).any() # some points were projected


def test_get_point_feature_names_from_dataset_and_default():
    class DS:
        point_feature_names = ["x", "y", "z", "intensity", "nx", "ny", "nz"]
    assert get_point_feature_names(DS(), 7) == ["x", "y", "z", "intensity", "nx", "ny", "nz"]
    # Defaults when the dataset declares none.
    assert get_point_feature_names(object(), 4) == ["x", "y", "z", "intensity"]
    assert get_point_feature_names(object(), 3) == ["x", "y", "z"]


def test_registered_thumbnail_and_boxes_fns():
    marker = {"called": False}

    def my_thumb(points):
        marker["called"] = True
        return np.full((16, 16, 3), 9, np.uint8)

    register_thumbnail_fn(my_thumb)
    try:
        img = render_thumbnail_2d_for_dataset(object(), _cloud())
        assert marker["called"] and np.asarray(img)[0, 0, 0] == 9
    finally:
        register_thumbnail_fn(None) # reset global state

    def my_boxes(boxes):
        return np.zeros((len(boxes), 6), np.float32)

    register_boxes_fn(my_boxes)
    try:
        from weightslab.data.point_cloud_utils import project_boxes_for_dataset
        out = project_boxes_for_dataset(object(), np.ones((3, 9), np.float32))
        assert out.shape == (3, 6)
    finally:
        register_boxes_fn(None)


def test_filter_valid_points_drops_pads_and_nonfinite():
    pts = _cloud(100)
    pts[10] = -1000.0 # pad row (all coords at PAD_VALUE)
    pts[20, 2] = np.nan
    out = filter_valid_points(pts)
    assert out.shape[0] == 98


# ---------------------------------------------------------------------------
# BEV image
# ---------------------------------------------------------------------------
def test_point_cloud_to_bev_image_shape_and_content():
    img = point_cloud_to_bev_image(_cloud(), pc_range=PC_RANGE, image_size=128)
    assert isinstance(img, Image.Image)
    assert img.size == (128, 128)
    arr = np.asarray(img)
    # Some pixels must differ from the background (points were splatted).
    assert (arr != arr[0, 0]).any()


def test_bev_image_empty_cloud_is_background_only():
    img = point_cloud_to_bev_image(np.zeros((0, 4), np.float32), pc_range=PC_RANGE, image_size=64)
    arr = np.asarray(img)
    assert (arr == arr[0, 0]).all()


def test_render_bev_for_dataset_honors_hook():
    class HookedDataset(_FakeDataset):
        def to_bev_image(self, points):
            return np.full((32, 32, 3), 7, np.uint8)

    img = render_bev_for_dataset(HookedDataset(), _cloud())
    assert img.size == (32, 32)
    assert np.asarray(img)[0, 0, 0] == 7


# ---------------------------------------------------------------------------
# Box projection
# ---------------------------------------------------------------------------
def test_project_boxes_to_bev_3d_geometry():
    # Axis-aligned box centered mid-range: easy to check normalized coords.
    boxes = np.array([[32.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.9]], np.float32)
    bev = project_boxes_to_bev(boxes, PC_RANGE, min_norm_size=0.0)
    assert bev.shape == (1, 6)
    x1, y1, x2, y2, cls, conf = bev[0]
    assert x1 == pytest.approx((32 - 2 - 0) / 64.0, abs=1e-5)
    assert x2 == pytest.approx((32 + 2 - 0) / 64.0, abs=1e-5)
    # y axis flips (image v grows downward): cy=0 -> centered.
    assert (y1 + y2) / 2 == pytest.approx(0.5, abs=1e-5)
    assert cls == 1.0 and conf == pytest.approx(0.9)


def test_project_boxes_yaw_rotation_grows_extent():
    no_yaw = project_boxes_to_bev(
        np.array([[32, 0, -1, 4.0, 2.0, 1.5, 0.0, 0, 1]], np.float32), PC_RANGE, 0.0)
    yawed = project_boxes_to_bev(
        np.array([[32, 0, -1, 4.0, 2.0, 1.5, np.pi / 4, 0, 1]], np.float32), PC_RANGE, 0.0)
    assert (yawed[0, 2] - yawed[0, 0]) > (no_yaw[0, 2] - no_yaw[0, 0]) - 1e-6
    assert (yawed[0, 3] - yawed[0, 1]) > (no_yaw[0, 3] - no_yaw[0, 1])


def test_project_boxes_min_size_clamp():
    tiny = np.array([[32.0, 0.0, -1.0, 0.01, 0.01, 0.01, 0.0, 0, 1]], np.float32)
    bev = project_boxes_to_bev(tiny, PC_RANGE, min_norm_size=0.01)
    assert (bev[0, 2] - bev[0, 0]) >= 0.0099
    assert (bev[0, 3] - bev[0, 1]) >= 0.0099


def test_project_boxes_2d_rows():
    boxes = np.array([[10.0, 5.0, 2.0, 2.0, 2.0, 0.7]], np.float32) # cx,cy,dx,dy,cls,conf
    assert boxes_dimensionality(boxes) == 2
    bev = project_boxes_to_bev(boxes, PC_RANGE, 0.0)
    assert bev[0, 4] == 2.0
    assert bev[0, 5] == pytest.approx(0.7)


def test_box_format_string():
    assert box_format_string(np.zeros((1, 9), np.float32)) == "cx_cy_cz_dx_dy_dz_yaw_cls_conf"
    assert box_format_string(np.zeros((1, 6), np.float32)) == "cx_cy_dx_dy_cls_conf"


# ---------------------------------------------------------------------------
# Payload + range resolution
# ---------------------------------------------------------------------------
def test_serialize_pointcloud_box_payload():
    ds = _FakeDataset()
    boxes = np.array([
        [32.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.3, 1.0, 0.8],
        [10.0, -5.0, -1.2, 0.8, 0.6, 1.7, -1.0, 2.0, 0.5],
    ], np.float32)
    payload = serialize_pointcloud_box_payload(ds, boxes)
    assert payload["format"] == "xyxy"
    assert len(payload["bboxes"]) == 2 and len(payload["bboxes"][0]) == 6
    assert len(payload["bboxes_3d"]) == 2 and len(payload["bboxes_3d"][0]) == 9
    assert payload["pc_range"] == list(PC_RANGE)


def test_get_pc_range_attr_and_auto():
    assert get_pc_range(_FakeDataset()) == PC_RANGE

    class Bare:
        pass
    bare = Bare()
    auto = get_pc_range(bare, _cloud())
    assert auto is not None and len(auto) == 6
    # Cached on the dataset for later (image/box alignment).
    assert get_pc_range(bare) == auto


# ---------------------------------------------------------------------------
# Binary packing (GetPointCloud)
# ---------------------------------------------------------------------------
def test_pack_point_cloud_roundtrip_and_downsample():
    pts = _cloud(5000)
    data, n, f = pack_point_cloud(pts, max_points=0)
    assert (n, f) == (5000, 4)
    decoded = np.frombuffer(data, dtype="<f4").reshape(n, f)
    np.testing.assert_allclose(decoded, pts, rtol=1e-6)

    data2, n2, f2 = pack_point_cloud(pts, max_points=1000, seed=7)
    assert n2 == 1000 and f2 == 4
    assert len(data2) == 1000 * 4 * 4
    # Deterministic for a given seed
    data3, _, _ = pack_point_cloud(pts, max_points=1000, seed=7)
    assert data2 == data3


# ---------------------------------------------------------------------------
# load_raw_image_array integration (BEV branch + image datasets untouched)
# ---------------------------------------------------------------------------
def test_load_raw_image_array_renders_bev_for_point_cloud():
    from weightslab.data.data_utils import load_raw_image_array

    class PcDataset:
        task_type = "detection_pointcloud"
        pc_range = PC_RANGE

        def __getitem__(self, idx):
            return _cloud(), f"uid_{idx}", None, None

        def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
            return _cloud(), f"uid_{idx}", None, None

    np_img, is_volumetric, shape, pil = load_raw_image_array(PcDataset(), 0)
    assert not is_volumetric
    assert pil is not None and pil.mode == "RGB"
    assert pil.size[0] == pil.size[1] # square BEV render
    assert np_img.ndim == 3 and np_img.shape[2] == 3


def test_load_raw_image_array_regular_images_unchanged():
    from weightslab.data.data_utils import load_raw_image_array

    class ImgDataset:
        task_type = "classification"

        def __getitem__(self, idx):
            return np.random.default_rng(0).uniform(0, 1, (32, 32, 3)).astype(np.float32), 0

    np_img, is_volumetric, shape, pil = load_raw_image_array(ImgDataset(), 0)
    assert not is_volumetric
    assert pil is not None
    assert pil.size == (32, 32)
