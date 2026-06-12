"""Download individual KITTI *raw* drive sequences from the public S3 mirror.

KITTI raw sequences (``<date>_<drive>_sync.zip``) contain synchronized
Velodyne scans, camera images and calibration — but **no 3D box labels**
(those ship only with the separate object-detection benchmark). So this path
is for exploring real-world LiDAR and viewing model predictions in the studio,
not for supervised training. See the example README.

On-disk layout after extraction (under ``dest_dir``):

    <date>/
      calib_cam_to_cam.txt
      calib_velo_to_cam.txt
      calib_imu_to_velo.txt
      <date>_<drive>_sync/
        velodyne_points/data/0000000000.bin ...   (x, y, z, reflectance float32)
        image_02/data/0000000000.png ...           (left colour camera)
        ...

Downloads stream to disk with a tqdm progress bar and are idempotent (a
present zip / extracted folder is reused).
"""
import os
import ssl
import zipfile
import urllib.request

from tqdm import tqdm

BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"


def _remote_url(date, drive, kind):
    """Build the S3 URL for a raw-data archive (files nest under <date>_<drive>/).

    kind is "sync" or "tracklets".
    """
    return f"{BASE_URL}{date}_{drive}/{date}_{drive}_{kind}.zip"

# A few small, known-good sequences (size estimates). The full list lives at
# https://www.cvlibs.net/datasets/kitti/raw_data.php
SMALL_SEQUENCES = {
    ("2011_09_26", "drive_0001"): "~0.5 GB",
    ("2011_09_26", "drive_0018"): "~0.7 GB",
    ("2011_09_26", "drive_0060"): "~0.7 GB",
    ("2011_09_28", "drive_0001"): "~1.1 GB",
}


def default_download_dir():
    """Stable temp directory used when the config gives no explicit path."""
    import tempfile
    return os.path.join(tempfile.gettempdir(), "weightslab_kitti_raw")


def _tqdm_urlretrieve(url, filepath):
    """urllib download with a tqdm progress bar (corporate-TLS fallback)."""
    def _open(ctx=None):
        return urllib.request.urlopen(url, timeout=60, context=ctx)

    try:
        resp = _open()
    except Exception as e:
        # Some corporate environments break TLS verification — retry unverified.
        print(f"[kitti] TLS verification failed ({e}); retrying without verification.", flush=True)
        resp = _open(ssl._create_unverified_context())

    total = int(resp.headers.get("content-length", 0))
    with open(filepath, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(filepath)
    ) as pbar:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            pbar.update(len(chunk))
    resp.close()


def ensure_calib(date, dest_dir=None, keep_zip=False):
    """Download + extract the date-level calibration (a small, separate zip).

    Calibration ships once per date as ``<date>_calib.zip`` (not inside the
    per-drive sync archive). Returns the ``<date>`` dir, or None on failure.
    """
    dest_dir = dest_dir or default_download_dir()
    os.makedirs(dest_dir, exist_ok=True)
    date_dir = os.path.join(dest_dir, date)
    if os.path.exists(os.path.join(date_dir, "calib_velo_to_cam.txt")):
        return date_dir

    filename = f"{date}_calib.zip"
    zip_path = os.path.join(dest_dir, filename)
    try:
        if not os.path.exists(zip_path):
            print(f"[kitti] Downloading {filename} (calibration) ...", flush=True)
            _tqdm_urlretrieve(BASE_URL + filename, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    except Exception as e:
        print(f"[kitti] Calibration download failed for {date} ({e}).", flush=True)
        return None
    finally:
        if not keep_zip and os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass
    return date_dir if os.path.exists(os.path.join(date_dir, "calib_velo_to_cam.txt")) else None


def ensure_tracklets(date, drive, dest_dir=None, keep_zip=False):
    """Download + extract the 3D-box tracklet labels for one raw drive.

    Many (not all) raw drives have a small ``<date>_<drive>_tracklets.zip``
    on the same mirror, extracting to
    ``<date>/<date>_<drive>_sync/tracklet_labels.xml``. Returns that XML path,
    or None if the drive has no tracklets / download failed.
    """
    dest_dir = dest_dir or default_download_dir()
    os.makedirs(dest_dir, exist_ok=True)

    xml_path = os.path.join(dest_dir, date, f"{date}_{drive}_sync", "tracklet_labels.xml")
    if os.path.exists(xml_path):
        return xml_path

    filename = f"{date}_{drive}_tracklets.zip"
    zip_path = os.path.join(dest_dir, filename)
    try:
        if not os.path.exists(zip_path):
            print(f"[kitti] Downloading {filename} (3D box labels) ...", flush=True)
            _tqdm_urlretrieve(_remote_url(date, drive, "tracklets"), zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    except Exception as e:
        print(f"[kitti] No tracklets for {date}_{drive} ({e}).", flush=True)
        return None
    finally:
        if not keep_zip and os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass
    return xml_path if os.path.exists(xml_path) else None


def ensure_sequence(date, drive, dest_dir=None, keep_zip=False):
    """Download + extract one raw sequence (idempotent). Returns the date dir.

    Args:
        date:     e.g. "2011_09_26".
        drive:    e.g. "drive_0001".
        dest_dir: where to download/extract (default: a temp dir).
        keep_zip: keep the downloaded .zip after extraction (default: delete).

    Returns:
        Path to ``<dest_dir>/<date>`` (the extracted sequence root).
    """
    dest_dir = dest_dir or default_download_dir()
    os.makedirs(dest_dir, exist_ok=True)

    seq_dir = os.path.join(dest_dir, date, f"{date}_{drive}_sync")
    if os.path.isdir(os.path.join(seq_dir, "velodyne_points", "data")):
        return os.path.join(dest_dir, date)  # already extracted

    filename = f"{date}_{drive}_sync.zip"
    zip_path = os.path.join(dest_dir, filename)
    if not os.path.exists(zip_path):
        print(f"[kitti] Downloading {filename} -> {zip_path}", flush=True)
        _tqdm_urlretrieve(_remote_url(date, drive, "sync"), zip_path)

    print(f"[kitti] Extracting {filename} ...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    if not keep_zip:
        try:
            os.remove(zip_path)
        except OSError:
            pass
    return os.path.join(dest_dir, date)


def list_sequence_frames(date_dir, date, drive):
    """List velodyne .bin frame stems for one extracted drive, sorted."""
    velo_dir = os.path.join(date_dir, f"{date}_{drive}_sync", "velodyne_points", "data")
    if not os.path.isdir(velo_dir):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(velo_dir) if f.endswith(".bin"))
