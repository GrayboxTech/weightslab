import logging
import torch as th
import numpy as np

from PIL import Image

from .lidar_utils import render_lidar, load_point_cloud_data, render_bev_mask

# Init global logger
logger = logging.getLogger(__name__)


def _is_point_cloud(x):
    """Heuristic check to see if a tensor/array is a 3D point cloud (N, 3) or (N, 4)."""
    try:
        # Check if it has shape attribute first
        if not hasattr(x, 'shape'):
            x_np = _to_numpy_safe(x)
            if x_np is None:
                return False
            x = x_np
        
        shape = x.shape
        # (N, 3) for XYZ or (N, 4) for XYZI with N > 10 (arbitrary threshold to avoid confusion with small vectors)
        if len(shape) == 2 and shape[1] in [3, 4] and shape[0] > 10:
            return True
        return False
    except Exception:
        return False


def _to_numpy_safe(x):
    if isinstance(x, (int, float)):
        return np.array([x])

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x)
        except Exception:
            return None

    try:
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    return None


def get_mask(raw, dataset=None, dataset_index=None, raw_data=None):
    # Check if prediction_raw is a numpy array (could be bboxes)
    if isinstance(raw, np.ndarray) and (raw.ndim == 2 or raw.ndim == 3) and raw.shape[-1] >= 4:
        # raw appears to be bboxes (N, 4+) format
        # Get the item (image) to determine mask dimensions
        raw_data = dataset[dataset_index] if dataset is not None and dataset_index is not None else raw_data
        if raw_data is None:
            return raw

        # Extract the item (first element of the tuple)
        if isinstance(raw_data, tuple):
            item = raw_data[0]
        else:
            item = raw_data

        # Convert item to numpy to get shape
        item_np = _to_numpy_safe(item)
        if item_np is not None:
            # Determine height and width from item
            if item_np.ndim == 3:
                # Channels-first format: (C, H, W)
                if item_np.shape[0] < item_np.shape[1]:
                    _, height, width = item_np.shape
                else:
                    # Channels-last format: (H, W, C)
                    height, width, _ = item_np.shape
            elif item_np.ndim == 2:
                # Grayscale: (H, W)
                height, width = item_np.shape
            else:
                # Cannot determine dimensions
                return raw

            # Generate segmentation map from bboxes
            segmentation_map = np.zeros((height, width), dtype=np.int64)

            # Return segmentation map directly if it matches raw shape
            if segmentation_map.shape == raw.shape[-2:]:  # B, C, H, W
                return raw

            # Generate segmentation map from bboxes
            raw = raw[0] if raw.ndim == 3 else raw  # Handle batch dimension if present
            for bbox_data in raw:
                x1, y1, x2, y2 = bbox_data[:4].astype(int)
                # Extract class id if available, otherwise use 1
                class_id = int(bbox_data[4]) if len(bbox_data) > 4 else 1

                # Clip to valid image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Fill the bounding box region
                if x2 > x1 and y2 > y1:
                    segmentation_map[y1:y2, x1:x2] = class_id

            return segmentation_map

    return raw


def load_label(dataset, sample_id):
    """Load label/target from dataset at given index.

    Returns the label in its native format (int, array, etc.).
    """
    # Get index from sample_id
    try:
        index = dataset.get_index_from_sample_id(sample_id)
    except (KeyError, ValueError, IndexError):
        logger.debug(f"Sample ID {sample_id} not found in current dataset. Likely a ghost record from a previous run.")
        return None

    # Get dataset wrapper if exists
    wrapped = getattr(dataset, "wrapped_dataset", dataset)

    # Try direct attribute first with torch dataset architecture
    if hasattr(wrapped, "dataset") and hasattr(wrapped.dataset, "targets"):
        label = wrapped.dataset.targets[index]
        if hasattr(label, 'numpy'):
            label = label.numpy()
        if hasattr(label, 'item') and hasattr(label, 'shape') and label.shape == ():
            label = label.item()
        if isinstance(label, np.ndarray) and (label.ndim == 0 or label.ndim == 1):
            label = label.flatten()
        elif isinstance(label, np.ndarray):
            label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=wrapped[index])
        return label

    # Try common dataset patterns first
    if hasattr(wrapped, '__getitem__'):
        data = wrapped[index]
        if isinstance(data, (list, tuple)) and len(data) > 2:
            # Detection/Segmentation often has extra elements
            classes = _to_numpy_safe(data[3]) if len(data) >= 4 else None
            if classes is not None:
                label = _to_numpy_safe(data[2])  # Second element is typically the label
                # Concat label with classes if available (detection)
                label = np.concatenate([label, classes[..., None]], axis=1)
            else:
                label = _to_numpy_safe(data[2])  # Second element is typically the label
        else:
            raw_label = data[1]
            if isinstance(raw_label, dict):
                # Check for Lidar Config to render BEV mask
                lidar_config = getattr(wrapped, "viz_config", getattr(dataset, "viz_config", {}))

                # If we have lidar config and boxes, render them as a mask for the UI
                if lidar_config and 'boxes' in raw_label:
                    boxes = _to_numpy_safe(raw_label['boxes'])
                    labels = _to_numpy_safe(raw_label['labels'])

                    # Convert to list-of-dicts format for render_bev_mask
                    # We map class IDs to typical Kitti strings or generic names
                    id_to_cls = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

                    fmt_labels = []
                    if boxes is not None and labels is not None:
                        for i in range(len(boxes)):
                            c_id = int(labels[i])
                            cls_name = id_to_cls.get(c_id, "Car")
                            fmt_labels.append({
                                'cls': cls_name,
                                'box': boxes[i]
                            })

                    # Get config params
                    bev_conf = lidar_config.get("bev", {})
                    return render_bev_mask(
                        fmt_labels,
                        res=bev_conf.get("resolution", 0.1),
                        size=bev_conf.get("image_size", 800),
                        cx=bev_conf.get("center_x", 400),
                        cy=bev_conf.get("center_y", 400)
                    )
                return raw_label
            label = _to_numpy_safe(raw_label)  # Second element is typically the label
        label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=data)
        
        if label is None:
            return None
            
        # Check if it has ndim before accessing
        if hasattr(label, 'ndim'):
            return label[0] if label.ndim == 1 else label
        return label

    if hasattr(wrapped, "targets"):
        label = wrapped.targets[index]
        if hasattr(label, 'numpy'):
            label = label.numpy()
        if hasattr(label, 'item') and hasattr(label, 'shape') and label.shape == ():
            label = label.item()
        label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=wrapped[index])
        return label

    if hasattr(wrapped, "labels"):
        label = wrapped.labels[index]
        if hasattr(label, 'numpy'):
            label = label.numpy()
        if hasattr(label, 'item') and hasattr(label, 'shape') and label.shape == ():
            label = label.item()
        label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=wrapped[index])
        return label

    # Try samples/imgs pattern (returns tuple of path, label)
    if hasattr(wrapped, "samples"):
        _, label = wrapped.samples[index]
        label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=wrapped[index])
        return label

    if hasattr(wrapped, "imgs"):
        _, label = wrapped.imgs[index]
        label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=wrapped[index])
        return label

    return None


def load_raw_image(dataset, index) -> Image.Image:
    """Load raw image from dataset at given index."""

    def to_uint8(np_img: np.ndarray) -> np.ndarray:
        """Convert an array to uint8 safely for PIL.
        - Floats in [0,1] -> scale by 255
        - Values outside [0,255] -> clip
        - Cast to uint8
        """
        if not isinstance(np_img, np.ndarray):
            np_img = np.array(np_img)

        if np_img.dtype == np.uint8:
            return np_img

        if np.issubdtype(np_img.dtype, np.floating):
            min_v = float(np.nanmin(np_img)) if np_img.size else 0.0
            max_v = float(np.nanmax(np_img)) if np_img.size else 1.0
            if max_v <= 128.0:
                np_img = (np_img - min_v) / (max_v - min_v) * 255.0
        # Clip to valid byte range then cast
        np_img = np.clip(np_img, 0, 255)
        return np_img.astype(np.uint8)

    # Get dataset wrapper if exists - RECURSIVE UNWRAP
    wrapped = dataset
    while hasattr(wrapped, "wrapped_dataset") or (hasattr(wrapped, "dataset") and hasattr(wrapped, "indices")):
        if hasattr(wrapped, "wrapped_dataset"):
            wrapped = wrapped.wrapped_dataset
        elif hasattr(wrapped, "dataset") and hasattr(wrapped, "indices"):
            # Handle torch.utils.data.Subset (or similar subsets)
            wrapped = wrapped.dataset
    
    # Check for Lidar Config (attached in train script)
    # If not present, default to empty dict (will use defaults in render_lidar)
    lidar_config = getattr(wrapped, "viz_config", {})

    # Debug logging
    # logger.info(f"load_raw_image index={index}, type(wrapped)={type(wrapped)}")
    
    if hasattr(wrapped, "images") and isinstance(wrapped.images, (list, tuple, np.ndarray)):
        try:
            img_path = wrapped.images[index]
        except IndexError:
             logger.error(f"Index {index} out of bounds for images list of len {len(wrapped.images)}")
             return None
             
        if isinstance(img_path, str) and img_path.lower().endswith(('.bin', '.npy', '.pcd')):
             points = load_point_cloud_data(img_path)
             if points is not None:
                 # Dispatch to main renderer with config (and file path for labels)
                 return render_lidar(points, lidar_config, file_path=img_path)
             else:
                 return Image.new('RGB', (800, 600), color=(50, 50, 50))

        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, "files") and isinstance(wrapped.files, list):
        img_path = wrapped.files[index]
        
        if isinstance(img_path, str) and img_path.lower().endswith(('.off', '.bin', '.pcd', '.ply', '.npy')):
             points = load_point_cloud_data(img_path)
             if points is not None:
                 # Auto-normalize ModelNet .off for visualization
                 if img_path.lower().endswith('.off') and points.size > 0:
                     points = points - np.mean(points, axis=0) # Center
                     dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                     if dist > 0: points = points / dist
                     
                 return render_lidar(points, lidar_config, file_path=img_path)
        
        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, "samples") or hasattr(wrapped, "imgs"):
        # Prioritize file-on-disk for standard ImageFolder/DatasetFolder patterns
        if hasattr(wrapped, "samples"):
            img_path, _ = wrapped.samples[index]
        else:
            img_path, _ = wrapped.imgs[index]
        
        # Check if it's a point cloud file
        if isinstance(img_path, str) and img_path.lower().endswith(('.off', '.bin', '.pcd', '.ply', '.npy')):
             # Load point cloud data
             points = load_point_cloud_data(img_path)
            
             if points is not None:
                 # Auto-normalize ModelNet .off for visualization
                 if img_path.lower().endswith('.off') and points.size > 0:
                     points = points - np.mean(points, axis=0) # Center
                     dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                     if dist > 0: points = points / dist

                 return render_lidar(points, lidar_config, file_path=img_path)
             else:
                 return Image.new('RGB', (800, 600), color=(50, 50, 50))

        img = Image.open(img_path)
        return img.convert("L") if img.mode in ["1", "L", "I;16", "I"] else img.convert("RGB")
    elif hasattr(wrapped, '__getitem__') or hasattr(wrapped, "data") or hasattr(wrapped, "dataset"):
        data_entry = wrapped[index]
        
        # DEBUG: Log raw data entry
        # print(f"DEBUG: load_raw_image[{index}] raw type: {type(data_entry)}")
        
        # Helper to find point cloud in nested structure
        def find_point_cloud(item):
            # print(f"DEBUG: checking item type={type(item)}")
            if _is_point_cloud(item):
                # print(f"DEBUG: _is_point_cloud passed! shape={getattr(item, 'shape', 'unknown')}")
                return item
            
            if isinstance(item, (list, tuple)):
                for sub in item:
                    res = find_point_cloud(sub)
                    if res is not None:
                        return res
            elif isinstance(item, dict):
                for v in item.values():
                    res = find_point_cloud(v)
                    if res is not None:
                        return res
            return None

        # Try to find point cloud in the entry
        pc_candidate = find_point_cloud(data_entry)
        
        if pc_candidate is not None:
            data_entry = pc_candidate
            # print(f"DEBUG: Point cloud found via recursion.")
        elif isinstance(data_entry, (list, tuple)):
            # Fallback to first element if no point cloud found but it's a tuple
            # print(f"DEBUG: No point cloud found, falling back to index 0.")
            data_entry = data_entry[0]
        
        # Check for Point Cloud auto-detection
        if _is_point_cloud(data_entry):
            points = _to_numpy_safe(data_entry)
            if points is not None:
                # Use default lidar config if none attached to dataset
                # print(f"DEBUG: Rendering LiDAR from points shape={points.shape}")
                res = render_lidar(points, lidar_config)
                if res is None:
                     print(f"DEBUG: render_lidar returned None!")
                return res
        else:
             pass
             # print(f"DEBUG: _is_point_cloud failed on final data_entry: type={type(data_entry)}")
             # if hasattr(data_entry, 'shape'):
             #    print(f"DEBUG: shape={data_entry.shape}")

        np_img = data_entry
        if hasattr(np_img, 'numpy'):
            np_img = np_img.numpy()
        if np_img.ndim == 2:
            np_img = to_uint8(np_img)
            return Image.fromarray(np_img, mode="L")
        elif np_img.ndim == 3:
            # Convert from channel-first (C, H, W) to channel-last (H, W, C) for PIL
            if np_img.shape[0] in [1, 3, 4] and np_img.shape[0] != np_img.shape[-1]:
                np_img = np.transpose(np_img, (1, 2, 0))
            np_img = to_uint8(np_img)
            # Choose mode based on channels
            if np_img.shape[-1] == 1:
                return Image.fromarray(np_img[..., 0], mode="L")
            if np_img.shape[-1] == 3:
                return Image.fromarray(np_img, mode="RGB")
            if np_img.shape[-1] == 4:
                return Image.fromarray(np_img, mode="RGBA")
            raise ValueError(f"Unsupported channel count: {np_img.shape[-1]}")
        else:
            raise ValueError(f"Unsupported image shape: {np_img.shape}")
    else:
        raise ValueError("Dataset type not supported for raw image extraction.")
