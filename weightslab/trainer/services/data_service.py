import io
from typing import List
import time
import logging
import os
import traceback
import threading
from hashlib import md5
import json

import numpy as np
import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2

from PIL import Image
from tqdm import tqdm
from typing import List
from typing import List
from pathlib import Path
from datetime import datetime
from concurrent import futures

from weightslab.data.sample_stats import SampleStatsEx
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.components.global_monitoring import pause_controller
from weightslab.trainer.services.service_utils import load_raw_image, load_label
from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.backend.ledgers import get_dataloaders, get_dataframe, list_signals


# Get global logger
logger = logging.getLogger(__name__)


def create_data_stat(name, stat_type, shape=None, value=None, value_string="", thumbnail=b""):
    """Helper to create DataStat with all fields properly initialized.

    Args:
        name: Stat name
        stat_type: Type string (scalar, array, string, etc)
        shape: List of shape dimensions
        value: List of float values
        value_string: String value
        thumbnail: Bytes object for thumbnail (default empty bytes)

    Returns:
        pb2.DataStat: Properly initialized DataStat
    """
    return pb2.DataStat(
        name=name,
        type=stat_type,
        shape=shape or [],
        value=value or [],
        value_string=value_string,
        thumbnail=thumbnail
    )


def generate_thumbnail(pil_image, max_size=(128, 128), quality=85):
    """Generate a JPEG thumbnail from a PIL image.

    Args:
        pil_image: PIL Image object
        max_size: Max dimensions (width, height) for thumbnail
        quality: JPEG quality (1-95)

    Returns:
        bytes: JPEG thumbnail as bytes
    """
    try:
        # Create a copy to avoid modifying original
        thumb = pil_image.copy()

        # Use LANCZOS for high-quality downsampling
        thumb.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed (JPEG doesn't support RGBA)
        if thumb.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', thumb.size, (255, 255, 255))
            if thumb.mode == 'P':
                thumb = thumb.convert('RGBA')
            if thumb.mode in ('RGBA', 'LA'):
                background.paste(thumb, mask=thumb.split()[-1])  # Use alpha channel as mask
                thumb = background
        elif thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')

        # Save as JPEG to buffer
        buffer = io.BytesIO()
        thumb.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Failed to generate thumbnail: {e}")
        return b""


class DataService:

    """
    Data service helpers + RPCs (for weights_studio UI).

    Images are sent over gRPC as bytes (JPEG) for simplicity and correctness.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._lock = threading.Lock()
        self._df_manager = get_dataframe()

        # init references to the context components
        self._ctx.ensure_components()

        self._root_log_dir = self._resolve_root_log_dir()
        self._h5_path = self._resolve_h5_path()
        self._stats_store = H5DataFrameStore(self._h5_path) if self._h5_path else None

        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()
        self._agent = DataManipulationAgent(self)

        self._last_internals_update_time = 0.0

        # Shared thread pool for data processing (avoid thread explosion)
        # Size: min(CPU cores * 2, 16) to balance concurrency without excessive threading
        self._data_executor = futures.ThreadPoolExecutor(
            thread_name_prefix="WL-DataProcessing",
            max_workers=8
        )

        self._is_filtered = False  # Track if the current view is filtered/modified by user

        # Always ensure natural sort stats are computed on startup
        # This occurs before the service is fully available to the UI.
        self._compute_natural_sort_stats()
        
        # Automatically compute registered custom signals
        self._compute_custom_signals()

        logger.info("DataService initialized.")

    def _get_loader_by_origin(self, origin: str):
        """Dynamically retrieve loader for a specific origin (on-demand).

        Avoids maintaining persistent _loaders state; fetches only when needed.
        """
        try:
            loader_names = get_dataloaders()
            for loader_name in loader_names:
                loader = self._ctx.components.get(loader_name)
                if loader is None:
                    continue
                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    if tracked_ds._dataset_split == origin:
                        return loader
                # Fallback: match by loader name
                elif origin in loader_name:
                    return loader
        except Exception as e:
            logger.debug(f"[_get_loader_by_origin] Failed to retrieve loader for origin={origin}: {e}")
        return None

    def _initialize_data_service(self):
        """Recreate the in-memory dataframe view from the shared H5 store."""
        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()

    def _slowUpdateInternals(self, force=False):
        """Force refresh of internal dataframe view."""
        # Simple alias for now if not defined
        if time.time() - self._last_internals_update_time > 5.0 or force:
             self._all_datasets_df = self._pull_into_all_data_view_df()
             self._load_existing_tags()
             self._last_internals_update_time = time.time()

    def _resolve_root_log_dir(self) -> Path:
        """Resolve root log directory from hyperparams/env, fallback to ./logs."""
        root = None
        try:
            hp = self._ctx.components.get("hyperparams")
            if hp is not None and hasattr(hp, "get"):
                hp_dict = hp.get() if not isinstance(hp, dict) else hp
                if isinstance(hp_dict, dict):
                    root = (
                        hp_dict.get("root_log_dir")
                        or hp_dict.get("root_directory")
                        or hp_dict.get("root")
                    )
        except Exception:
            root = None

        root = root or os.getenv("WEIGHTSLAB_ROOT_LOG_DIR")
        if root is None:
            root = Path("logs").absolute()
        return Path(root)

    def _resolve_h5_path(self) -> Path | None:
        """Return the H5 path used by tracked datasets and the streaming view."""
        if self._root_log_dir is None:
            return None
        data_dir = Path(self._root_log_dir) / "checkpoints" / "data"
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return data_dir / "data_with_ops.h5"

    def is_agent_available(self) -> bool:
        """
        Check if the agent (Ollama) is available for natural language queries.

        Returns:
            bool: True if agent is available, False otherwise
        """
        if self._agent is None:
            return False
        try:
            return self._agent.is_ollama_available()
        except Exception as e:
            logger.debug(f"Error checking agent availability: {e}")
            return False

    def _is_training_active(self) -> bool:
        """Return True if training is currently running (not paused)."""
        try:
            hp = self._ctx.components.get("hyperparams")
            if hp is not None and hasattr(hp, "get"):
                hp_dict = hp.get() if not isinstance(hp, dict) else hp
                if isinstance(hp_dict, dict):
                    flag = hp_dict.get("is_training")
                    if flag is not None:
                        return bool(flag)
        except Exception:
            pass
        # Fall back to pause controller state if hyperparams missing
        try:
            return not pause_controller.is_paused()
        except Exception:
            return True

    def _pull_into_all_data_view_df(self):
            """Stream stats from the global in-memory dataframe (ledger manager).

            Uses the shared dataframe manager instead of the H5 store and avoids
            blocking on IO. Falls back to last snapshot if retrieval fails.
            """
            try:
                # Load dataframe from the shared dataframe manager with arrays autoloaded from h5 storage
                df = self._df_manager.get_combined_df() if self._df_manager is not None else pd.DataFrame()
                if df.empty:
                    return df

                # Ensure sample_id is in index for consistency
                df = df.reset_index().set_index(["sample_id"])

                return df
            except Exception:
                logger.debug("[DataService] Falling back to cached snapshot of dataframe")
                return self._all_datasets_df if self._all_datasets_df is not None else pd.DataFrame()

    def _get_origin_filter(self, request):
        """Extract requested origins if present on request (backward compatible)."""
        origins = None
        for attr in ("sample_origins", "origins", "origin"):
            try:
                val = getattr(request, attr, None)
                if val:
                    # Normalize to list
                    if isinstance(val, str):
                        origins = [val] if val.strip() else []  # Filter empty strings
                    else:
                        # Filter out empty strings from list
                        origins = [o for o in list(val) if o and str(o).strip()]
                    break
            except Exception:
                continue
        return origins or []

    def _filter_df_by_origin(self, df: pd.DataFrame, origins: list[str]) -> pd.DataFrame:
        if df is None or df.empty or not origins:
            return df
        try:
            if isinstance(df.index, pd.MultiIndex):
                if "origin" in df.index.names:
                    mask = df.index.get_level_values("origin").isin(origins)
                    return df.loc[mask]
            if "origin" in df.columns:
                return df[df["origin"].isin(origins)]
        except Exception as e:
            logger.debug(f"[_filter_df_by_origin] failed to filter by origins {origins}: {e}")
        return df

    def _load_existing_tags(self):
        """Ensure tags and natural sort columns are present on the streamed dataframe."""
        if self._all_datasets_df is None or self._all_datasets_df.empty:
            return

        if SampleStatsEx.TAGS.value not in self._all_datasets_df.columns:
            try:
                self._all_datasets_df[SampleStatsEx.TAGS.value] = ""
            except Exception:
                pass
        
        # Ensure natural_sort_score exists (init with NaN if missing)
        if "natural_sort_score" not in self._all_datasets_df.columns:
            try:
                self._all_datasets_df["natural_sort_score"] = np.nan
            except Exception:
                pass

    def _get_dataset(self, origin: str):
        loader = self._get_loader_by_origin(origin)
        if loader is not None:
            dataset = getattr(loader, "tracked_dataset", None)
            return dataset

    def _is_nan_value(self, value):
        """Check if a value is NaN, handling both scalars and arrays."""
        try:
            # For scalars, np.isnan works directly
            if np.isscalar(value):
                return np.isnan(value)
            return False
        except (TypeError, ValueError):
            # For arrays/tensors, check if it's a single NaN value
            try:
                value_arr = np.asarray(value)
                return value_arr.size == 1 and np.isnan(value_arr.item())
            except (TypeError, ValueError):
                return False


    def _compute_natural_sort_stats(self):
        """
        Compute natural sort statistics (brightness, hue, saturation, entropy) for all samples
        and update the dataframe.
        
        Includes a 'natural_sort_score' for sorting, configured by weights.
        """
        # --- CONFIGURATION: Define Natural Sort Cues & Weights ---
        # Weights should sum to 1.0 ideally, but relative magnitude matters most.
        # Strategies:
        # 1. "Day vs Night" focus: Brightness=0.8, Entropy=0.2
        # 2. "Complexity" focus: Brightness=0.2, Entropy=0.8
        # 3. "Balanced": Brightness=0.5, Entropy=0.5
        # 4. "Grouped" (Pseudo-primary key): Brightness=5.0, Entropy=1.0 (Forces clustering by light)
        
        SORT_WEIGHTS = {
            "brightness": 0.7,  # Primary cue: Lighting conditions
            "entropy": 0.3,     # Secondary cue: Texture/Scene complexity
            "hue": 0.0          # Optional: Color tint
        }
        
        logger.info(f"[DataService] Starting natural sort stats computation with weights: {SORT_WEIGHTS}")

        try:
            import cv2
        except ImportError:
            logger.warning("[DataService] OpenCV not found. Skipping natural sort computation.")
            return "OpenCV not installed"

        if self._all_datasets_df is None or self._all_datasets_df.empty:
             return "No data to process"

        # Helper: Calculate Shannon Entropy (Complexity)
        def calc_entropy(img_gray):
            try:
                # Calculate histogram (256 bins for 8-bit)
                hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
                # Normalize histogram to get probabilities
                p = hist.ravel() / hist.sum()
                # Filter out zero probabilities to avoid log(0)
                p = p[p > 0]
                # Shannon Entropy in bits
                return -np.sum(p * np.log2(p))
            except Exception:
                return 0.0

        # helper to process a single image
        def process_sample(args):
            idx, row = args
            try:
                origin = row.get(SampleStatsEx.ORIGIN.value, 'unknown')
                
                # Sample ID is likely the index if not in columns
                if SampleStatsEx.SAMPLE_ID.value in row:
                     sample_id = int(row[SampleStatsEx.SAMPLE_ID.value])
                else:
                     sample_id = int(idx)

                # Skip if already computed (optimization for blocking startup)
                if "natural_sort_score" in row and not pd.isna(row["natural_sort_score"]):
                    return None

                dataset = self._get_dataset(origin)
                if not dataset:
                    logger.warning(f"[Natural Sort] Dataset not found for origin: {origin}")
                    return None

                # Use dataset index, not dataframe index
                ds_idx = dataset.get_index_from_sample_id(sample_id)
                if ds_idx is None:
                     logger.warning(f"[Natural Sort] Could not map sample_id {sample_id} to dataset index")
                     return None

                pil_img = load_raw_image(dataset, ds_idx)

                if pil_img is None:
                    # silenced to avoid spam if images are genuinely missing for many
                    # logger.warning(f"[Natural Sort] Failed to load image for sample {sample_id} at index {ds_idx}")
                    return None

                # Convert to numpy (RGB)
                img_np = np.array(pil_img)

                # Brightness (mean pixel intensity)
                # If RGB, convert to Gray, else just mean
                if img_np.ndim == 3:
                    # OpenCV expects BGR usually, but PIL gives RGB.
                    # cvtColor RGB2GRAY is correct.
                    try:
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    except Exception:
                         gray = img_np
                else:
                    gray = img_np

                brightness = np.mean(gray)
                entropy = calc_entropy(gray)

                # HSV Stats
                if img_np.ndim == 3:
                    try:
                        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                        hue = np.mean(hsv[:, :, 0])
                        saturation = np.mean(hsv[:, :, 1])
                    except Exception:
                        hue = 0.0
                        saturation = 0.0
                else:
                    hue = 0.0
                    saturation = 0.0

                # --- Compute Composite Score ---
                # Normalize robustly to 0-1 range for combination
                # Brightness: 0-255 typical for 8-bit
                norm_brightness = min(max(brightness / 255.0, 0.0), 1.0)
                
                # Entropy: 0-8 bits typical for 8-bit image
                norm_entropy = min(max(entropy / 8.0, 0.0), 1.0)
                
                # Hue: 0-179 in OpenCV
                norm_hue = min(max(hue / 179.0, 0.0), 1.0)

                score = (
                    SORT_WEIGHTS.get("brightness", 0) * norm_brightness +
                    SORT_WEIGHTS.get("entropy", 0) * norm_entropy +
                    SORT_WEIGHTS.get("hue", 0) * norm_hue
                )

                return {
                    "sample_id": sample_id,
                    "origin": origin,
                    "brightness": brightness,
                    "entropy": entropy,
                    "hue": hue,
                    "saturation": saturation,
                    "natural_sort_score": score
                }
            except Exception as e:
                # Log only the first few errors globally (using a simple counter if we could, but here we can't share state easily)
                # Fallback: Just log warning. To avoid spam, we can check if it's one of the first few tasks?
                # No, standard logger is fine, just use debug for spammy ones or INFO for specific tracking.
                logger.warning(f"Failed to compute stats for sample {idx} (id={row.get(SampleStatsEx.SAMPLE_ID.value, 'unknown')}): {e}")
                return None

        # Prepare tasks
        tasks = []
        for idx, row in self._all_datasets_df.iterrows():
             tasks.append((idx, row))

        logger.info(f"[DataService] Computing sort stats for {len(tasks)} samples...")

        # Run in parallel
        # Use a smaller executor if needed, but self._data_executor is shared
        results = self._data_executor.map(process_sample, tasks)

        # Collect results
        updates_by_origin = {}
        processed_count = 0
        failure_count = 0

        # Wrap results with tqdm for progress bar
        for res in tqdm(results, total=len(tasks), desc="Computing Natural Sort Stats", unit="samples"):
            if res:
                processed_count += 1
                origin = res["origin"]
                updates_by_origin.setdefault(origin, []).append(res)
            else:
                failure_count += 1
                if failure_count <= 5:
                     # We can't easily get the specific error detail here because it was swallowed in process_sample
                     # But we rely on process_sample to log it.
                     pass

        if failure_count > 0:
             logger.warning(f"[DataService] Failed to compute stats for {failure_count} samples. Check logs for details.")

        # Upsert to storage
        if self._df_manager:
            for origin, rows in updates_by_origin.items():
                df_update = pd.DataFrame(rows).set_index("sample_id")
                # Drop origin column from update if it's in the index or redundant
                if "origin" in df_update.columns:
                     del df_update["origin"]

                self._df_manager.upsert_df(df_update, origin=origin, force_flush=True)

        # Force refresh internal view
        self._slowUpdateInternals(force=True)

        logger.info(f"[DataService] Completed stats computation for {processed_count} samples")
        print(f"\n\nNatural sort computation finished for {processed_count} samples\n\n")
        return f"Computed stats for {processed_count} samples"
        
    def _compute_custom_signals(self):
        """
        Discover and compute registered custom signals for all active dataloaders.
        This allows scripts to simply register @wl.signal and have them computed automatically.
        """
        # Local import to avoid circular dependency
        import weightslab.src as wl_src
        
        # Get all registered signal names
        signals = list_signals()
        if not signals:
            return
            
        # Get all active dataloaders
        # use get_dataloaders without args to get all registered ones
        loaders = get_dataloaders(None) 
        
        logger.info(f"[DataService] Checking custom signals {signals} for {len(loaders)} loaders...")
        
        for loader_name, loader in loaders.items():
            if not loader or not hasattr(loader, "dataset"):
                continue
                
            try:
                # We need to determine the origin/split name for the compute_signals function
                # Try to inspect the loader/dataset
                origin = loader_name # Fallback
                ds = getattr(loader, "tracked_dataset", None)
                if ds and hasattr(ds, "_dataset_split"):
                    origin = ds._dataset_split
                elif hasattr(loader, "dataset") and hasattr(loader.dataset, "split"):
                    origin = loader.dataset.split
                
                # Run computation
                logger.info(f"[DataService] Computing signals {signals} for loader '{loader_name}' (origin={origin})")
                wl_src.compute_signals(loader, origin=origin, signals=signals)
                
            except Exception as e:
                logger.error(f"[DataService] Failed to compute signals for loader '{loader_name}': {e}")
                
        # Force view update
        self._slowUpdateInternals(force=True)

    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args
        start_total = time.time()
        try:
            origin = row.get(SampleStatsEx.ORIGIN.value, 'unknown')
            sample_id = int(row.get(SampleStatsEx.SAMPLE_ID.value, 0))

            # ===== Step 0: Initialize Variables======
            raw_shape, data_stats = [], []
            raw_data_bytes = b""

            # ====== Step 1: Load dataset ======
            dataset = self._get_dataset(origin)

            # ====== Step 2: Determine task type ======
            # ====== Step 2: Determine task type ======
            label = row.get(SampleStatsEx.TARGET.value)
            
            # Force reload label if dataset has viz_config (fixes issue where H5 has stringified dict)
            force_reload = False
            if dataset:
                curr_ds = dataset
                # Unwrap to find config
                while True:
                    if hasattr(curr_ds, "viz_config"):
                        force_reload = True
                        break
                    if hasattr(curr_ds, "dataset"):
                        curr_ds = curr_ds.dataset
                    elif hasattr(curr_ds, "wrapped_dataset"):
                        curr_ds = curr_ds.wrapped_dataset
                    else:
                        break
            
            if (label is None or (isinstance(label, list) and label == []) or force_reload) and dataset:
                try:
                    label = load_label(dataset, sample_id)
                except Exception as e:
                    logger.warning(f"Failed to load label for sample {sample_id}: {e}")
                    label = None

            # Scalar / single element -> treat as classification
            label_ndim = label.ndim if hasattr(label, 'ndim') else len(getattr(label, 'shape', []))
            label_size = label.size if hasattr(label, 'size') else (1 if label_ndim == 0 else None)

            # Enccode task type based on label shape heuristics
            # TODO (GP): more robust task type inference
            task_type = 'segmentation'  # _infer_task_type_from_label(label, default='Segmentation')
            if label_ndim == 0 or label_size == 1:
                task_type = "classification"
            elif label_ndim >= 2:
                # Cache shape to avoid multiple H5 accesses
                shape = label.shape
                if shape[-2] > 28 or shape[-1] > 28:
                    task_type = 'segmentation'
                else:
                    task_type = "unknown"

            # ====== Step 5a: Process stats ======
            stats_to_retrieve = list(request.stats_to_retrieve)
            if not stats_to_retrieve:
                # Use set for O(1) lookup instead of O(n) list lookup
                exclude_cols = {
                    SampleStatsEx.SAMPLE_ID.value,
                    SampleStatsEx.ORIGIN.value,
                    SampleStatsEx.TARGET.value,
                    SampleStatsEx.PREDICTION.value,
                    # SampleStatsEx.PREDICTION_RAW.value,
                    SampleStatsEx.TASK_TYPE.value,
                }
                stats_to_retrieve = [col for col in df_columns if col not in exclude_cols]

            # Optimized bulk processing of stats
            for stat_name in stats_to_retrieve:
                value = row.get(stat_name)
                # Skip prediction raw array
                if (isinstance(value, np.ndarray) and value.ndim > 1) or (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0):
                    continue
                if isinstance(value, float):
                    value = round(value, 7)
                if isinstance(value, bool):
                    value = int(value)
                data_stats.append(
                    create_data_stat(stat_name, "string", shape=[1], value_string=str(value)[:512], thumbnail=b"")
                )

            # ====== Step 6: Add origin and task_type stats ======
            data_stats.append(
                create_data_stat(
                    "origin", 'string', shape=[1], value_string=origin, thumbnail=b""
                )
            )
            data_stats.append(
                create_data_stat(
                    "task_type", 'string', shape=[1], value_string=str(task_type), thumbnail=b""
                )
            )

            # ====== Step 7: Process labels ======
            if task_type != "classification":
                if label is None:
                    label_arr = np.asarray(row.get(SampleStatsEx.TARGET.value))
                else:
                    label_arr = label

                # Treat label as segmentation mask -> array stat
                data_stats.append(
                    create_data_stat(
                        name='label',
                        stat_type='array',
                        shape=list(label_arr.shape),
                        value=label_arr.astype(float).ravel().tolist(),
                        thumbnail=b""
                    )
                )

                # Prefer row attribute if available, fallback to dataset, then model
                num_classes = (
                    row.get('num_classes')
                    or (getattr(dataset, "num_classes", None) if dataset else None)
                    or getattr(self._ctx.components.get("model"), "num_classes", None)
                )
                if num_classes is None:
                    # Fallback: infer from this label
                    if label_arr.size > 0:
                        max_id = int(label_arr.max())
                        num_classes = max(1, max_id + 1)
                    else:
                        num_classes = 1

                data_stats.append(
                    create_data_stat(
                        name="num_classes",
                        stat_type="scalar",
                        shape=[1],
                        value=[float(num_classes)],
                        thumbnail=b""
                    )
                )

                # Append class_names if available
                model_comp = self._ctx.components.get("model")
                class_names = (
                    row.get('class_names')
                    or (getattr(dataset, "class_names", None) if dataset else None)
                    or getattr(model_comp, "class_names", None)
                )

                if class_names:
                    val_str = ""
                    try:
                        if isinstance(class_names, (list, tuple)):
                            val_str = json.dumps(list(class_names))
                        elif isinstance(class_names, dict):
                            # Convert int keys to string for JSON if all keys are ints
                            if all(isinstance(k, int) for k in class_names.keys()):
                                class_names = {str(k): v for k, v in class_names.items()}
                            val_str = json.dumps(class_names)
                        else:
                            val_str = str(class_names)
                    except Exception:
                        val_str = str(class_names)

                    data_stats.append(
                        create_data_stat(
                            name="class_names",
                            stat_type="string",
                            shape=[1],
                            value_string=val_str,
                            thumbnail=b""
                        )
                    )

            elif label is not None:
                # Classification / other scalar-like labels
                
                # Handle dictionary labels (e.g. detection targets)
                if isinstance(label, dict):
                    # Check for Lidar Config to render BEV mask
                    # Need to unwrap dataset to find config
                    curr_ds = dataset
                    lidar_config = {}
                    while True:
                         if hasattr(curr_ds, "viz_config"):
                             lidar_config = curr_ds.viz_config
                             break
                         if hasattr(curr_ds, "dataset"):
                             curr_ds = curr_ds.dataset
                         elif hasattr(curr_ds, "wrapped_dataset"):
                             curr_ds = curr_ds.wrapped_dataset
                         else:
                             break
                    
                    if lidar_config and 'boxes' in label:
                        logger.info(f"[DataService] Rendering BEV mask. Boxes: {len(label['boxes'])}")
                        boxes = _to_numpy_safe(label['boxes'])
                        labels_ = _to_numpy_safe(label['labels'])
                        
                        id_to_cls = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
                        fmt_labels = []
                        if boxes is not None and labels_ is not None:
                            for i in range(len(boxes)):
                                c_id = int(labels_[i])
                                cls_name = id_to_cls.get(c_id, "Car")
                                fmt_labels.append({
                                    'cls': cls_name,
                                    'box': boxes[i]
                                })
                        
                        bev_conf = lidar_config.get("bev", {})
                        mask = render_bev_mask(
                            fmt_labels,
                            res=bev_conf.get("resolution", 0.1),
                            size=bev_conf.get("image_size", 800),
                            cx=bev_conf.get("center_x", 400),
                            cy=bev_conf.get("center_y", 400)
                        )
                        data_stats.append(
                            create_data_stat(
                                name='label_mask', # Use label_mask convention for segmentation overlay
                                stat_type='array',
                                shape=list(mask.shape),
                                value=mask.astype(float).ravel().tolist(),
                                thumbnail=b""
                            )
                        )
                    else:
                        data_stats.append(
                            create_data_stat(
                                name='label',
                                stat_type='string',
                                shape=[1],
                                value_string=str(label), # Simplified visualization
                                thumbnail=b""
                            )
                        )
                else:
                    # Check if label is NaN (handle both scalars and arrays)
                    if self._is_nan_value(label):
                        pass  # Skip NaN labels

                    # Handle scalar labels
                    try:
                        data_stats.append(
                            create_data_stat(
                                name='label',
                                stat_type='scalar',
                                shape=[1],
                                value=[float(label)],
                                thumbnail=b""
                            )
                        )
                    except (ValueError, TypeError):
                        # Fallback for non-scalar labels in non-segmentation tasks
                        try:
                            label_arr = np.asanyarray(label)
                            data_stats.append(
                                create_data_stat(
                                    name='label',
                                    stat_type='array',
                                    shape=list(label_arr.shape),
                                    value=label_arr.astype(float).ravel().tolist(),
                                    thumbnail=b""
                                )
                            )
                        except (ValueError, TypeError) as e:
                            # Use logger.debug to avoid spamming console
                            logger.debug(f"Could not convert label to array: {label}, error: {e}")

            # ====== Step 8: Process predictions ======
            pred = row.get(SampleStatsEx.PREDICTION.value)
            if task_type != "classification":
                # Handle Dict Prediction (Detection -> BEV Mask)
                if isinstance(pred, dict) and lidar_config and 'boxes' in pred:
                     # Render Prediction Mask similar to Ground Truth
                     boxes = _to_numpy_safe(pred['boxes'])
                     labels_ = _to_numpy_safe(pred['labels'])
                     
                     # Map IDs
                     id_to_cls = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
                     fmt_labels = []
                     if boxes is not None and labels_ is not None:
                         for i in range(len(boxes)):
                             c_id = int(labels_[i])
                             cls_name = id_to_cls.get(c_id, "Car")
                             fmt_labels.append({
                                 'cls': cls_name,
                                 'box': boxes[i]
                             })
                             
                     bev_conf = lidar_config.get("bev", {})
                     mask = render_bev_mask(
                         fmt_labels,
                         res=bev_conf.get("resolution", 0.1),
                         size=bev_conf.get("image_size", 800),
                         cx=bev_conf.get("center_x", 400),
                         cy=bev_conf.get("center_y", 400)
                     )
                     data_stats.append(
                         create_data_stat(
                             name='pred_mask',
                             stat_type='array',
                             shape=list(mask.shape),
                             value=mask.astype(float).ravel().tolist(),
                             thumbnail=b""
                         )
                     )
                elif pred is not None:
                    try:
                        pred_arr = np.asarray(pred)
                        # Add predicted mask stat
                        data_stats.append(
                            create_data_stat(
                                name='pred_mask',
                                stat_type='array',
                                shape=list(pred_arr.shape),
                                value=pred_arr.astype(float).ravel().tolist(),
                                thumbnail=b""
                            )
                        )
                    except Exception:
                         pass
            else:
                # Classification: get prediction from row or dataset
                if pred is None:
                    pass  # No prediction to process

                else:
                    # Handle scalar predictions (int, float, or unwrapped from H5)
                    try:
                        pred_val = float(np.asanyarray(pred).item()) if hasattr(pred, '__iter__') else float(pred)
                        data_stats.append(
                            create_data_stat(
                                name='pred',
                                stat_type='scalar',
                                shape=[1],
                                value=[pred_val],
                                thumbnail=b""
                            )
                        )
                    except (ValueError, TypeError):
                        # Fallback for non-scalar predictions in non-segmentation tasks
                        try:
                            pred_ = np.array(pred)
                            data_stats.append(
                                create_data_stat(
                                    name='pred',
                                    stat_type='array',
                                    shape=list(pred_.shape),
                                    value=pred_.astype(float).ravel().tolist(),
                                    thumbnail=b""
                                )
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not convert prediction to array: {pred}, error: {e}")

            # ====== Step 9: Generate raw data bytes and thumbnail ======
            if request.include_raw_data:
                raw_img = load_raw_image(dataset, dataset.get_index_from_sample_id(sample_id))
                original_size = raw_img.size

                # Handle resize request
                target_width = original_size[0]
                target_height = original_size[1]
                aspect_ratio = original_size[0] / original_size[1] if original_size[1] > 0 else 1.0

                if request.resize_width < 0 and request.resize_height < 0:
                    percent = abs(request.resize_width) / 100.0
                    target_width = int(original_size[0] * percent)
                    target_height = int(original_size[1] * percent)

                elif request.resize_width > 0 and request.resize_height > 0:
                    # Bounding box resize preserving aspect ratio.
                    # Fit the image inside (resize_width x resize_height).
                    w_limit, h_limit = request.resize_width, request.resize_height
                    if w_limit / h_limit > aspect_ratio:
                        # Box is wider than image relative to height, height is the constraint
                        target_height = h_limit
                        target_width = int(target_height * aspect_ratio)
                    else:
                        # Image is wider than box relative to height, width is the constraint
                        target_width = w_limit
                        target_height = int(target_width / aspect_ratio)

                elif request.resize_width == 0 and request.resize_height == 0:
                     # Default to 360p (height=360) maintaining aspect ratio if no resize requested
                     target_height = 360
                     target_width = int(target_height * aspect_ratio)

                # Ensure dimensions are at least 1x1
                target_width = max(1, target_width)
                target_height = max(1, target_height)

                # Resize image if needed
                if target_width != original_size[0] or target_height != original_size[1]:
                    raw_img = raw_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                # Generate raw data bytes
                raw_buf = io.BytesIO()
                raw_img.save(raw_buf, format='JPEG')
                raw_data_bytes = raw_buf.getvalue()
                raw_shape = [target_height, target_width, len(raw_img.getbands())]

                data_stats.append(
                    create_data_stat(
                        name='raw_data',
                        stat_type='bytes',
                        value=raw_data_bytes,
                        shape=raw_shape,
                    )
                )

            # ====== Step 10: Create DataRecord ======
            record = pb2.DataRecord(sample_id=sample_id, data_stats=data_stats)

            return record

        except Exception as e:
            total_time = time.time() - start_total
            logger.error(f"[Sample {row.get(SampleStatsEx.SAMPLE_ID.value, -1)}] X Error after {total_time:.3f}s: {e}", exc_info=True)
            return None

    def _get_unique_tags(self) -> List[str]:
        """Collect all unique tags currently present in the tracked datasets."""
        tags = set()
        try:
            # Extract tags from the dataframe if it exists and has a tags column
            if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                if SampleStatsEx.TAGS.value in self._all_datasets_df.columns:
                    tag_values = self._all_datasets_df[SampleStatsEx.TAGS.value].dropna()
                    for tag_val in tag_values:
                        if tag_val:
                            # Tags are stored as comma-separated strings
                            for t in str(tag_val).split(';'):
                                clean_t = t.strip()
                                if clean_t:
                                    tags.add(clean_t)
        except Exception as e:
            logger.warning(f"Error collecting unique tags: {e}")
        return sorted(list(tags))

    def _build_success_response(
        self,
        df,
        message: str,
        intent_type=pb2.INTENT_FILTER,
        analysis_result=""
    ) -> pb2.DataQueryResponse:
        """
        Centralized helper so every code path reports counts consistently.

        - number_of_all_samples: all rows in df
        - number_of_discarded_samples: rows with deny_listed == True (if column exists)
        - number_of_samples_in_the_loop: rows not deny_listed
        """
        total_count = len(df)
        discarded_count = (
            len(df[df.get("deny_listed", False) == True])  # noqa: E712
            if df is not None and "deny_listed" in df.columns
            else 0
        )
        in_loop_count = total_count - discarded_count
        unique_tags = self._get_unique_tags()

        return pb2.DataQueryResponse(
            success=True,
            message=message,
            number_of_all_samples=total_count,
            number_of_samples_in_the_loop=in_loop_count,
            number_of_discarded_samples=discarded_count,
            unique_tags=unique_tags,
            agent_intent_type=intent_type,
            analysis_result=analysis_result
        )

    def _parse_direct_query(self, query: str) -> list:
        """
        Parse a simple direct query string into operations list.
        Expected format: "col > val and col2 == val2 sortby col desc"
        This bypasses the LLM agent for deterministic filtering.
        """
        operations = []
        query = query.strip()

        logger.debug(f"[_parse_direct_query] Parsing query: {repr(query)}")

        if query.lower().startswith("sortby "):
            filter_part = None
            sort_part = query[7:].strip()
        elif " sortby " in query.lower():
            parts = query.split(" sortby ", 1)
            filter_part = parts[0].strip() if parts[0].strip() else None
            sort_part = parts[1].strip() if len(parts) > 1 else None
        else:
            filter_part = query
            sort_part = None

        logger.debug(f"[_parse_direct_query] filter_part: {repr(filter_part)}, sort_part: {repr(sort_part)}")

        # Parse filter part
        if filter_part:
            # For now, treat the entire filter as a pandas query expression
            operations.append({
                "function": "df.query",
                "params": {"expr": filter_part}
            })
        # Parse sort part
        if sort_part:
            # Check for Scope flag
            # Format: "col desc scope:global" or "col desc scope:subset"
            sort_scope = "subset" # Default to preserving current context
            view_start = 0
            view_count = 0

            if "scope:global" in sort_part:
                sort_scope = "global"
                sort_part = sort_part.replace("scope:global", "")
            elif "scope:subset" in sort_part:
                sort_scope = "subset"
                sort_part = sort_part.replace("scope:subset", "")
            elif "scope:view" in sort_part:
                sort_scope = "view"
                sort_part = sort_part.replace("scope:view", "")
                
                # Extract start/count if present
                # Split by space to find params, reassemble sort string
                tokens = sort_part.split()
                clean_tokens = []
                for t in tokens:
                    low_t = t.lower()
                    if low_t.startswith("start:"):
                        try: view_start = int(low_t.split(":")[1])
                        except: pass
                    elif low_t.startswith("count:"):
                        try: view_count = int(low_t.split(":")[1])
                        except: pass
                    else:
                        clean_tokens.append(t)
                sort_part = " ".join(clean_tokens)

                sort_part = " ".join(clean_tokens)

            # If Global scope, we map it to subset behavior (sort current filtered view)
            # We do NOT reset context anymore.
            if sort_scope == "global":
                sort_scope = "subset"

            sort_cols = []
            sort_ascs = []

            # Split by comma to support multiple columns: "tags asc, target desc"
            parts = [p.strip() for p in sort_part.split(',')]

            for p in parts:
                if not p:
                    continue

                ascending = True
                col_raw = p

                # Check for direction suffix
                lower_s = p.lower()
                if lower_s.endswith(" asc"):
                    ascending = True
                    col_raw = p[:-4].strip()
                elif lower_s.endswith(" desc"):
                    ascending = False
                    col_raw = p[:-5].strip()

                # Clean up quotes from column name
                col = col_raw.replace('`', '').strip()

                if col:
                    sort_cols.append(col)
                    sort_ascs.append(ascending)

            if sort_cols:
                logger.debug(f"[_parse_direct_query] Sort: cols={sort_cols}, asc={sort_ascs}, scope={sort_scope}")

                if sort_scope == "view":
                     operations.append({
                        "function": "df.sort_view_slice",
                        "params": {"by": sort_cols, "ascending": sort_ascs, "start": view_start, "count": view_count}
                    })
                elif len(sort_cols) == 1 and sort_cols[0].lower() == 'index':
                    # Optimization for single 'index' sort
                    operations.append({
                        "function": "df.sort_index",
                        "params": {"ascending": sort_ascs[0]}
                    })
                else:
                    # Standard sort
                    operations.append({
                        "function": "df.sort_values",
                        "params": {"by": sort_cols, "ascending": sort_ascs}
                    })
        logger.debug(f"[_parse_direct_query] Parsed into {len(operations)} operations: {operations}")
        return operations

    def _apply_agent_operation(self, df, func: str, params: dict) -> str:
        """
        Apply an agent-described operation to df in-place.
        Now supports Actions and Clarifications.
        """
        # --- 1. CLARIFICATION & SCOPE ---
        if func == "out_of_scope":
            return params.get("reason", "I cannot help with that.")

        if func == "clarify":
            # Pass the LLM's question back to the UI
            return params.get("reason", "I need more information to complete this request.")

        # --- 2. ACTIONS (New Capability) ---
        if func.startswith("action."):
            action_name = func.replace("action.", "")

            # Example: "Save Dataset" Action
            if action_name == "save_dataset":
                filename = params.get("filename", "dataset_export")
                # TODO: Implement your actual save logic here
                # e.g., self._stats_store.save_snapshot(df, filename)
                logger.info(f"Action triggered: Saving dataset as {filename}")
                return f"Action: Dataset saved as '{filename}'"

            # Example: "Plot" Action
            elif action_name == "plot_distribution":
                col = params.get("column")
                # TODO: Implement plot logic
                return f"Action: Plotted distribution for {col}"

            # Example: "Compute Natural Sort" Action
            elif action_name == "compute_natural_sort":
                msg = self._compute_natural_sort_stats()
                return f"Action: {msg}"

            return f"Action triggered: {action_name} (Not implemented)"

        # --- 3. DATAFRAME MANIPULATION ---

        # A) Agent-driven df.apply_mask (for complex filters)
        if func == "df.apply_mask":
            code = params.get("code", "")
            try:
                mask = eval(code, {"df": df, "np": np, "pd": pd})
                if isinstance(mask, (pd.Series, np.ndarray, list, pd.Index)):
                    if isinstance(mask, (list, pd.Index)) and not pd.api.types.is_bool_dtype(pd.Series(mask)):
                         kept = df.loc[mask]
                    else:
                         kept = df[mask]

                    df.drop(index=df.index.difference(kept.index), inplace=True)
                    return f"Applied mask: {code}"
                else:
                    return f"Expression returned unsupported type {type(mask).__name__}: {code}"
            except Exception as e:
                logger.error(f"Failed to apply mask {code}: {e}")
                return f"Failed to apply mask: {e}"

        # B) Agent-driven df.query
        if func == "df.query":
            expr = params.get("expr", "")
            try:
                kept = df.query(expr)
                df.drop(index=df.index.difference(kept.index), inplace=True)
                return f"Applied query: {expr}"
            except Exception as e:
                # Fallback to eval if query fails
                try:
                    mask = df.eval(expr, local_dict={"df": df, "np": np, "pd": pd})
                    kept = df[mask]
                    df.drop(index=df.index.difference(kept.index), inplace=True)
                    return f"Applied query (eval): {expr}"
                except Exception as eval_e:
                    logger.error(f"Query failed: {e} | Eval failed: {eval_e}")
                    return f"Failed to filter: {expr}"

        # C) Column Modification (Transform)
        if func == "df.modify":
            col = params.get("col")
            code = params.get("code")
            try:
                # 0. Safety Check: If target column exists, check compatibility
                if col in df.columns:
                     # Heuristic: If existing column is not numeric, but code implies math (contains +,-,*,/), warn/block
                     # This prevents accidental string concatenation (e.g. 1.0 + "tag" -> "tag1.0")
                     if not pd.api.types.is_numeric_dtype(df[col]):
                         # Reliance on try-except to catch invalid math is safer than heuristic string checking
                         # because heuristics fail on column names like 'signals//loss'
                         pass

                # 1. Evaluate the expression with safe context
                new_values = eval(code, {"df": df, "np": np, "pd": pd})

                # 2. Check for scalar vs series compatibility
                # (Pandas handles most of this, but we ensure robustness)
                if isinstance(new_values, (pd.Series, np.ndarray, list)):
                    if len(new_values) != len(df):
                        # Attempt alignment if it's a Series
                        # ... (existing code)
                        if isinstance(new_values, pd.Series):
                            # Auto-align to df index
                            df[col] = new_values
                        else:
                            return f"Error: Length mismatch. Result has {len(new_values)}, df has {len(df)}"
                    else:
                         df[col] = new_values
                else:
                    # Scalar assignment
                    df[col] = new_values

                # 3. CRITICAL: Persist to Ledger (H5/Disk)
                # We must split the updates by origin and upsert them to the manager
                if self._df_manager is not None:
                    # Create a minimal update dataframe with just the modified column
                    update_payload = df[[col]].copy()

                    # Ensure origin is available for grouping
                    if isinstance(df.index, pd.MultiIndex) and "origin" in df.index.names:
                        # Index is (origin, sample_id) - ideal for grouping
                        for origin, group in update_payload.groupby(level="origin"):
                            # Reset index to just sample_id for upsert
                            # group.index is (origin, sample_id), droplevel(0) gives sample_id
                            clean_group = group.droplevel("origin")
                            clean_group[SampleStatsEx.ORIGIN.value] = origin
                            self._df_manager.upsert_df(clean_group, origin=origin, force_flush=True)

                    elif SampleStatsEx.ORIGIN.value in df.columns:
                        # Origin is a column
                        update_payload[SampleStatsEx.ORIGIN.value] = df[SampleStatsEx.ORIGIN.value]
                        for origin, group in update_payload.groupby(SampleStatsEx.ORIGIN.value):
                            # Ensure index is sample_id
                            if group.index.name != SampleStatsEx.SAMPLE_ID.value:
                                # Try to find sample_id
                                if SampleStatsEx.SAMPLE_ID.value in group.columns:
                                    group = group.set_index(SampleStatsEx.SAMPLE_ID.value)

                            self._df_manager.upsert_df(group, origin=origin, force_flush=True)

                # Explicitly flush to disk to avoid race conditions with _slowUpdateInternals
                if self._df_manager:
                    try:
                        self._df_manager.flush()
                    except Exception as e:
                        logger.warning(f"Flush after modify failed: {e}")

                return f"Modified column '{col}' using: {code}"
            except Exception as e:
                logger.error(f"Modify failed: {e}")
                return f"Failed to modify column {col}: {e}"

        # C) Standard Pandas Ops (drop, sort, head, tail, sample, view_sort)
        if func in {"df.drop", "df.sort_values", "df.sort_index", "df.head", "df.tail", "df.sample", "df.sort_view_slice"}:
            func_name = func.replace("df.", "")
            try:
                if func_name == "sort_view_slice":
                     start = int(params.get("start", 0))
                     count = int(params.get("count", 0))
                     if count <= 0: count = len(df)
                     end = min(start + count, len(df))

                     if start < len(df):
                         # Extract and sort slice
                         sub_df = df.iloc[start:end].copy()

                         # Apply sort to slice
                         # Filter params for sort_values
                         sort_params = {k: v for k, v in params.items() if k not in ["start", "count"]}
                         
                         by_cols = sort_params.get("by", [])
                         # Handle special 'index' sort case which sort_values doesn't handle if 'index' is not a column
                         if by_cols and len(by_cols) == 1 and by_cols[0] == "index":
                             asc = sort_params.get("ascending", [True])
                             ascending_val = asc[0] if isinstance(asc, list) and asc else True
                             sub_df.sort_index(inplace=True, ascending=ascending_val)
                         else:
                             try:
                                 sub_df.sort_values(inplace=True, **sort_params)
                             except TypeError:
                                 # Fallback for mixed types
                                 if "key" not in sort_params:
                                     sort_params["key"] = lambda x: x.astype(str)
                                     sub_df.sort_values(inplace=True, **sort_params)

                         # Reassign values (ignoring index alignment to swap rows in place)
                         # We use .values to ensure we just paste the sorted data into these slots
                         df.iloc[start:end] = sub_df.values
                         
                         # CRITICAL: We must also update the index (Sample IDs) to match the moved data,
                         # otherwise Sample ID X will point to data from Sample ID Y (corruption).
                         try:
                             idx_name = df.index.name
                             new_index = df.index.to_numpy().copy()
                             new_index[start:end] = sub_df.index.to_numpy()
                             df.index = pd.Index(new_index, name=idx_name)
                         except Exception as e:
                             logger.error(f"Failed to update index in sort_view_slice: {e}")
                             # Fallback: try to reconstruct if possible or fail gracefully
                             raise e
                     
                     return f"Applied operation: sort_view_slice({start}:{end})"

                if func_name == "drop" and "index" in params:
                    index_to_drop = eval(params["index"], {"df": df, "np": np})
                    df.drop(index=index_to_drop, inplace=True)
                    return "Applied operation: drop"

                if func_name == "sort_values":
                    # Params are already cleaned by the Agent's SortHandler
                    try:
                        df.sort_values(inplace=True, **params)
                    except TypeError as e:
                        # Fallback for mixed types (e.g. lists vs strings/floats): sort by string representation
                        logger.warning(f"Sort failed due to type mismatch ({e}). Retrying with string conversion...")
                        if "key" not in params:
                            params["key"] = lambda x: x.astype(str)
                            df.sort_values(inplace=True, **params)
                        else:
                            raise e
                    return "Applied operation: sort_values"

                if func_name == "sort_index":
                    df.sort_index(inplace=True, **params)
                    return "Applied operation: sort_index"

                if func_name == "head":
                    n_raw = params.get("n", 5)
                    # Handle "%" strings
                    if isinstance(n_raw, str) and "%" in n_raw:
                        try:
                            n = int(len(df) * float(n_raw.replace("%", "")) / 100.0)
                        except:
                            n = 5
                    else:
                        n = int(n_raw)

                    if n < len(df):
                        df.drop(index=df.index.difference(df.index[:n]), inplace=True)
                    return f"Applied operation: head({n})"

                if func_name == "tail":
                    n_raw = params.get("n", 5)
                    # Handle "%" strings
                    if isinstance(n_raw, str) and "%" in n_raw:
                        try:
                            n = int(len(df) * float(n_raw.replace("%", "")) / 100.0)
                        except:
                            n = 5
                    else:
                        n = int(n_raw)

                    if n < len(df):
                        df.drop(index=df.index.difference(df.index[-n:]), inplace=True)
                    return f"Applied operation: tail({n})"

                # ... existing sample logic ...

            except Exception as e:
                logger.error(f"Op {func_name} failed: {e}")
                return f"Failed to apply {func_name}: {e}"

        # D) Analysis (Read-Only)
        if func == "df.analyze":
            code = params.get("code")
            if not code: return "No code provided"
            if "import " in code or "__" in code: return "Safety Violation"
            try:
                result = eval(code, {"df": df, "pd": pd, "np": np})
                return f"Analysis Result: {result}"
            except Exception as e:
                return f"Analysis Error: {e}"

        return "No operation applied"

    def _slowUpdateInternals(self, force=False):
        current_time = time.time()
        if not force and self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= 10:
            return

        updated_df = self._pull_into_all_data_view_df()

        # Guard against init race conditions
        if updated_df is None:
            return

        # Ensure default columns exist (persists them across updates even if not in H5 yet)
        if SampleStatsEx.TAGS.value not in updated_df.columns:
             updated_df[SampleStatsEx.TAGS.value] = ""
        if "natural_sort_score" not in updated_df.columns:
             updated_df["natural_sort_score"] = np.nan
        if "deny_listed" not in updated_df.columns:
             updated_df["deny_listed"] = False

        if self._is_filtered and self._all_datasets_df is not None:
            # The user has applied a custom view (Filter, Sort, or Aggregation).
            # We want to support Live Updates of values where possible (e.g. "Keep highest loss"),
            # but prevent overwriting the structure with the raw dataset.

            # Check if the current view's index is compatible with the raw source (Sample IDs)
            # If there is overlap, it's likely a Filter/Subset operation -> Update values, keep rows.
            # If no overlap, it's likely an Aggregation (Index changed) -> Freeze view (can't update from raw).

            try:
                # Use intersection to detect compatibility
                # We need to handle MultiIndex vs Index comparisons carefully, generally simplistic check is enough
                common_indices = self._all_datasets_df.index.intersection(updated_df.index)

                if len(common_indices) > 0:
                     # Case A: Filter/Subset. Indices match.
                     # We force the new data to conform to the USER'S current view (rows/order).
                     # providing live updates for the specific samples they are watching.
                     updated_df = updated_df.reindex(self._all_datasets_df.index)
                else:
                     # Case B: Aggregation/Transformation. Indices don't match.
                     # We cannot update an aggregated view (e.g. "Mean Loss by Class") from raw samples
                     # without re-running the aggregation query.
                     # Best behavior: Freeze the view (keep current df) so the user doesn't lose their chart.
                     return
            except Exception as e:
                logger.debug(f"[_slowUpdateInternals] Error matching indices for filtered view: {e}")
                return

        elif hasattr(self, "_all_datasets_df") and self._all_datasets_df is not None and not self._all_datasets_df.empty:
            # Case C: Standard/Unfiltered View.
            # Preserves Sticky Sort if user manually sorted the full list.

            # Simple heuristic: if the index has changed (reordered), re-apply it.

            # 1. Update the new DF with the new data
            # 2. Reindex the new DF to match the old DF's index order (intersection)
            # This keeps the user's sort valid for existing items.

            # Check if we have a custom sort (index is not strictly increasing monotonic)
            # AND if the index types are compatible (both numeric)
            if not self._all_datasets_df.index.is_monotonic_increasing:
                 # We have a custom sort.
                 # 1. Check strict equality first (fastest)
                 if self._all_datasets_df.index.equals(updated_df.index):
                     pass  # Index match, just use updated_df as is

                 else:
                     # We have a custom sort or mismatch.
                     old_index = self._all_datasets_df.index
                     new_index = updated_df.index

                     # Optimize intersection using set for O(1) lookups
                     new_index_set = set(new_index)
                     kept_indices = [x for x in old_index if x in new_index_set]

                     # Identify rows that are NEW
                     old_index_set = set(old_index)
                     newly_added_indices = [x for x in new_index if x not in old_index_set]

                     # Construct full order
                     full_order = kept_indices + newly_added_indices

                     # Reindex using this full order.
                     updated_df = updated_df.reindex(full_order)

        self._all_datasets_df = updated_df
        self._last_internals_update_time = current_time

    def _process_get_data_samples(self, request, context):
        """
        Actual implementation of GetDataSamples.
        Process the request and retrieve data samples from the dataframe.
        """
        try:
            start_time = time.time()
            logger.debug(
                "GetDataSamples processing: start_index=%s, records_cnt=%s, peer=%s, thread=%s, timestart=%s",
                request.start_index,
                request.records_cnt,
                context.peer(),
                threading.current_thread().name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # Validate request parameters
            if request.start_index < 0 or request.records_cnt <= 0:
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Invalid start_index or records_cnt",
                    data_records=[]
                )

            with self._lock:
                self._slowUpdateInternals()
                df_slice = self._all_datasets_df.iloc[request.start_index:request.start_index + request.records_cnt].reset_index()  # Reset index for proper row access with sample_id column

            if df_slice.empty:
                logger.warning(f"No samples found at index {request.start_index}:{request.start_index + request.records_cnt}")
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"No samples found at index {request.start_index}:{request.start_index + request.records_cnt}",
                    data_records=[]
                )

            logger.debug(
                "Retrieving samples from %s to %s", request.start_index, request.start_index + request.records_cnt)

            # Build the data records list using shared executor
            data_records = []
            tasks = [(row, request, df_slice.columns) for _, row in df_slice.iterrows()]

            logger.debug("Processing %s samples at %s", len(tasks), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            data_records = self._data_executor.map(self._process_sample_row, tasks, timeout=120)
            data_records = [r for r in data_records if r is not None]
            logger.debug("Completed processing at %s in %.2f seconds\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time)

            return pb2.DataSamplesResponse(
                success=True,
                message=f"Retrieved {len(data_records)} data records",
                data_records=data_records
            )

        except Exception as e:
            logger.error("Failed to retrieve samples: %s", str(e), exc_info=True)
            return pb2.DataSamplesResponse(
                success=False,
                message=f"Failed to retrieve samples: {str(e)}\n{traceback.format_exc()}",
                data_records=[]
            )

    # RPC Implementations
    # ===================
    def ApplyDataQuery(self, request, context):
        """
        Apply a query on the in-memory dataframe.

        Modes:
          - request.query == ""  -> just return counts, do not modify df
          - request.query != ""  -> always handled by the agent (natural language path)

        Counts returned:
          - number_of_all_samples: all rows currently in the dataframe
          - number_of_samples_in_the_loop: rows not deny_listed
          - number_of_discarded_samples: rows with deny_listed == True
        """
        self._ctx.ensure_components()
        components = self._ctx.components

        # 1) No query: just report counts (Needs lock for consistency)
        if request.query == "":
            with self._lock:
                 df = self._all_datasets_df
                 return self._build_success_response(
                    df=df,
                    message=f"Current dataframe has {len(df)} samples",
                )

        try:
            # 2) Check if we should bypass the agent (Quick Filters path)
            if not request.is_natural_language:
                logger.info(
                    "[ApplyDataQuery] BYPASSING AGENT - Direct query execution: %r",
                    request.query,
                )

                # Parse the query directly without agent
                # Expected format: "col > val and col2 == val2 sortby col desc"
                operations = self._parse_direct_query(request.query)

                # Apply operations with lock
                with self._lock:
                    df = self._all_datasets_df
                    messages = []


                    for op in operations:
                        func = op.get("function")
                        params = op.get("params", {}) or {}
                        msg = self._apply_agent_operation(df, func, params)
                        messages.append(msg)

                    final_message = " | ".join(messages) if messages else "No operation performed"
                    self._all_datasets_df = df

                    # Direct queries are manipulations -> Freeze the view
                    if operations:
                         self._is_filtered = True

                return self._build_success_response(
                    df=df,
                    message=final_message,
                    intent_type=pb2.INTENT_FILTER
                )

            # 3) Natural language path - go through agent
            logger.debug(
                "[ApplyDataQuery] Using AGENT for natural language query: %r",
                request.query,
            )

            if self._agent is None:
                return pb2.DataQueryResponse(
                    success=False,
                    message="Natural language queries require agent (not available)",
                )

            # Agent translates query text -> operations spec (List[dict])
            # Executed outside the lock to keep grid responsive during LLM waiting time
            abort_event = threading.Event()
            if context:
                context.add_callback(lambda: abort_event.set())

            def status_cb(msg: str):
                logger.debug(f"[ApplyDataQuery] Status: {msg}")

            operations = self._agent.query(request.query, abort_event=abort_event, status_callback=status_cb)
            if isinstance(operations, dict): operations = [operations] # Backwards compat
            if not operations: operations = []

            # 3) Apply Operations (CPU/MEMORY BOUND - REQUIRES LOCK)
            with self._lock:
                # Start with the current authoritative DF
                df = self._all_datasets_df
                messages = []
                # Default to FILTER, switch to ANALYSIS if we detect analysis/action output
                intent_type = pb2.INTENT_FILTER
                analysis_result = ""

                for i, op in enumerate(operations):
                    func = op.get("function")
                    params = op.get("params", {}) or {}

                    # 2a) Agent-driven RESET has highest priority
                    if params.get("__agent_reset__"):
                        logger.debug("[ApplyDataQuery] Agent requested reset")
                        # Rebuild from loaders; this is the only place we replace the df object
                        self._all_datasets_df = self._pull_into_all_data_view_df()
                        df = self._all_datasets_df  # Reset df to full dataset
                        self._is_filtered = False   # Unfreeze updates
                        messages.append("Reset view")
                        continue

                    # 2b) All other agent operations mutate df in-place
                    # df is now carried forward across iterations
                    msg = self._apply_agent_operation(df, func, params)
                    messages.append(msg)

                    # --- UPDATED INTENT CLASSIFICATION ---
                    # Check for Clarification
                    if "Clarification needed" in msg or "I need more information" in msg:
                        intent_type = pb2.INTENT_ANALYSIS  # Usually presented as a message/analysis
                        analysis_result = msg

                    # Check for Action Triggers (e.g. "Action: Dataset saved...")
                    elif msg.startswith("Action:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg

                    # Check for Analysis Results
                    elif msg.startswith("Analysis Result:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg.replace("Analysis Result:", "").strip()

                    # Check for Errors
                    elif msg.startswith("Analysis Error:") or msg.startswith("Safety Violation:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg

                final_message = " | ".join(messages) if messages else "No operation performed"

                # 4) Persist the filtered df back for next query (CRITICAL for sequential filters!)
                # Only update if it was a manipulation query, not analysis
                if intent_type == pb2.INTENT_FILTER:
                    self._all_datasets_df = df
                    # If we modified the DF, we should freeze it (unless it was a Reset which handled above)
                    # We check if *any* operation was effectively applied.
                    # For simplicity, if intent is FILTER, we assume manipulation happened.
                    self._is_filtered = True

                # 5) Return updated counts after mutation
                return self._build_success_response(
                    df=df,
                    message=final_message,
                    intent_type=intent_type,
                    analysis_result=analysis_result
                )

        except Exception as e:
            logger.error(f"ApplyDataQuery: Failed to apply query: {e}", exc_info=True)
            return pb2.DataQueryResponse(
                success=False,
                message=f"Failed to apply query: {str(e)}",
            )

    def GetDataSamples(self, request, context):
        """
        Retrieve samples from the dataframe with their data statistics.
        Implements request deduplication to prevent duplicate concurrent requests.
        """

        try:
            # Process the request directly without deduplication logic
            return self._process_get_data_samples(request, context)

        except Exception as e:
            logger.error("Error in GetDataSamples: %s", str(e), exc_info=True)
            return pb2.DataSamplesResponse(
                success=False,
                message=f"Failed to retrieve samples: {str(e)}",
                data_records=[]
            )

    def EditDataSample(self, request, context):
        """
        Edit sample metadata (tags and deny_listed).
        """

        if self._all_datasets_df is None:
            self._initialize_data_service()

        self._ctx.ensure_components()
        components = self._ctx.components

        # Pause training if it's currently running
        trainer = components.get("trainer")
        hp = components.get("hyperparams")
        if trainer:
            logger.info("Pausing training before restore...")
            trainer.pause()
            if "is_training" in hp:
                hp['is_training'] = False
            else:
                hp["is_training"] = False

        if request.stat_name not in [SampleStatsEx.TAGS.value, SampleStatsEx.DENY_LISTED.value]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags' and 'deny_listed' stat editing is supported for now.",
            )

        if request.stat_name == SampleStatsEx.TAGS.value:
            request.string_value = request.string_value or ""

        # No dataset lookups needed; all edits apply directly to the global dataframe.

        # ---------------------------------------------------------------------
        # 1) Mirror edits into the in-memory DataFrame
        # ---------------------------------------------------------------------
        with self._lock:
            if self._all_datasets_df is not None:
                uses_multiindex = isinstance(self._all_datasets_df.index, pd.MultiIndex)
                for sid, origin in zip(request.samples_ids, request.sample_origins):
                    if request.stat_name == SampleStatsEx.TAGS.value:
                        new_val = request.string_value
                        if request.type == pb2.SampleEditType.EDIT_OVERRIDE:
                            # EDIT_OVERRIDE: directly use the new value
                            target_val = new_val or ""
                        elif (request.type == pb2.SampleEditType.EDIT_ACCUMULATE or request.type == pb2.SampleEditType.EDIT_REMOVE):
                            try:
                                if uses_multiindex:
                                    current_val = self._all_datasets_df.loc[(origin, sid), SampleStatsEx.TAGS.value]
                                else:
                                    current_val = self._all_datasets_df.loc[sid, SampleStatsEx.TAGS.value]
                            except KeyError:
                                current_val = ""

                            if pd.isna(current_val): current_val = ""
                            current_tags = [t.strip() for t in str(current_val).split(';') if t.strip()]

                            if request.type == pb2.SampleEditType.EDIT_ACCUMULATE:
                                if new_val not in current_tags:
                                    current_tags.append(new_val)
                            else: # EDIT_REMOVE
                                if new_val in current_tags:
                                    current_tags.remove(new_val)

                            target_val = ";".join(current_tags)
                        else:
                            target_val = new_val or ""
                    else:
                        target_val = request.bool_value

                    try:
                        if uses_multiindex:
                            self._all_datasets_df.loc[(origin, sid), request.stat_name] = target_val
                        else:
                            # Fallback: origin / sample_id as columns
                            mask = (
                                (self._all_datasets_df.index == sid)
                                & (self._all_datasets_df[SampleStatsEx.ORIGIN.value] == origin)
                            )
                            self._all_datasets_df.loc[mask, request.stat_name] = target_val


                    except Exception as e:
                        logger.debug(
                            f"[EditDataSample] Failed to update dataframe for sample {sid}: {e}"
                        )

            # ------------------------------------------------------------------
            # 3) Persist edits into the global dataframe manager (in-memory ledger)
            # MOVED INSIDE LOCK to prevent race conditions with _slowUpdateInternals
            # ------------------------------------------------------------------
            try:
                if request.samples_ids and self._df_manager is not None:
                    updates_by_origin = {}
                    for sid, origin in zip(request.samples_ids, request.sample_origins):
                        # Tags
                        if request.stat_name == SampleStatsEx.TAGS.value:
                            # Logic to calculate new tags based on edit type
                            # use self._all_datasets_df (which was just updated above) as source of truth
                            # instead of pulling from _df_manager again, to ensure consistency inside the lock.
                            uses_multiindex = self._all_datasets_df is not None and isinstance(self._all_datasets_df.index, pd.MultiIndex)
                            current_val = ""
                            try:
                                if uses_multiindex:
                                    current_val = self._all_datasets_df.loc[(origin, sid), SampleStatsEx.TAGS.value]
                                else:
                                    # Fallback
                                    mask = (self._all_datasets_df.index == sid) & (self._all_datasets_df[SampleStatsEx.ORIGIN.value] == origin)
                                    if mask.any():
                                        current_val = self._all_datasets_df.loc[mask, SampleStatsEx.TAGS.value].iloc[0]
                            except Exception:
                                pass

                            target_val = current_val # Already updated above in step 1

                            updates_by_origin.setdefault(origin, []).append({
                                "sample_id": int(sid),
                                SampleStatsEx.ORIGIN.value: origin,
                                request.stat_name: target_val,
                            })
                        else:
                            # Deny_listed
                            updates_by_origin.setdefault(origin, []).append({
                                "sample_id": int(sid),
                                SampleStatsEx.ORIGIN.value: origin,
                                request.stat_name: request.bool_value,
                            })

                    for origin, rows in updates_by_origin.items():
                        df_update = pd.DataFrame(rows).set_index("sample_id")
                        # upsert_df updates the ledger's dataframe immediately
                        self._df_manager.upsert_df(df_update, origin=origin, force_flush=True)

            except Exception as e:
                logger.debug(f"[EditDataSample] Failed to upsert edits into global dataframe: {e}")

            # Prevent _slowUpdateInternals from overwriting our in-memory edits with stale data
            # from the disk/db for a few seconds.
            self._last_internals_update_time = time.time()

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )

    def GetDataSplits(self, request, context):
        """
        Return the list of available dataset splits (train, test, val, etc.)
        Extracted from the global dataframe's origin column.
        """

        try:
            split_names = []
            with self._lock:
                self._slowUpdateInternals()
                if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                    if SampleStatsEx.ORIGIN.value in self._all_datasets_df.columns:
                        split_names = sorted(self._all_datasets_df[SampleStatsEx.ORIGIN.value].unique().tolist())
                    elif isinstance(self._all_datasets_df.index, pd.MultiIndex):
                        if SampleStatsEx.ORIGIN.value in self._all_datasets_df.index.names:
                            split_names = sorted(self._all_datasets_df.index.get_level_values(SampleStatsEx.ORIGIN.value).unique().tolist())
            logger.info(f"GetDataSplits returning: {split_names}")

            return pb2.DataSplitsResponse(
                success=True,
                split_names=split_names
            )

        except Exception as e:
            logger.error(f"GetDataSplits failed: {e}", exc_info=True)
            return pb2.DataSplitsResponse(
                success=False,
                split_names=[]
            )

    def CheckAgentHealth(self, request, context):
        """
        gRPC method to check if the agent is available for natural language queries.
        Returns:
            AgentHealthResponse { available: bool, message: str }
        """

        try:
            available = self.is_agent_available()
            msg = "Agent is available" if available else "Agent is not available"
            return pb2.AgentHealthResponse(available=available, message=msg)
        except Exception as e:
            return pb2.AgentHealthResponse(available=False, message=f"Error: {e}")
