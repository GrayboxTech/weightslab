"""
Instance Merger: Reconstruct original multi-instance data from expanded dataframe rows.

When the dataframe is expanded with (sample_id, annotation_id) multi-index,
each instance gets its own row. This module merges them back to original format
when querying samples from the UI.

Rules:
- Detection: List of bboxes → merged to list
- Segmentation: List of masks → merged to list; single mask stays as array
- Classification: List of labels → merged to list; single label stays as scalar
"""

import numpy as np
import torch
from typing import Any, List, Tuple, Union


def _is_array_like(obj):
    """Check if object is a numpy array, tensor, or list."""
    return isinstance(obj, (np.ndarray, torch.Tensor, list))


def _convert_to_list(obj) -> list:
    """Convert array-like object to list."""
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, list):
        return obj
    return [obj]


def _count_instances_for_merge(value: Any) -> int:
    """Count instances in a single value (before expansion).

    Used to determine if a value represents single or multiple instances.
    """
    if value is None:
        return 1

    # List of array-like items → multiple instances
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 1
        first_item = value[0]
        if isinstance(first_item, (np.ndarray, torch.Tensor, list)):
            return len(value)
        # List of scalars → single instance with multiple values
        try:
            all_scalar = all(isinstance(item, (int, float, np.integer, np.floating)) for item in value)
            if all_scalar:
                return 1
        except:
            pass
        return len(value)

    # Single array/tensor → single instance
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return 1

    # Scalar → single instance
    return 1


def merge_detection_instances(instance_values: List[Any]) -> Union[list, np.ndarray]:
    """Merge detection bboxes from multiple rows back to list format.

    Args:
        instance_values: List of bbox values, one per row
                        Each can be a single bbox [x1,y1,x2,y2,...] or None

    Returns:
        List of bboxes (original format)
        Example: [[10,20,30,40], [50,60,70,80], [90,100,110,120]]
    """
    merged = []
    for val in instance_values:
        if val is not None:
            merged.append(val)
    return merged if merged else None


def merge_segmentation_instances(instance_values: List[Any], task_type: str = "segmentation") -> Union[np.ndarray, None]:
    """Merge segmentation masks from multiple rows back to original format.

    Segmentation masks with pixel labels are AGGREGATED using MAX operation.
    This preserves all instance labels by taking the maximum value at each pixel.

    Args:
        instance_values: List of mask values, one per row
                        Each can be a single binary/label mask (H, W) array or None

    Returns:
        Aggregated array of all masks:
        - Multiple masks: MAX aggregation across instances at each pixel
        - Single mask: Returned as (H, W) array
        - All None: Return None

    Example (binary masks):
        - Input: [mask0, mask1, mask2] where each is (512, 512)
          mask0: [[1, 0], [1, 0]]
          mask1: [[0, 1], [0, 1]]
          mask2: [[1, 0], [0, 1]]
          Output: np.max([mask0, mask1, mask2], axis=0)
                = [[1, 1], [1, 1]]  [MAX aggregated!]

        - Input: [mask0] (single mask)
          Output: mask0 as-is (512, 512)
    """
    masks = []
    for val in instance_values:
        if val is not None:
            masks.append(val)

    if not masks:
        return None

    # Convert to numpy arrays for aggregation
    masks_np = []
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            masks_np.append(mask.cpu().numpy())
        elif isinstance(mask, np.ndarray):
            masks_np.append(mask)
        else:
            masks_np.append(np.asanyarray(mask))

    # Aggregate masks using MAX operation
    if len(masks_np) == 1:
        # Single mask: return as (H, W) array
        return masks_np[0]
    else:
        # Multiple masks: aggregate using max at each pixel
        # Stack temporarily for max operation, then return result
        stacked = np.stack(masks_np, axis=0)
        return np.max(stacked, axis=0)  # Take max across instances → (H, W)


def merge_classification_instances(instance_values: List[Any], task_type: str = "classification") -> Union[list, None]:
    """Merge classification labels from multiple rows back to original format.

    Classification labels are ALWAYS returned as a list (even for single labels).

    Args:
        instance_values: List of label values, one per row
                        Each can be a single label (int, str, etc.) or None

    Returns:
        List of labels:
        - Single label: [label] (wrapped in list)
        - Multiple labels: [label1, label2, ...] (list of labels)
        - All None: Return None

    Example:
        - Input: ['cat', None, None] → Output: ['cat']  [LIST!]
        - Input: ['cat', 'dog', 'animal'] → Output: ['cat', 'dog', 'animal']
    """
    labels = []
    for val in instance_values:
        if val is not None:
            labels.append(val)

    if not labels:
        return None

    # Always return as list (even single label)
    return labels


def merge_instances_by_task_type(
    instance_rows: List[Tuple[str, Any]],
    task_type: str,
    column_name: str = "target"
) -> Any:
    """Merge instances based on task type.

    Args:
        instance_rows: List of (annotation_id, value) tuples from expanded dataframe
        task_type: Task type ('detection', 'segmentation', 'classification', etc.)
        column_name: Name of the column being merged (for context)

    Returns:
        Merged value in original format (list or scalar depending on instance count)
    """
    # Extract values in order of annotation_id
    instance_rows_sorted = sorted(instance_rows, key=lambda x: int(x[0]))
    values = [val for _, val in instance_rows_sorted]

    task_type_lower = str(task_type or "").strip().lower()

    if task_type_lower == "detection":
        return merge_detection_instances(values)
    elif task_type_lower == "segmentation":
        return merge_segmentation_instances(values, task_type)
    elif task_type_lower == "classification":
        return merge_classification_instances(values, task_type)
    else:
        # Default: treat as detection (most common)
        return merge_detection_instances(values)


def group_instances_by_sample(df_slice, target_column: str, task_type: str):
    """Group multi-index dataframe rows by sample_id and merge instances.

    Used when querying a sample that has multiple annotation rows.

    Args:
        df_slice: Dataframe slice with (sample_id, annotation_id) multi-index
        target_column: Column to merge (e.g., 'target', 'prediction')
        task_type: Task type for determining merge strategy

    Returns:
        Dict mapping sample_id to merged value
        Example: {
            'sample_0': [bbox0, bbox1, bbox2],  # Detection: list of bboxes
            'sample_1': [mask0, mask1],          # Segmentation: list of masks
            'sample_2': 'cat',                   # Classification: single label
        }
    """
    if df_slice.empty or target_column not in df_slice.columns:
        return {}

    merged_data = {}

    # Check if index is MultiIndex with (sample_id, annotation_id)
    if isinstance(df_slice.index, pd.MultiIndex) and df_slice.index.nlevels >= 2:
        # Group by sample_id (first index level)
        sample_ids = df_slice.index.get_level_values(0).unique()

        for sample_id in sample_ids:
            # Get all rows for this sample
            sample_rows = df_slice.loc[sample_id]

            # Handle both Series (single row) and DataFrame (multiple rows)
            if isinstance(sample_rows, pd.Series):
                # Single annotation for this sample
                merged_data[sample_id] = sample_rows.get(target_column)
            else:
                # Multiple annotations for this sample
                instance_values = [
                    (row.name if hasattr(row, 'name') else str(idx), row.get(target_column))
                    for idx, (_, row) in enumerate(sample_rows.iterrows())
                ]
                # Sort by annotation_id (second index level)
                instance_values_sorted = sorted(instance_values, key=lambda x: int(x[0]))

                # Merge based on task type
                merged_data[sample_id] = merge_instances_by_task_type(
                    instance_values_sorted, task_type, target_column
                )
    else:
        # Regular single-index dataframe (no expansion)
        for _, row in df_slice.iterrows():
            sample_id = row.get('sample_id') or row.name
            merged_data[sample_id] = row.get(target_column)

    return merged_data


def apply_merged_data_to_rows(rows: List[dict], merged_data: dict, column_name: str):
    """Apply merged instance data back to rows for UI display.

    Args:
        rows: List of row dicts (one per sample_id)
        merged_data: Dict mapping sample_id to merged values
        column_name: Column to update

    Returns:
        Updated rows with merged data
    """
    for row in rows:
        sample_id = row.get('sample_id')
        if sample_id in merged_data:
            row[column_name] = merged_data[sample_id]
    return rows


# Lazy import to avoid circular dependencies
import pandas as pd
