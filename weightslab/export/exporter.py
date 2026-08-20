"""Top-level dispatcher for annotation export -- the single entry point shared
by the Python API (``wl.export_annotations``), the gRPC handler backing the
Weights Studio "Export" button, and the ``weightslab export`` CLI command.
"""

import logging
import os
from typing import List, Optional, Tuple, Union

from weightslab.export.collect import collect_image_annotations
from weightslab.export.formats.cvat import to_cvat_xml
from weightslab.export.formats.label_studio import to_label_studio_json
from weightslab.export.formats.v7_darwin import to_v7_darwin_zip

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("cvat", "label_studio", "v7")

# format -> (encoder, default filename, mime type)
_ENCODERS = {
    "cvat": (to_cvat_xml, "annotations_cvat.xml", "application/xml"),
    "label_studio": (to_label_studio_json, "annotations_label_studio.json", "application/json"),
    "v7": (to_v7_darwin_zip, "annotations_v7_darwin.zip", "application/zip"),
}


def export_annotations(
    fmt: str,
    origin: Optional[str] = None,
    class_names: Optional[Union[dict, list, tuple]] = None,
    use_predictions: bool = False,
    tags: Optional[List[str]] = None,
) -> Tuple[bytes, str, str, int]:
    """Collect annotations from the registered dataframe and encode them as `fmt`.

    Args:
        fmt: one of ``SUPPORTED_FORMATS`` (``"cvat"``, ``"label_studio"``, ``"v7"``).
        origin: restrict to one split/loader name; ``None`` exports every split.
        class_names: explicit class-id -> name mapping, overriding any
            auto-detected ``dataset.class_names``.
        use_predictions: export model predictions instead of ground-truth targets.
        tags: restrict to samples carrying ANY of these tags (e.g. ``["ToReview"]``);
            ``None``/empty exports every sample.

    Returns:
        ``(payload_bytes, filename, mime_type, image_count)``.
    """
    fmt = (fmt or "").strip().lower()
    if fmt not in _ENCODERS:
        raise ValueError(f"Unknown export format {fmt!r}. Supported: {', '.join(SUPPORTED_FORMATS)}")

    images = collect_image_annotations(
        origin=origin, class_names=class_names, use_predictions=use_predictions, tags=tags,
    )
    encoder, filename, mime_type = _ENCODERS[fmt]
    payload = encoder(images)
    logger.info("[export] Encoded %d image(s) to %s format (%d bytes) at %s", len(images), fmt, len(payload), filename)
    return payload, filename, mime_type, len(images)


def save_export(fmt: str, output_path: str, **kwargs) -> str:
    """Export and write the result to `output_path`.

    If `output_path` is an existing directory (or ends in a path separator),
    the format's default filename is appended. Returns the path written.
    """
    payload, default_filename, _mime_type, image_count = export_annotations(fmt, **kwargs)

    _root, _ext = os.path.splitext(output_path)
    treat_as_dir = (
        os.path.isdir(output_path)
        or output_path.endswith(("/", "\\"))
        or (_ext == "" and not os.path.exists(output_path))
    )

    if treat_as_dir:
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, default_filename)
    else:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(payload)

    logger.info("[export] Wrote %d image(s) to %s (%s format)", image_count, output_path, fmt)
    return output_path
