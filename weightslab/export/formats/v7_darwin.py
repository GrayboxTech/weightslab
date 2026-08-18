"""V7 (Darwin JSON 2.0) export encoder.

Schema reference: https://docs.v7labs.com/reference/darwin-json
Darwin expects one JSON annotation file per image (matched by filename on
import), so this bundles all per-image documents into a single zip.
"""

import io
import json
import os
import uuid
import zipfile
from typing import List

from weightslab.export.models import ImageAnnotations


def _image_document(img: ImageAnnotations) -> dict:
    annotations = []
    for box in img.boxes:
        annotations.append({
            "id": str(uuid.uuid4()),
            "name": box.label,
            "bounding_box": {
                "x": box.x1,
                "y": box.y1,
                "w": box.x2 - box.x1,
                "h": box.y2 - box.y1,
            },
        })
    for poly in img.polygons:
        annotations.append({
            "id": str(uuid.uuid4()),
            "name": poly.label,
            "polygon": {"paths": [[{"x": x, "y": y} for x, y in poly.points]]},
        })

    return {
        "version": "2.0",
        "item": {
            "name": img.filename,
            "path": "/",
            "source_info": {"item_id": img.sample_id},
            "slots": [{
                "type": "image",
                "slot_name": "0",
                "width": img.width,
                "height": img.height,
            }],
        },
        "annotations": annotations,
    }


def to_v7_darwin_zip(images: List[ImageAnnotations]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        used_names = set()
        for img in images:
            json_name = os.path.splitext(img.filename)[0] + ".json"
            # Guard against filename collisions (e.g. synthetic "sample_<id>"
            # names sharing a stem after extension-stripping).
            candidate = json_name
            suffix = 1
            while candidate in used_names:
                candidate = f"{os.path.splitext(json_name)[0]}_{suffix}.json"
                suffix += 1
            used_names.add(candidate)
            zf.writestr(candidate, json.dumps(_image_document(img), indent=2))
    return buf.getvalue()
