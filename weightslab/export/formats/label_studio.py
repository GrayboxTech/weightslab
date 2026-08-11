"""Label Studio JSON export encoder.

Schema reference: https://labelstud.io/guide/export.html#JSON
Rectangle/polygon coordinates are stored as percentages (0-100) of the
image's original width/height, per Label Studio's ``rectanglelabels`` /
``polygonlabels`` result types.
"""

import json
import uuid
from typing import List

from weightslab.export.models import ImageAnnotations


def to_label_studio_json(images: List[ImageAnnotations]) -> bytes:
    tasks = []
    for task_id, img in enumerate(images, start=1):
        width = img.width or 1
        height = img.height or 1
        results = []

        for box in img.boxes:
            results.append({
                "id": uuid.uuid4().hex[:10],
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img.width,
                "original_height": img.height,
                "value": {
                    "x": box.x1 / width * 100.0,
                    "y": box.y1 / height * 100.0,
                    "width": (box.x2 - box.x1) / width * 100.0,
                    "height": (box.y2 - box.y1) / height * 100.0,
                    "rotation": 0,
                    "rectanglelabels": [box.label],
                },
            })

        for poly in img.polygons:
            results.append({
                "id": uuid.uuid4().hex[:10],
                "type": "polygonlabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": img.width,
                "original_height": img.height,
                "value": {
                    "points": [[x / width * 100.0, y / height * 100.0] for x, y in poly.points],
                    "polygonlabels": [poly.label],
                },
            })

        tasks.append({
            "id": task_id,
            "data": {"image": img.filename},
            "annotations": [{"id": task_id, "result": results}],
        })

    return json.dumps(tasks, indent=2).encode("utf-8")
