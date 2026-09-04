"""CVAT XML 1.1 (images task) encoder.

Schema reference: https://opencv.github.io/cvat/docs/manual/advanced/xml_format/
"""

import xml.etree.ElementTree as ET
from typing import List
from xml.dom import minidom

from weightslab.export.models import ImageAnnotations


def to_cvat_xml(images: List[ImageAnnotations]) -> bytes:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "weightslab-export"
    ET.SubElement(task, "size").text = str(len(images))
    ET.SubElement(task, "mode").text = "annotation"

    labels_el = ET.SubElement(task, "labels")
    seen_labels: List[str] = []
    for img in images:
        for ann in (*img.boxes, *img.polygons):
            if ann.label not in seen_labels:
                seen_labels.append(ann.label)
    for label in seen_labels:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = label

    for image_id, img in enumerate(images):
        image_el = ET.SubElement(
            root, "image",
            id=str(image_id), name=img.filename,
            width=str(img.width), height=str(img.height),
        )
        for box in img.boxes:
            ET.SubElement(
                image_el, "box",
                label=box.label,
                xtl=f"{box.x1:.2f}", ytl=f"{box.y1:.2f}",
                xbr=f"{box.x2:.2f}", ybr=f"{box.y2:.2f}",
                occluded="0", z_order="0",
            )
        for poly in img.polygons:
            points_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in poly.points)
            ET.SubElement(
                image_el, "polygon",
                label=poly.label, points=points_str,
                occluded="0", z_order="0",
            )

    raw = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")
