"""Unit tests for the CVAT / Label Studio / V7 Darwin format encoders.

Each encoder is tested against a manually built ImageAnnotations fixture --
no dependency on the dataframe/ledger layer here, just IR -> bytes.
"""

import json
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

import pytest

from weightslab.export.formats.cvat import to_cvat_xml
from weightslab.export.formats.label_studio import to_label_studio_json
from weightslab.export.formats.v7_darwin import to_v7_darwin_zip
from weightslab.export.models import BoxAnnotation, ImageAnnotations, PolygonAnnotation


@pytest.fixture
def images():
    return [
        ImageAnnotations(
            sample_id="s1",
            filename="img1.jpg",
            width=200,
            height=100,
            boxes=[BoxAnnotation(10.0, 20.0, 110.0, 70.0, "car")],
            polygons=[PolygonAnnotation(points=[(0.0, 0.0), (50.0, 0.0), (25.0, 50.0)], label="person")],
        ),
        ImageAnnotations(
            sample_id="s2",
            filename="img2.jpg",
            width=50,
            height=50,
            boxes=[],
            polygons=[],
        ),
    ]


class TestCvatEncoder:

    def test_produces_well_formed_xml_with_expected_structure(self, images):
        payload = to_cvat_xml(images)
        root = ET.fromstring(payload)
        assert root.tag == "annotations"

        image_els = root.findall("image")
        assert len(image_els) == 2
        assert image_els[0].get("name") == "img1.jpg"
        assert image_els[0].get("width") == "200"
        assert image_els[0].get("height") == "100"

        box_els = image_els[0].findall("box")
        assert len(box_els) == 1
        assert box_els[0].get("label") == "car"
        assert box_els[0].get("xtl") == "10.00"
        assert box_els[0].get("ybr") == "70.00"

        poly_els = image_els[0].findall("polygon")
        assert len(poly_els) == 1
        assert poly_els[0].get("label") == "person"
        assert poly_els[0].get("points") == "0.00,0.00;50.00,0.00;25.00,50.00"

    def test_labels_declared_once_in_meta(self, images):
        payload = to_cvat_xml(images)
        root = ET.fromstring(payload)
        label_names = [el.find("name").text for el in root.findall("./meta/task/labels/label")]
        assert label_names == ["car", "person"]

    def test_empty_images_list_still_produces_valid_xml(self):
        payload = to_cvat_xml([])
        root = ET.fromstring(payload)
        assert root.findall("image") == []


class TestLabelStudioEncoder:

    def test_coordinates_converted_to_percent(self, images):
        payload = to_label_studio_json(images)
        tasks = json.loads(payload)
        assert len(tasks) == 2

        result = tasks[0]["annotations"][0]["result"]
        box_result = next(r for r in result if r["type"] == "rectanglelabels")
        # x=10/200*100=5.0, y=20/100*100=20.0, width=100/200*100=50.0, height=50/100*100=50.0
        assert box_result["value"]["x"] == pytest.approx(5.0)
        assert box_result["value"]["y"] == pytest.approx(20.0)
        assert box_result["value"]["width"] == pytest.approx(50.0)
        assert box_result["value"]["height"] == pytest.approx(50.0)
        assert box_result["value"]["rectanglelabels"] == ["car"]

        poly_result = next(r for r in result if r["type"] == "polygonlabels")
        assert poly_result["value"]["polygonlabels"] == ["person"]
        assert poly_result["value"]["points"][1] == pytest.approx([25.0, 0.0])

    def test_task_without_annotations_has_empty_result(self, images):
        payload = to_label_studio_json(images)
        tasks = json.loads(payload)
        assert tasks[1]["annotations"][0]["result"] == []

    def test_data_image_field_is_filename(self, images):
        payload = to_label_studio_json(images)
        tasks = json.loads(payload)
        assert tasks[0]["data"]["image"] == "img1.jpg"


class TestV7DarwinEncoder:

    def test_zip_contains_one_json_per_image(self, images):
        payload = to_v7_darwin_zip(images)
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            names = sorted(zf.namelist())
            assert names == ["img1.json", "img2.json"]

            doc = json.loads(zf.read("img1.json"))
            assert doc["version"] == "2.0"
            assert doc["item"]["name"] == "img1.jpg"

            box_ann = next(a for a in doc["annotations"] if "bounding_box" in a)
            assert box_ann["name"] == "car"
            assert box_ann["bounding_box"] == {"x": 10.0, "y": 20.0, "w": 100.0, "h": 50.0}

            poly_ann = next(a for a in doc["annotations"] if "polygon" in a)
            assert poly_ann["name"] == "person"
            assert poly_ann["polygon"]["paths"][0][0] == {"x": 0.0, "y": 0.0}

    def test_filename_collisions_are_disambiguated(self):
        images = [
            ImageAnnotations(sample_id="a", filename="dup.jpg", width=10, height=10),
            ImageAnnotations(sample_id="b", filename="dup.jpg", width=10, height=10),
        ]
        payload = to_v7_darwin_zip(images)
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            names = sorted(zf.namelist())
            assert len(names) == 2
            assert len(set(names)) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
