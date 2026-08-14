"""Plain intermediate-representation types for annotation export.

Every format encoder (CVAT / Label Studio / V7 Darwin) consumes a list of
``ImageAnnotations`` — this is the single point where WeightsLab's internal
dataframe representation gets translated into something format-agnostic.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BoxAnnotation:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str


@dataclass
class PolygonAnnotation:
    points: List[Tuple[float, float]]
    label: str


@dataclass
class ImageAnnotations:
    sample_id: str
    filename: str
    width: int
    height: int
    origin: str = ""
    boxes: List[BoxAnnotation] = field(default_factory=list)
    polygons: List[PolygonAnnotation] = field(default_factory=list)
