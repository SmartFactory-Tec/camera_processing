from dataclasses import dataclass

BoundingBox = tuple[int, int, int, int]


@dataclass
class Detection:
    bounding_box: BoundingBox
    confidence: float
