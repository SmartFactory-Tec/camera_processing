import numpy as np

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Generator, Type

from sems_vision.frame_packet import FramePacket

BoundingBox = tuple[int, int, int, int]


@dataclass
class Detection:
    bounding_box: BoundingBox
    confidence: float
