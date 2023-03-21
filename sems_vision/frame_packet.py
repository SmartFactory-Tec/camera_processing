from dataclasses import dataclass, field
from time import time
from typing import Any, Generator

import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    values: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)


FramePacketGenerator = Generator[FramePacket, FramePacket, None]
