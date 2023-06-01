from dataclasses import dataclass, field
import datetime
from typing import Any, Generator, Union
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    values: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.datetime.now)


FramePacketProcessor = Generator[FramePacket, FramePacket, None]
FramePacketGenerator = Union[Generator[FramePacket, None, None], FramePacketProcessor]
