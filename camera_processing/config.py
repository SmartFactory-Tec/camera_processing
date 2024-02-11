from dataclasses import dataclass
from . import camera_service


@dataclass
class CameraStreamerConfig:
    hostname: str
    port: int
    use_https: bool


@dataclass
class KafkaConfig:
    hostnames: list[str]


@dataclass
class Config:
    # kafka: KafkaConfig
    camera_service: camera_service.Config
    camera_streamer: CameraStreamerConfig
