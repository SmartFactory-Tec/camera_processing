from dataclasses import dataclass
from . import camera_service

@dataclass
class CameraServerConfig:
    hostname: str
    port: int
    use_https: bool


@dataclass
class Config:
    camera_service: camera_service.Config
    camera_server: CameraServerConfig
