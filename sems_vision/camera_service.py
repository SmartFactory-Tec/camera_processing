import requests
from dacite import from_dict
from dataclasses import dataclass
from .camera import Camera


class InvalidResponseError(Exception):
    pass


@dataclass
class Config:
    hostname: str
    port: int
    use_https: bool


class CameraService:
    def __init__(self, config: Config):
        self.url = 'http%s://%s:%d' % (
            's' if config.use_https else '',
            config.hostname,
            config.port)

    def get_cameras(self) -> list[Camera]:
        response = requests.get(self.url + '/cameras')

        body = response.json()

        cameras: list[Camera] = []

        for element in body:
            camera = from_dict(Camera, element)
            cameras.append(camera)

        return cameras
