import datetime

import requests
from dacite import from_dict
from dataclasses import dataclass
from .camera import Camera
from enum import StrEnum


class InvalidResponseError(Exception):
    pass


class Direction(StrEnum):
    LEFT = 'left'
    RIGHT = 'right'
    NONE = 'none'


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

    def post_detection(self, camera_id: int, detection_date: datetime.datetime, target_direction: Direction):
        response = requests.post(self.url + f'/cameras/{camera_id}/personDetections', json={
            'camera_id': camera_id,
            'detection_date': detection_date.astimezone().isoformat(),
            'target_direction': target_direction
        })

        if response.status_code != 201:
            raise RuntimeError(f'could not post detection to camera_service, with status code {response.status_code}')
