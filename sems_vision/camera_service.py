import requests
from dacite import from_dict

from sems_vision.camera import Camera


class InvalidResponseError(Exception):
    pass


class CameraService:
    def __init__(self, hostname: str, port: int, use_https: bool):
        self.url = 'http%s://%s:%d' % (
            's' if use_https else '',
            hostname,
            port)

    def get_cameras(self) -> list[Camera]:
        response = requests.get(self.url + '/cameras')

        body = response.json()

        cameras: list[Camera] = []

        for element in body:
            camera = from_dict(Camera, element)
            cameras.append(camera)

        return cameras
