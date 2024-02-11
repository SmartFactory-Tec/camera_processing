from datetime import datetime
from camera_processing.camera_service import CameraService, Direction
from multiprocessing import Queue
from dataclasses import dataclass


@dataclass
class PersonDetection:
    camera_id: int
    detection_date: datetime
    target_direction: Direction


class PersonDetectionPublisher:
    def __init__(self, camera_service: CameraService):
        self.__camera_service = camera_service
        self.__publishing_queue: Queue[PersonDetection] = Queue()

    def update(self):
        while not self.__publishing_queue.empty():
            detection = self.__publishing_queue.get()
            self.__camera_service.post_detection(detection.camera_id, detection.detection_date,
                                                 detection.target_direction)

    def publish_detection(self, camera_id: int, detection_date: datetime, target_direction: Direction):
        self.__publishing_queue.put(PersonDetection(camera_id, detection_date, target_direction))
