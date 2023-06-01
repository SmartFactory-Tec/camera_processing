from math import floor
import cv2
import dlib
from camera_processing.detection import Detection
from camera_processing.frame_packet import FramePacketGenerator
from typing import Optional


class DetectionCorrelationTrackerProcessor:
    def __init__(self):
        self.frame_shape: Optional[tuple[int, int]] = None
        self.trackers: list[dlib.correlation_tracker] = []
        self.detections: list[Detection] = []

    def _reset_tracking(self):
        self.detections = []
        self.trackers = []

    def _register_detections(self, rgb, detections: list[Detection]):
        for detection in detections:
            self.detections.append(detection)
            [x1, y1, x2, y2] = detection.bounding_box
            bb = dlib.rectangle(x1, y1, x2, y2)
            tracker = dlib.correlation_tracker()
            tracker.start_track(rgb, bb)
            self.trackers.append(tracker)

    def process(self, source: FramePacketGenerator, detections_key='detections'):
        for packet in source:
            rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)

            # If there are any detections in the packet, register these as basis for the tracking
            if detections_key in packet.values:
                self._reset_tracking()
                self._register_detections(rgb, packet.values[detections_key])

                yield packet

            for tracker, person in zip(self.trackers, self.detections):
                tracker.update(rgb)
                pos = tracker.get_position()

                x1 = floor(pos.left())
                y1 = floor(pos.top())
                x2 = floor(pos.right())
                y2 = floor(pos.bottom())

                person.bounding_box = (x1, y1, x2, y2)

            packet.values[detections_key] = self.detections

            yield packet
