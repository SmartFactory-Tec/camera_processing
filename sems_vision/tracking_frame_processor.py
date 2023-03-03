import cv2
import dlib

from sems_vision.detection import Detection
from sems_vision.frame_packet import FramePacketGenerator


class CorrelationTrackingFrameProcessor:
    def __init__(self):
        self.frame_shape: tuple[int, int] | None = None
        self.trackers: list[dlib.correlation_tracker] = []
        self.detections: list[Detection] = []
        self.pending_trackers: list[int] = []

    def _reset_tracking(self):
        self.detections = []
        self.pending_trackers = []
        self.trackers = []

    def _register_detections(self, detections: list[Detection]):
        for detection in detections:
            self.detections.append(detection)
            self.pending_trackers.append(len(self.trackers))

            tracker = dlib.correlation_tracker()
            self.trackers.append(tracker)

    def process(self, source: FramePacketGenerator, detections_key='detections'):
        for packet in source:
            # If there are any detections in the packet, register these for future tracking
            if detections_key in packet.values and len(packet.values[detections_key]) != 0:
                self._reset_tracking()
                self._register_detections(packet.values[detections_key])

                yield packet

            # else use the stored detections to generate new ones

            # dlib requires frame to be in RGB
            rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)

            for _ in range(len(self.pending_trackers)):
                idx = self.pending_trackers.pop()

                person = self.detections[idx]

                [x1, y1, x2, y2] = person.bounding_box
                rect = dlib.rectangle(x1, y1, x2, y2)

                self.trackers[idx].start_track(rgb, rect)

            for tracker, person in zip(self.trackers, self.detections):
                tracker.update(rgb)
                pos = tracker.get_position()

                x1 = int(pos.left())
                y1 = int(pos.top())
                x2 = int(pos.right())
                y2 = int(pos.bottom())

                person.bounding_box = (x1, y1, x2, y2)

            packet.values[detections_key] = self.detections

            yield packet
