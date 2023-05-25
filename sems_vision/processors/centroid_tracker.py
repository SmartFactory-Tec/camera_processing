from collections import OrderedDict
from typing import Optional
from sems_vision.detection_centroid_tracker import Centroid, DetectionCentroidTracker
from sems_vision.frame_packet import FramePacketGenerator


class CentroidTrackerProcessor:
    """
    CentroidTrackerFrameProcessor associates Detection objects to Centroid objects, maintaining an ID using the given
    centroid object tracking algorithm
    """

    def __init__(self, max_disappeared_frames=50, max_distance=100):
        self.frame_shape: Optional[tuple[int, int]] = None

        self.people_in_frame_time_avg = 0
        self.people_count = 0

        self.total_exited_left = 0
        self.total_exited_right = 0

        self.social_distance_threshold = 20
        self.do_distance_violation = True
        self._removed_centroids: OrderedDict[int, Centroid] = OrderedDict()

        self._tracker = DetectionCentroidTracker(max_disappeared_frames=max_disappeared_frames,
                                                 max_distance=max_distance,
                                                 on_centroid_removed=self._on_centroid_removed)

    def process(self, source: FramePacketGenerator, detections_key='detections',
                centroids_key='centroids', removed_centroids_key='removed_centroids'):
        for packet in source:
            frame = packet.frame

            if self.frame_shape is None:
                self.frame_shape = frame.shape[:2]

            self._removed_centroids = OrderedDict()

            self._tracker.update(packet.values[detections_key])
            packet.values[centroids_key] = self._tracker.centroids

            packet.values[removed_centroids_key] = self._removed_centroids

            yield packet

    def _on_centroid_removed(self, centroid_id: int, centroid: Centroid):
        self._removed_centroids[centroid_id] = centroid
