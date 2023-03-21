import time
import numpy as np

from collections import OrderedDict
from time import time
from scipy.spatial.distance import cdist

from sems_vision.centroid_tracker import DetectionCentroidTracker, Centroid
from sems_vision.frame_packet import FramePacketGenerator


def centroid_count_processor(source: FramePacketGenerator, centroids_key='centroids',
                             centroid_count_key='centroid_count',
                             total_centroid_count_key='total_centroid_count'):
    centroid_ids: set[int] = set()
    for packet in source:
        centroids: OrderedDict[int, Centroid] = packet.values[centroids_key]
        for centroid_id, _ in centroids:
            centroid_ids.add(centroid_id)

        packet.values[centroid_count_key] = len(centroid_ids)
        packet.values[total_centroid_count_key] = len(centroids)

        yield packet


def average_centroid_duration_processor(source: FramePacketGenerator, centroids_value_name='centroids',
                                        centroid_count_value_name='centroid_count'):
    average_centroid_duration = 0
    for packet in source:
        centroids = packet.values[centroids_value_name]
        centroid_count = packet.values[centroid_count_value_name]

        for centroid in centroids:
            time_delta = time.time() - centroid.creation_time
            average_centroid_duration = (average_centroid_duration * centroid_count + time_delta) / (
                    centroid_count + 1)
        yield packet


# TODO implement inverse projection mapping
def process_social_distance_violations(social_distance_threshold: int, source: FramePacketGenerator,
                                       centroids_key='centroids',
                                       social_distance_violations_key='social_distance_violations'):
    for packet in source:
        # Ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps).
        centroids = packet.values[centroids_key]
        point_violations: set[int] = set()
        packet.values[social_distance_violations_key] = point_violations
        if len(centroids) >= 2:
            # Extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids.
            centroid_ids = list([centroid_id for centroid_id, _ in centroids])
            centroid_positions = list([centroid.pos for _, centroid in centroids])
            centroids_array = np.array(centroid_positions)

            distances = cdist(centroids_array, centroids_array, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, distances.shape[0]):
                for j in range(i + 1, distances.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if distances[i, j] < social_distance_threshold:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        point_violations.add(centroid_ids[i])
                        point_violations.add(centroid_ids[j])
        yield packet


class CentroidTrackingFrameProcessor:
    """
    CentroidTrackerFrameProcessor associates Detection objects to Centroid objects, maintaining an ID using the given
    centroid object tracking algorithm
    """

    def __init__(self):
        self.frame_shape: tuple[int, int] | None = None

        self.people_in_frame_time_avg = 0
        self.people_count = 0

        self.total_exited_left = 0
        self.total_exited_right = 0

        self.social_distance_threshold = 20
        self.do_distance_violation = True
        self._removed_centroids: dict[int, Centroid] = {}

        self._tracker = DetectionCentroidTracker(max_disappeared_frames=40, max_distance=50,
                                                 on_centroid_removed=self._on_centroid_removed)

    def process(self, source: FramePacketGenerator, detections_value_name='detections',
                centroids_value_name='centroids', removed_centroids_value_name='removed_centroids'):
        for packet in source:
            frame = packet.frame

            if self.frame_shape is None:
                self.frame_shape = frame.shape[:2]

            self._tracker.update(packet.values[detections_value_name])
            packet.values[centroids_value_name] = self._tracker.centroids
            packet.values[removed_centroids_value_name] = self._removed_centroids
            self._removed_centroids = []

            yield packet

    def _on_centroid_removed(self, centroid_id: int, centroid: Centroid):
        self._removed_centroids[centroid_id] = centroid
