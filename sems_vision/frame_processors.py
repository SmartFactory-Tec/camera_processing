import time
import numpy as np

from collections import OrderedDict
from time import time
from scipy.spatial.distance import cdist

from sems_vision.centroid_tracker import Centroid
from sems_vision.frame_packet import FramePacketGenerator


def centroid_exit_direction_processor(source: FramePacketGenerator, centroids_key='centroids',
                                      removed_centroids_key='removed_centroids',
                                      left_exit_count_key='left_exit_count', right_exit_count_key='right_exit_count'):
    entrypoints: dict[int, tuple[int, int]] = {}
    for packet in source:
        packet.values[left_exit_count_key] = 0
        packet.values[right_exit_count_key] = 0

        frame_shape = packet.frame.shape

        midpoint = frame_shape[1] // 2

        for centroid_id, centroid in packet.values[centroids_key].items():
            if centroid_id not in entrypoints:
                entrypoints[centroid_id] = centroid.pos

        for centroid_id, centroid in packet.values[removed_centroids_key].items():
            entrypoint = entrypoints[centroid_id]
            exitpoint = centroid.pos

            del entrypoints[centroid_id]

            if entrypoint[0] >= midpoint >= exitpoint[0]:
                packet.values[left_exit_count_key] += 1
                print("Exited left!")
            elif entrypoint[0] <= midpoint <= exitpoint[0]:
                packet.values[right_exit_count_key] += 1
                print("exited right!")
            else:
                print(f"exited from {entrypoint[0]} to {exitpoint[0]}")

        yield packet


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
