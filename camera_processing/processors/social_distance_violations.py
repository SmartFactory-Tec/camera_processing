import numpy as np

from scipy.spatial.distance import cdist

from camera_processing.frame_packet import FramePacketGenerator


# TODO implement inverse projection mapping
def social_distance_violations_processor(social_distance_threshold: int, source: FramePacketGenerator,
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
