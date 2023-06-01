from typing import Optional, Union

from math import floor

import numpy as np
from time import time
from dataclasses import dataclass, field
from scipy.spatial import distance as dist
from collections import OrderedDict
from collections.abc import Callable

from camera_processing.detection import Detection


@dataclass
class Centroid:
    pos: tuple[int, int]
    disappeared_frames: int
    creation_time: float = field(default_factory=time)


def detection_centroid(detection: Detection):
    [x1, y1, x2, y2] = detection.bounding_box
    x = floor((x1 + x2) / 2.0)
    y = floor((y1 + y2) / 2.0)

    return Centroid((x, y), 0)


class DetectionCentroidTracker:
    def __init__(self, max_disappeared_frames: int = 50, max_distance: float = 50,
                 on_centroid_removed: Union[Callable[[int, Optional[Centroid]], None], None] = None):
        self._next_object_id = 0  # used to generate centroid ids
        self.centroids: OrderedDict[int, Centroid] = OrderedDict()

        self.max_disappeared_frames: int = max_disappeared_frames

        self.max_distance: float = max_distance

        # callback function to call when a centroid is removed
        self.remove_action: Optional[Callable[int, Centroid]] = on_centroid_removed

    def _handle_missing_centroid(self, centroid_id: int):
        centroid = self.centroids[centroid_id]
        centroid.disappeared_frames += 1

        if centroid.disappeared_frames > self.max_disappeared_frames:
            del self.centroids[centroid_id]

            if self.remove_action is not None:
                self.remove_action(centroid_id, centroid)

    def _register_centroid(self, centroid: Centroid):
        self.centroids[self._next_object_id] = centroid

        self._next_object_id += 1

    def update(self, detections: list[Detection]):
        """
        Update the tracker's registered centroids from a list of detections
        :param detections: list of detections to use for their centroids
        """

        # if there are no detections then all centroids are missing
        if len(detections) == 0:
            # copy the keys so that we can mutate the map
            centroid_ids = list(self.centroids.keys())
            for centroid_id in centroid_ids:
                self._handle_missing_centroid(centroid_id)
            return

        # convert the incoming detections into new centroids
        new_centroids = list(detection_centroid(detection) for detection in detections)
        new_centroids_pos = list(centroid.pos for centroid in new_centroids)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.centroids) == 0:
            for detection in detections:
                self._register_centroid(detection_centroid(detection))
            return

        # otherwise, are currently tracking objects, so we need to
        # try to match the input centroids to existing object centroids

        # list to convert from centroid list indices to their actual stored IDs
        centroid_ids = list(self.centroids.keys())
        centroids_pos = list(centroid.pos for centroid in self.centroids.values())

        # matrix that contains all the distances for each combination of old centroid and new centroid
        distances = dist.cdist(np.array(centroids_pos), np.array(new_centroids_pos))

        # obtain the old centroid's indices sorted by how close they are to a new centroid, to ensure that the matching
        # uses the closest pair possible
        min_dist_centroids = distances.min(axis=1).argsort()

        # get the new centroid's indices for the previous step
        closest_new_centroids = distances.argmin(axis=1)[min_dist_centroids]

        # store the centroids that have been matched
        matched_centroids: set[int] = set()
        matched_new_centroids: set[int] = set()

        # loop over the combination old and new centroids
        for (centroid_idx, new_centroid_idx) in zip(min_dist_centroids, closest_new_centroids):
            # if either of the centroids have already been matched, ignore
            if centroid_idx in matched_centroids or new_centroid_idx in matched_new_centroids:
                continue
            # if the distance between centroids is greater than
            # the maximum distance, do not associate the two
            # centroids to the same object
            if distances[centroid_idx, new_centroid_idx] > self.max_distance:
                continue
            # otherwise, grab the centroid ID for the centroid, and set its value to the new centroid
            centroid_id = centroid_ids[centroid_idx]
            self.centroids[centroid_id] = new_centroids[new_centroid_idx]

            # mark them as matched
            matched_centroids.add(centroid_idx)
            matched_new_centroids.add(new_centroid_idx)

        # get the centroids that remain unmatched
        unused_centroids: set[int] = set(range(0, distances.shape[0])).difference(matched_centroids)
        unused_new_centroids: set[int] = set(range(0, distances.shape[1])).difference(matched_new_centroids)

        # if old centroids are unused, they have disappeared
        for row in unused_centroids:
            centroid_id = centroid_ids[row]
            self._handle_missing_centroid(centroid_id)

        # if new centroids are unused, they must be registered as new objects
        for new_centroid_idx in unused_new_centroids:
            self._register_centroid(new_centroids[new_centroid_idx])
