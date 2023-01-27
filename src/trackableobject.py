import time


class TrackableObject:
    def __init__(self, object_id, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = [centroid]

        # time objected created
        self.creation_time = time.time()
