from collections import OrderedDict

from sems_vision.frame_packet import FramePacketGenerator
from sems_vision.detection_centroid_tracker import Centroid


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
