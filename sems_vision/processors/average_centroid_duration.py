import time
from sems_vision.frame_packet import FramePacketGenerator


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
