from sems_vision.frame_packet import FramePacketGenerator


def centroid_exit_direction_processor(source: FramePacketGenerator, centroids_key='centroids',
                                      removed_centroids_key='removed_centroids',
                                      left_exit_count_key='left_exit_count', right_exit_count_key='right_exit_count', unknown_exit_count_key='unknown_exit_count'):
    entrypoints: dict[int, tuple[int, int]] = {}
    for packet in source:
        packet.values[left_exit_count_key] = 0
        packet.values[right_exit_count_key] = 0
        packet.values[unknown_exit_count_key] = 0

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
            elif entrypoint[0] <= midpoint <= exitpoint[0]:
                packet.values[right_exit_count_key] += 1
            else:
                packet.values[unknown_exit_count_key] += 1

        yield packet
