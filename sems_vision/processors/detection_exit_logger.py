from sems_vision.frame_packet import FramePacketGenerator
from sems_vision.camera import Camera


def detection_exit_logger_processor(source: FramePacketGenerator, camera: Camera, logger, left_exit_count_key='left_exit_count',
                                    right_exit_count_key='right_exit_count', unknown_exit_count_key='unknown_exit_count'):
    logger = logger.bind(camera_id=camera.id, camera_name=camera.name)
    for packet in source:
        right_exits = packet.values[right_exit_count_key]
        left_exits = packet.values[left_exit_count_key]
        unknown_exits = packet.values[unknown_exit_count_key]

        if right_exits > 0:
            logger.info("%d exit(s) to the right detected", right_exits)

        if left_exits > 0:
            logger.info("%d exit(s) to the left detected", left_exits)

        if unknown_exits > 0:
            logger.info("%d exit(s) in unknown direction", unknown_exits)

        yield packet
