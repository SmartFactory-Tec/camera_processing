from camera_processing.MultiprocessDetectionPublisher import MultiprocessDetectionPublisher
from camera_processing.camera import Camera
from camera_processing.camera_service import Direction
from camera_processing.frame_packet import FramePacketGenerator


def detection_exit_publisher_processor(source: FramePacketGenerator, publisher: MultiprocessDetectionPublisher,
                                       camera: Camera, logger, left_exit_count_key='left_exit_count',
                                       right_exit_count_key='right_exit_count',
                                       unknown_exit_count_key='unknown_exit_count'):
    logger = logger.bind(camera_id=camera.id, camera_name=camera.name)
    for packet in source:
        right_exits = packet.values[right_exit_count_key]
        left_exits = packet.values[left_exit_count_key]
        unknown_exits = packet.values[unknown_exit_count_key]

        for _ in range(right_exits):
            publisher.publish_detection(camera.id, packet.timestamp, Direction.RIGHT)

        for _ in range(left_exits):
            publisher.publish_detection(camera.id, packet.timestamp, Direction.LEFT)

        for _ in range(unknown_exits):
            publisher.publish_detection(camera.id, packet.timestamp, Direction.NONE)

        if right_exits > 0:
            logger.info("%d exit(s) to the right detected", right_exits)

        if left_exits > 0:
            logger.info("%d exit(s) to the left detected", left_exits)

        if unknown_exits > 0:
            logger.info("%d exit(s) in unknown direction", unknown_exits)

        yield packet
