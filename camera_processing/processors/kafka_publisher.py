import json
import time

from camera_processing.camera import Camera
from camera_processing.detection import Detection, BoundingBox
from camera_processing.frame_packet import FramePacketGenerator
from pykafka import KafkaClient


def kafka_publisher(source: FramePacketGenerator,
                    camera: Camera, topic_name: str, kafka_hosts: list[str], detections_key='detections',
                    skip_frames=10):
    client = KafkaClient(hosts=','.join(kafka_hosts))
    topic = client.topics[topic_name]
    prev_no_detections = False
    frame_counter = 0
    with topic.get_producer() as producer:
        for packet in source:
            frame_counter += 1
            frame_counter %= skip_frames
            if frame_counter != 0:
                yield packet
                continue

            detections: list[Detection] = packet.values[detections_key]
            print(len(detections))
            if not prev_no_detections or len(detections) > 0:
                prev_no_detections = len(detections) == 0

                serializable_detections = list(map(lambda d: {
                    'confidence': d.confidence,
                    'bounding_box': tuple(d.bounding_box)
                }, detections))

                message = json.dumps(serializable_detections)

                producer.produce(bytes(message, 'utf-8'), partition_key=bytes(f'{camera.id}', 'utf-8'))

            yield packet
