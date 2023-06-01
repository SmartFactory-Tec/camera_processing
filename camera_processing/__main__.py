import signal
from multiprocessing import Process, Event
import requests.exceptions

from .MultiprocessDetectionPublisher import MultiprocessDetectionPublisher
from .executors import pipeline_executor, imshow_pipeline_executor
from .logger import get_logger
from .config_loader import load_config
from .camera_service import CameraService
from .camera_streamer_frame_src import camera_streamer_frame_src
from .processors import YoloV8DetectionProcessor, DetectionCorrelationTrackerProcessor, CentroidTrackerProcessor, \
    centroid_exit_direction_processor
from .processors.detection_exit_logger import detection_exit_logger_processor
from .processors.detection_publisher import detection_exit_publisher_processor

logger = get_logger()

config = load_config(logger)

camera_service = CameraService(config.camera_service)
detection_publisher = MultiprocessDetectionPublisher(camera_service)

try:
    cameras = camera_service.get_cameras()
except requests.exceptions.ConnectionError:
    logger.error("unable to connect to connect to camera_service at %s", camera_service.url)
    exit(1)

logger.info("initializing processes for %d cameras registered in camera_service", len(cameras),
            camera_service_url=camera_service.url)

process_handles: list[Process] = []

stop_event = Event()

for camera in cameras:
    log2 = logger.bind(id=camera.id, name=camera.name)
    log2.debug("initializing camera")
    frame_src = camera_streamer_frame_src(config.camera_streamer.hostname, config.camera_streamer.port, camera.id,
                                          False, log2)
    detector = YoloV8DetectionProcessor(0.7, 0.6)
    detecting_processor = detector.process(frame_src, skip_frames=60)
    tracker = DetectionCorrelationTrackerProcessor()
    tracking_processor = tracker.process(detecting_processor)
    centroid_tracker = CentroidTrackerProcessor(max_disappeared_frames=90, max_distance=150)
    centroid_processor = centroid_tracker.process(tracking_processor)
    exit_processor = centroid_exit_direction_processor(centroid_processor)
    publisher_processor = detection_exit_publisher_processor(exit_processor, detection_publisher, camera, logger)
    executor = pipeline_executor(publisher_processor)

    pipeline_process = Process(target=executor, daemon=True)

    process_handles.append(pipeline_process)
    pipeline_process.start()
    log2.debug("started camera process")

try:
    while True:
        detection_publisher.update()
except KeyboardInterrupt:
    logger.info("stopping processes...")