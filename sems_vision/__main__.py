import signal
from multiprocessing import Process
import requests.exceptions
from .executors import pipeline_executor
from .logger import get_logger
from .config_loader import load_config
from .camera_service import CameraService
from .camera_server_frame_src import camera_server_frame_src
from .processors import YoloV8DetectionProcessor, DetectionCorrelationTrackerProcessor, CentroidTrackerProcessor, \
    centroid_exit_direction_processor
from .processors.detection_exit_logger import detection_exit_logger_processor

logger = get_logger()

config = load_config(logger)

camera_service = CameraService(config.camera_service)

try:
    cameras = camera_service.get_cameras()
except requests.exceptions.ConnectionError:
    logger.error("unable to connect to connect to camera_service at %s", camera_service.url)
    exit(1)

logger.info("initializing processes for %d cameras registered in camera_service", len(cameras),
            camera_service_url=camera_service.url)

main_signal_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
process_handles: list[Process] = []
for camera in cameras:
    log2 = logger.bind(id=camera.id, name=camera.name)
    log2.debug("initializing camera")
    frame_src = camera_server_frame_src('localhost', 3001, camera.id)
    detector = YoloV8DetectionProcessor(0.5, 0.5)
    detecting_processor = detector.process(frame_src, skip_frames=60)
    tracker = DetectionCorrelationTrackerProcessor()
    tracking_processor = tracker.process(detecting_processor)
    centroid_tracker = CentroidTrackerProcessor(max_disappeared_frames=90, max_distance=150)
    centroid_processor = centroid_tracker.process(tracking_processor)
    exit_processor = centroid_exit_direction_processor(centroid_processor)
    detection_logger = detection_exit_logger_processor(exit_processor, camera, logger)
    executor = pipeline_executor(detection_logger)

    pipeline_process = Process(target=executor)

    process_handles.append(pipeline_process)
    pipeline_process.start()
    log2.debug("started camera process")

    signal.signal(signal.SIGINT, main_signal_handler)

try:
    signal.pause()
except KeyboardInterrupt:
    logger.info("exiting...")

for processors in process_handles:
    processors.join()
