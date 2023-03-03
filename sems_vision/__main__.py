import signal
import sys
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import structlog
from multiprocessing import Process, Event

import cv2
import imutils
import requests

from sems_vision.camera import Camera
from sems_vision import load_config, SemsStateManager, SocketIOProcess, CamaraProcessing
from sems_vision.camera_service import CameraService
from sems_vision.frame_processors import CentroidTrackingFrameProcessor, centroid_count_processor, \
    imshow_pipeline_executor, shared_memory_src, MultiprocessFrameSrcs
from sems_vision.logger import get_logger
from sems_vision.shared_state import SharedFrame
from sems_vision.tracking_frame_processor import CorrelationTrackingFrameProcessor
from sems_vision.yolov3_person_detector import PersonDetectingFrameProcessor

logger = get_logger()

config = load_config()
logger.info('loaded config')

camera_service_config = config['camera_service']

camera_service = CameraService(camera_service_config['hostname'], camera_service_config['port'],
                               camera_service_config['use_https'])
logger.info('loading cameras')

cameras = camera_service.get_cameras()

# TODO this is only for testing
for camera in cameras:
    if camera.id != 4:
        continue

    cameras = [camera]
    break

with MultiprocessFrameSrcs(cameras) as frame_srcs:
    process_handles: list[Process] = []

    for camera in cameras:
        src = frame_srcs.frame_src(camera.id)
        detector = PersonDetectingFrameProcessor(0.6, 0.1)
        tracker = CorrelationTrackingFrameProcessor()
        centroidTracker = CentroidTrackingFrameProcessor()

        detect = detector.process(src)
        # track = tracker.process(detect)
        # centroidTrack = centroidTracker.process(track)
        # count = centroid_count_processor(centroidTrack)

        executor = imshow_pipeline_executor(detect)

        pipeline_process = Process(target=executor)

        process_handles.append(pipeline_process)
        pipeline_process.start()
        print("added camera")

    signal.pause()


    # for camera in cameras:
    #     vs = cv2.VideoCapture(camera.connection_string)
    #     vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    #     captures.append(vs)
