import signal
from multiprocessing import Process, Event
from sems_vision import load_config, CameraService, imshow_pipeline_executor, \
    get_logger, TrackingFrameProcessor, YoloV3DetectingProcessor, Camera
from sems_vision.centroid_tracking_frame_processor import CentroidTrackingFrameProcessor
from sems_vision.multiprocess_frame_srcs import MultiprocessFrameSrcs

logger = get_logger()

config = load_config()
logger.info('loaded config')

camera_service_config = config['camera_service']

camera_service = CameraService(camera_service_config['hostname'], camera_service_config['port'],
                               camera_service_config['use_https'])

logger.info('loading cameras')

cameras = camera_service.get_cameras()

# ignore interrupt signal in all child threads
main_signal_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
process_handles: list[Process] = []
with MultiprocessFrameSrcs(cameras) as frame_srcs:
    for camera in cameras:
        src = frame_srcs.frame_src(camera.id)
        detector = YoloV3DetectingProcessor(0.6, 0.1)
        tracker = TrackingFrameProcessor()
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

    signal.signal(signal.SIGINT, main_signal_handler)
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("exiting...")

for process in process_handles:
    process.join()

# for camera in cameras:
#     vs = cv2.VideoCapture(camera.connection_string)
#     vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
#     captures.append(vs)
