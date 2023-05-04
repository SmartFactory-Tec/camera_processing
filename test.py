import cv2
from camera_server_frame_src import camera_server_frame_src
from sems_vision.executors import imshow_pipeline_executor
from sems_vision.yolov5_detecting_processor import YoloV5DetectingProcessor


def main():
    frame_src = camera_server_frame_src('10.22.244.185', 3002, 4)

    detector = YoloV5DetectingProcessor(0.5, 0.5)
    detecting_processor = detector.process(frame_src)

    executor = imshow_pipeline_executor(detecting_processor)

    executor()


# def main():
#     camera = Camera(0, 'cam1', 'rtsp://admin:L2321800@10.22.240.56:80/cam/realmonitor?channel=1&subtype=0&proto=Onvif',
#                     'test', 0)
#
#     main_signal_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
#
#     with MultiprocessFrameSrcs([camera]) as frame_srcs:
#         frame_src = frame_srcs.frame_src(camera.id, (500, 500))
#
#         executor = imshow_pipeline_executor(frame_src)
#
#         process = Process(target=executor)
#
#         process.start()
#
#         signal.signal(signal.SIGINT, main_signal_handler)
#
#         try:
#             signal.pause()
#         except KeyboardInterrupt:
#             pass
#
#     process.join()


if __name__ == '__main__':
    main()
