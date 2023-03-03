from multiprocessing import Process, Event

from typing import List, Dict

from sems_vision.frame_processors import CamaraProcessing
from flask import Flask
import imutils
import cv2
from sems_vision.config import load_config
from sems_vision.shared_state import SemsStateManager
from sems_vision.socket_io_process import SocketIOProcess
from sems_vision.camera import construct_camera_blueprint


def create_app():
    app = Flask(__name__)

    config = load_config()

    app.config.from_mapping(config)

    manager = SemsStateManager()

    manager.start()

    backend_socket: SocketIOProcess = manager.SocketIOProcess(config)
    frame_store = manager.FrameStore()
    process_handles: List[Process] = manager.list()
    sources: list = manager.list()
    frame_ready_events: Dict[int, Event] = manager.dict()

    # Wait till Camaras Info Received.
    while not backend_socket.get_camera_info():
        pass

    per_camera_info = backend_socket.get_camera_info()

    for index, camera in enumerate(per_camera_info):
        cap = cv2.VideoCapture(camera["source"])
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)

        shape = frame.shape
        dtype = frame._dtype

        cap.release()

        frame_store.register_input_frame(index, shape, dtype)
        frame_store.register_output_frame(index, shape, dtype)

        frame_ready_events[index] = Event()

        process_args = (
            index, camera['v_orientation'], camera['run_distance_violation'], camera['detect_just_left_side'],
            camera['last_record'][0], frame_store, frame_ready_events[index], backend_socket, config)

        process = Process(target=CamaraProcessing, args=process_args)

        process_handles.append(process)
        process.start()

        sources.append(camera["source"])

    process_args = (sources, frame_store, frame_ready_events, config)
    read_process_ref = Process(target=CamaraRead, args=process_args)
    read_process_ref.start()

    camera_bp = construct_camera_blueprint(frame_store)

    from sems_vision.index import bp as index_bp

    app.register_blueprint(index_bp)
    app.register_blueprint(camera_bp)

    return app
