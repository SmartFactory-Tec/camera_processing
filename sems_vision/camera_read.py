import threading
import time
from multiprocessing import Event

import cv2

from sems_vision.camera import Camera

from sems_vision.shared_state import SharedFrame


def read_camera_to_shared_frame(camera: Camera, shared_frame: SharedFrame):
    print("opening camera ", camera.id)

    while True:
        prev_exec = time.time()
        ret = vs.grab()

        if not ret:
            print("failed reading")
            # I assume this reinitializes the device if it fails
            # TODO nicer health check and reinitialization
            vs = cv2.VideoCapture(camera.connection_string)
            vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            continue

        if not shared_frame.new_frame_is_set():
            _, frame = vs.retrieve()

            shared_nd_array = shared_frame.get_ndarray()
            shared_nd_array[:] = frame
            shared_frame.set_new_frame()
