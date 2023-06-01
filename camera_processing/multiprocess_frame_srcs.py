from multiprocessing import Condition, Event, Process, Lock
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from threading import Thread

import cv2
import numpy as np

from camera_processing import Camera, FramePacket


class MultiprocessFrameSrcs:
    def __init__(self, cameras: list[Camera]):
        self._manager = SharedMemoryManager()

        self._cameras = cameras

        self._shared_mems: dict[int, SharedMemory] = {}
        self._shapes: dict[int, tuple] = {}
        self._dtypes: dict[int, type] = {}

        self._frame_conditions: dict[int, Condition] = {}

        self._stop_event = Event()

        self._process_handle = Process(target=MultiprocessFrameSrcs._start_captures,
                                       args=(
                                           self._cameras, self._shared_mems, self._shapes, self._dtypes,
                                           self._frame_conditions, self._stop_event))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def start(self):
        self._manager.start()

        for camera in self._cameras:
            print('initialized camera with id: %d' % camera.id)
            capture = cv2.VideoCapture(camera.connection_string)

            ok, reference_frame = capture.read()

            if not ok:
                print('unable to grab frame from camera with id: %d' % camera.id)
                return

            self._shared_mems[camera.id] = self._manager.SharedMemory(reference_frame.nbytes)
            self._dtypes[camera.id] = reference_frame.dtype
            self._shapes[camera.id] = reference_frame.shape
            self._frame_conditions[camera.id] = Condition()

        self._process_handle.start()

    def shutdown(self):
        self._stop_event.set()
        self._process_handle.join()

        self._shared_mems = {}
        self._shapes = {}
        self._dtypes = {}
        self._frame_conditions = {}

        self._manager.shutdown()

    @staticmethod
    def _start_captures(cameras: list[Camera], shared_mems: dict[int, SharedMemory], shapes: dict[int, tuple],
                        dtypes: dict[int, type], frame_conditions: list[Condition], stop_event: Event):
        grabber_threads: list[Thread] = []
        retriever_threads: list[Thread] = []

        for camera in cameras:
            capture = cv2.VideoCapture(camera.connection_string)
            frame_lock = Lock()

            shape = shapes[camera.id]
            dtype = dtypes[camera.id]
            shared_mem = shared_mems[camera.id]
            frame_condition = frame_conditions[camera.id]

            grabber_thread = Thread(target=MultiprocessFrameSrcs.start_frame_grabber,
                                    args=[capture, frame_lock, stop_event])

            shared_frame = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
            retriever_thread = Thread(target=MultiprocessFrameSrcs.start_frame_retriever,
                                      args=[capture, shared_frame, frame_condition, frame_lock, stop_event])
            grabber_threads.append(grabber_thread)
            retriever_threads.append(retriever_thread)

            grabber_thread.start()
            retriever_thread.start()

        stop_event.wait()

        for thread in grabber_threads:
            thread.join()

        for thread in retriever_threads:
            thread.join()

    @staticmethod
    def start_frame_grabber(capture: cv2.VideoCapture,
                            frame_lock: Lock, stop_event: Event):
        while not stop_event.is_set():
            with frame_lock:
                ret = capture.grab()
                if not ret:
                    print("error grabbing frame")
                    break

    @staticmethod
    def start_frame_retriever(capture: cv2.VideoCapture, shared_frame: np.ndarray,
                              new_frame_condition: Condition,
                              frame_lock: Lock, stop_event: Event):
        while not stop_event.is_set():
            with frame_lock:
                _, frame = capture.retrieve()

            with new_frame_condition:
                shared_frame[:] = frame[:]
                new_frame_condition.notify_all()

    def frame_src(self, camera_id: int, shape: tuple | None = None):
        shared_frame = np.ndarray(shape=self._shapes[camera_id], dtype=self._dtypes[camera_id],
                                  buffer=self._shared_mems[camera_id].buf)

        # this allows pickling of the generator object
        frame_condition = self._frame_conditions[camera_id]
        stop_event = self._stop_event

        while not stop_event.is_set():
            with frame_condition:
                frame_condition.wait()

                if shape is None:
                    frame = cv2.copy(shared_frame)
                else:
                    frame = cv2.resize(shared_frame, shape)

            yield FramePacket(frame)
