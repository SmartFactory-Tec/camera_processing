from collections import OrderedDict
from multiprocessing import Event, Process, Manager
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from threading import Thread

import cv2
import time
import numpy as np
from scipy.spatial.distance import cdist

from sems_vision.camera import Camera
from sems_vision.centroid_tracker import DetectionCentroidTracker, Centroid
from sems_vision.frame_packet import FramePacket, FramePacketGenerator
from time import time

from sems_vision.shared_state import SharedFrame


def imshow_pipeline_executor(source: FramePacketGenerator):
    def executor():
        nonlocal source
        for packet in source:
            cv2.imshow('imshow_executor', packet.frame)
            # centroid_count = packet.values['centroid_count']
            # if centroid_count > 0:
            #     print(centroid_count)

            key = cv2.waitKey(25)
            if key == ord('q'):
                break

    return executor


class MultiprocessFrameSrcs:
    def __init__(self, cameras: list[Camera]):
        self._manager = SharedMemoryManager()

        self._cameras = cameras
        self._shared_mems: dict[int, SharedMemory] = {}
        self._shapes: dict[int, tuple] = {}
        self._dtypes: dict[int, type] = {}
        self._new_frame_events: dict[int, Event] = {}

        self._stop_event = Event()

        self._process_handle = Process(target=MultiprocessFrameSrcs._start_grabbers,
                                       args=(
                                           self._cameras, self._shared_mems, self._shapes, self._dtypes,
                                           self._new_frame_events, self._stop_event))

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

            ok, frame = capture.read()

            if not ok:
                print('unable to grab frame from camera with id: %d' % camera.id)
                return

            self._shared_mems[camera.id] = self._manager.SharedMemory(frame.nbytes)
            self._shapes[camera.id] = frame.shape
            self._dtypes[camera.id] = frame.dtype
            self._new_frame_events[camera.id] = Event()

        self._process_handle.start()

    def shutdown(self):
        self._stop_event.set()
        self._process_handle.join()

        self._shared_mems = {}
        self._shapes = {}
        self._dtypes = {}
        self._new_frame_events = {}

        self._manager.shutdown()

    @staticmethod
    def _start_grabbers(cameras: list[Camera], shared_mems: list[SharedMemory], shapes: list[tuple], dtypes: list[type],
                        new_frame_events: list[Event], stop_event: Event):
        grabber_threads: list[Thread] = []

        for camera in cameras:
            camera_id = camera.id
            thread = Thread(target=MultiprocessFrameSrcs.start_frame_grabber,
                            args=[camera, shared_mems[camera_id], shapes[camera_id], dtypes[camera_id],
                                  new_frame_events[camera_id], stop_event])
            grabber_threads.append(thread)
            thread.start()

        stop_event.wait()

        for thread in grabber_threads:
            thread.join()

    @staticmethod
    def start_frame_grabber(camera: Camera, shared_mem: SharedMemory, shape: tuple, dtype: type,
                            new_frame_event: Event, stop_event: Event):
        capture = cv2.VideoCapture(camera.connection_string)
        while not stop_event.is_set():
            ret = capture.grab()

            if not ret:
                print("unable to grab frame from camera with id: %d" % camera.id)
                break

            shared_frame = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)

            if not new_frame_event.is_set():
                _, frame = capture.retrieve()

                shared_frame[:] = frame[:]
                new_frame_event.set()

    def frame_src(self, camera_id: int):
        shared_frame = np.ndarray(shape=self._shapes[camera_id], dtype=self._dtypes[camera_id],
                                  buffer=self._shared_mems[camera_id].buf)

        new_frame_event = self._new_frame_events[camera_id]
        while True:
            new_frame_event.wait()

            frame = shared_frame.copy()

            new_frame_event.clear()

            yield FramePacket(frame)


def multithreaded_frame_src(camera: Camera, manager: SharedMemoryManager):
    # get frame dimensions and dtype
    cap = cv2.VideoCapture(camera.connection_string)
    ret, frame = cap.read()
    cap.release()

    shape = frame.shape
    dtype = frame.dtype
    new_frame_event = Event()
    shared_mem = manager.SharedMemory(size=frame.nbytes)

    # def start_frame_grabber():


def shared_memory_src(shared_mem: SharedMemory, shape: tuple, dtype: type, new_frame_event: Event):
    frame = np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)
    while True:
        new_frame_event.wait()

        frame_copy = frame.copy()

        new_frame_event.clear()

        yield FramePacket(frame_copy)


def centroid_count_processor(source: FramePacketGenerator, centroids_key='centroids',
                             centroid_count_key='centroid_count',
                             total_centroid_count_key='total_centroid_count'):
    centroid_ids: set[int] = set()
    for packet in source:
        centroids: OrderedDict[int, Centroid] = packet.values[centroids_key]
        for centroid_id, _ in centroids:
            centroid_ids.add(centroid_id)

        packet.values[centroid_count_key] = len(centroid_ids)
        packet.values[total_centroid_count_key] = len(centroids)

        yield packet


def average_centroid_duration_processor(source: FramePacketGenerator, centroids_value_name='centroids',
                                        centroid_count_value_name='centroid_count'):
    average_centroid_duration = 0
    for packet in source:
        centroids = packet.values[centroids_value_name]
        centroid_count = packet.values[centroid_count_value_name]

        for centroid in centroids:
            time_delta = time.time() - centroid.creation_time
            average_centroid_duration = (average_centroid_duration * centroid_count + time_delta) / (
                    centroid_count + 1)
        yield packet


# TODO implement inverse projection mapping
def process_social_distance_violations(social_distance_threshold: int, source: FramePacketGenerator,
                                       centroids_key='centroids',
                                       social_distance_violations_key='social_distance_violations'):
    for packet in source:
        # Ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps).
        centroids = packet.values[centroids_key]
        point_violations: set[int] = set()
        packet.values[social_distance_violations_key] = point_violations
        if len(centroids) >= 2:
            # Extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids.
            centroid_ids = list([centroid_id for centroid_id, _ in centroids])
            centroid_positions = list([centroid.pos for _, centroid in centroids])
            centroids_array = np.array(centroid_positions)

            distances = cdist(centroids_array, centroids_array, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, distances.shape[0]):
                for j in range(i + 1, distances.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if distances[i, j] < social_distance_threshold:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        point_violations.add(centroid_ids[i])
                        point_violations.add(centroid_ids[j])
        yield packet


class CentroidTrackingFrameProcessor:
    """
    CentroidTrackerFrameProcessor associates Detection objects to Centroid objects, maintaining an ID using the given
    centroid object tracking algorithm
    """

    def __init__(self):
        self.frame_shape: tuple[int, int] | None = None

        self.people_in_frame_time_avg = 0
        self.people_count = 0

        self.total_exited_left = 0
        self.total_exited_right = 0

        self.social_distance_threshold = 20
        self.do_distance_violation = True
        self._removed_centroids: dict[int, Centroid] = {}

        self._tracker = DetectionCentroidTracker(max_disappeared_frames=40, max_distance=50,
                                                 on_centroid_removed=self._on_centroid_removed)

    def process(self, source: FramePacketGenerator, detections_value_name='detections',
                centroids_value_name='centroids', removed_centroids_value_name='removed_centroids'):
        for packet in source:
            frame = packet.frame

            if self.frame_shape is None:
                self.frame_shape = frame.shape[:2]

            self._tracker.update(packet.values[detections_value_name])
            packet.values[centroids_value_name] = self._tracker.centroids
            packet.values[removed_centroids_value_name] = self._removed_centroids
            self._removed_centroids = []

            yield packet

    def _on_centroid_removed(self, centroid_id: int, centroid: Centroid):
        self._removed_centroids[centroid_id] = centroid


class CamaraProcessing:

    def gen_frames(self):
        # Loop over frames from the video stream.

        while True:
            # Counter for social distance violations.
            self.distance_violation_count = 0

            # TODO wastes CPU time, rework with waits
            # Grab the next frame if available.
            self.frame_ready_event.wait()
            self.frame_ready_event.clear()

            input_frame = self.frame_store.get_input_frame_ref(self.camera_id)

    def end_process(self):
        # close any open windows
        cv2.destroyAllWindows()
