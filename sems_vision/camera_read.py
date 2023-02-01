import threading
import time
from multiprocessing import Event

import cv2
import imutils
from queue import Queue

from typing import Dict

from sems_vision.frame_store import FrameStore


class CamaraRead:
    MAX_FPS = 34
    MAX_SKIP = 3

    def __init__(self, sources, frame_store: FrameStore, frame_ready_events: Dict[int, Event], config):
        self.sources = sources
        self.frame_store = frame_store
        self.frame_ready_events = frame_ready_events
        self.config = config
        for index in range(len(sources)):
            read_thread = threading.Thread(target=self.main_loop, args=(index,), daemon=True)
            read_thread.start()

        # TODO weird, analyze for possible error
        read_thread.join()

    def main_loop(self, frame_id):
        source = self.sources[frame_id]

        print("[INFO] opening video file...", source)
        vs = cv2.VideoCapture(source)
        vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        input_frame = self.frame_store.get_input_frame_ref(frame_id)
        flag: Event = self.frame_ready_events[frame_id]

        q = Queue(maxsize=0)
        not_taken_counter = 0

        while True:
            prev_exec = time.time()
            status, frame = vs.read()

            if not status:
                # I assume this reinitializes the device if it fails
                # TODO nicer health check and reinitialization
                vs = cv2.VideoCapture(source)
                vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                continue

            frame = imutils.resize(frame, width=500)

            if not_taken_counter == 0:
                q.put(frame)
            if not flag.is_set():
                if (q.empty()):
                    input_frame[:] = frame
                    not_taken_counter = 0
                else:
                    input_frame[:] = q.get()
                flag.set()

            not_taken_counter = (not_taken_counter + 1) % CamaraRead.MAX_SKIP

            while time.time() - prev_exec < 1 / CamaraRead.MAX_FPS:
                # TODO this needlessly consumes CPU time, rework using waits
                pass
