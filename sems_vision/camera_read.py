import threading
import time
import cv2
import numpy as np
import imutils
from queue import Queue


class CamaraRead:
    MAX_FPS = 34
    MAX_SKIP = 3

    def __init__(self, sources, input_frames, frame_shapes, flags, args):
        self.sources = sources
        self.input_frames = input_frames
        self.frame_shapes = frame_shapes
        self.flags = flags
        self.args = args
        for index in range(len(sources)):
            read_thread = threading.Thread(target=self.main_loop, args=(index,), daemon=True)
            read_thread.start()

        # TODO weird, analyze for possible error
        read_thread.join()

    def main_loop(self, index):
        source = self.sources[index]
        initial_input_frame = self.input_frames[index]
        frame_shape = self.frame_shapes[index]
        flag = self.flags[index]

        print("[INFO] opening video file...", source)
        vs = cv2.VideoCapture(source)
        vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        # why do we need to interpret the frame as a one dimensional array and then reshape it back into its original
        # shape??
        input_frame = np.frombuffer(initial_input_frame, dtype=np.uint8)
        input_frame = input_frame.reshape(frame_shape)

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

            if not flag.value:
                if (q.empty()):
                    input_frame[:] = frame
                    not_taken_counter = 0
                else:
                    input_frame[:] = q.get()
                flag.value = True

            not_taken_counter = (not_taken_counter + 1) % CamaraRead.MAX_SKIP

            while time.time() - prev_exec < 1 / CamaraRead.MAX_FPS:
                # TODO this needlessly consumes CPU time, rework using waits
                pass
