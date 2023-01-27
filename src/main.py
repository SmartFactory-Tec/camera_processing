# USAGE
# python3 main.py
import os
# Import Dependencies
from multiprocessing import Process, Array, Value
from multiprocessing.managers import BaseManager
from src.centroidtracker import CentroidTracker
from src.trackableobject import TrackableObject
from flask import Flask, render_template, Response
from imutils.video import FPS
from scipy.spatial import distance as dist
from queue import Queue
import numpy as np
import imutils
import time
import dlib
import cv2
import threading
import ctypes
import math
import socketio
import socket

ARGS = {
    "CAMARAIDS": [8],
    "BACK_ENDPOINT": "http://localhost:3001/",
    "NGROK_AVAILABLE": False,
    "GPU_AVAILABLE": True,
    "FORWARD_CAMERA": False,
    "VERBOSE": False,
    "CONFIDENCE": 0.3,
    "SKIP_FRAMES": 25,
}

app = Flask(__name__)


class SocketIOProcess:
    sio = socketio.Client()

    def __init__(self, args):
        self.args = args
        self.camera_ids = self.args["CAMARAIDS"]
        self.camera_count = len(self.camera_ids)
        self.per_camera_info = []
        self.is_connected = False
        self.has_camera_info = False

        self.sio.on('connect', self.connect)
        self.sio.on('disconnect', self.disconnect)
        self.sio.on('visionInit', self.init_vision)
        self.sio.connect(self.args["BACK_ENDPOINT"])

    def connect(self):
        print('Connected')
        self.is_connected = True
        self.sio.emit('visionInit', self.camera_ids)

    def disconnect(self):
        print('Disconnected')
        self.is_connected = False
        self.has_camera_info = False
        self.per_camera_info = []

    def wait(self):
        self.sio.wait()

    def init_vision(self, per_camera_info):
        self.per_camera_info = per_camera_info
        self.has_camera_info = True
        print('CamaraInfo ', per_camera_info)

    def get_camera_info(self, camera_id=None):
        if not self.has_camera_info:
            return False

        if not camera_id:
            return self.per_camera_info

        return self.per_camera_info[camera_id]

    def send_camera_data(self, camera_id, data):
        if self.is_connected:
            self.sio.emit('visionPost', data=(
                self.per_camera_info[camera_id]['camera_id'],
                data["in_direction"],
                data["out_direction"],
                data["counter"],
                data["social_distancing_v"],
                data["in_frame_time_avg"],
                data["fps"],
            ))

    def set_camera_url(self, camera_id):
        if self.is_connected:
            if self.args["NGROK_AVAILABLE"] and self.args["FORWARD_CAMERA"]:
                endpoint = 'http://sems.ngrok.io/camara/'
            elif self.args["FORWARD_CAMERA"]:
                endpoint = 'http://' + socket.getfqdn() + ':8080/camara/'
            else:
                endpoint = ''

            self.sio.emit('updateCamara', data=(
                self.per_camera_info[camera_id]['camera_id'],
                endpoint + str(camera_id)
            ))


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


class CamaraProcessing:
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLACK = (0, 0, 0)
    social_distance_threshold = 90

    CLASSES = None

    script_dir = os.path.dirname(__file__)
    rel_path = "../models/people/coco.names"
    abs_file_path = os.path.join(script_dir, rel_path)

    with open(abs_file_path, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]

    # why is this init so damn long
    def __init__(self, id, v_orientation, run_distance_violation, detect_just_left_side, last_record, input_frame,
                 output_frame, frame_shape, flag, socket_manager, args):
        self.id = id
        self.v_orientation = v_orientation
        self.run_distance_violation = run_distance_violation
        self.detect_just_left_side = detect_just_left_side
        self.camera_id = "Camara" + str(self.id)
        self.socket_manager = socket_manager
        self.socket_manager.set_camera_url(self.id)
        self.args = args

        # Load Model
        self.net = cv2.dnn.readNetFromDarknet('models/people/yolov3.cfg', 'models/people/yolov3.weights')
        if self.args["GPU_AVAILABLE"]:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Get the output layer names of the model
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # initialize the video writer (we'll instantiate later if need be)
        self.writer = None

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        # Non maxima supression threshold
        self.NMS_THRESH = 0.3

        # People in Frame - Time Average
        self.people_in_frame_time_avg = 0
        self.people_in_frame_count = 0

        # Instantiate our centroid tracker, initialize a list to store
        # each of our dlib correlation trackers and a dictionary to
        # map each unique object ID to a TrackableObject
        self.trackers = []
        self.trackable_objects = {}

        # Instantiate custom remove_action for centroid tracker.
        def remove_action(object_id):
            def get_avg(prev_avg, x, n):
                return (prev_avg * n + x) / (n + 1)

            trackable_object = self.trackable_objects[object_id]
            self.people_in_frame_time_avg = get_avg(self.people_in_frame_time_avg,
                                                    time.time() - trackable_object.startTime,
                                                    self.people_in_frame_count)
            self.people_in_frame_count += 1

            def determine_direction(self, to):
                if self.v_orientation:
                    x = [c[0] for c in to.centroids]

                    if x[len(x) - 1] < (self.W // 2) < x[0]:
                        self.total_going_in += 1

                    elif x[len(x) - 1] > (self.W // 2) > x[0]:
                        self.total_going_out += 1
                else:
                    y = [c[1] for c in to.centroids]

                    if y[len(y) - 1] < (self.H // 2) < y[0]:
                        self.total_going_in += 1
                    elif y[len(y) - 1] > (self.H // 2) > y[0]:
                        self.total_going_out += 1

            determine_direction(self, trackable_object)
            self.overpass_post_condition = True
            del self.trackable_objects[object_id]

        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50, removeAction=remove_action)

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.total_frames = 0
        self.total_going_in = last_record["in_direction"]
        self.total_going_out = last_record["out_direction"]
        self.status = "Waiting"
        self.fps = 0

        # Counter for social distance violations.
        self.distance_violation_count = 0

        self.data = {
            "in_direction": self.total_going_in,
            "out_direction": self.total_going_out,
            "counter": 0,
            "social_distancing_v": 0,
            "in_frame_time_avg": 0,
            "fps": 0,
        }

        # Start the frames per second throughput estimator
        self.fps = None
        call_fps_thread = threading.Thread(target=self.call_fps, args=(), daemon=True)
        call_fps_thread.start()

        # Start data post Thread
        self.overpass_post_condition = False
        call_post_thread = threading.Thread(target=self.call_post, args=(), daemon=True)
        call_post_thread.start()

        # again doing this, why?
        input_frame_2 = np.frombuffer(input_frame, dtype=np.uint8)
        input_frame_2 = input_frame_2.reshape(frame_shape)
        output_frame_2 = np.frombuffer(output_frame, dtype=np.uint8)
        output_frame_2 = output_frame_2.reshape(frame_shape)
        try:
            self.gen_frames(input_frame_2, output_frame_2, flag)
        except KeyboardInterrupt:
            self.end_process()

    def call_post(self):
        call_post_thread = threading.Timer(3.0, self.call_post, args=())
        call_post_thread.start()

        # Sending Camara Data
        if self.data["counter"] != 0 or self.overpass_post_condition:
            self.overpass_post_condition = False
            self.socket_manager.send_camera_data(self.id, self.data)

    def call_fps(self):
        if self.fps is not None:
            self.fps.stop()
            if self.args["VERBOSE"]:
                print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
                print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
            self.fps = self.fps.fps()

        self.fps = FPS().start()

        call_fps_thread = threading.Timer(2.0, self.call_fps, args=())
        call_fps_thread.start()

    '''
    Function to get the social distance violations based on the position
    of the centroids detected in the frame.

    @objects (array): centroids (tuple) for every detected object.
    @return (set)		: coordinates of the centroids that violate
                                        social distancing.

    TODO
        Implement Bird Eye View (also called Inverse Perspective Mapping) for 
        better accuracy on social distancing violation detections.
        https://developer.ridgerun.com/wiki/index.php?title=Birds_Eye_View/Introduction/Research
    '''

    def get_social_distance_violations(self, objects):
        # Ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps).
        point_violations = set()
        if len(objects) >= 2:
            # Extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids.
            centroids = objects.values()
            np_centroids = np.array(list(centroids))
            D = dist.cdist(np_centroids, np_centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < self.social_distance_threshold:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        point_violations.add(i)
                        point_violations.add(j)
        return point_violations

    def generate_boxes_confidences_class_ids(self, outs, threshold):
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                # Get the scores, class_id, and the confidence of the prediction
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > threshold:
                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = np.array(detection[0:4]) * np.array([self.W, self.H, self.W, self.H])
                    (center_x, center_y, width, height) = box.astype("int")

                    start_x = int(center_x - (width / 2))
                    start_y = int(center_y - (height / 2))

                    # Append to list
                    boxes.append([start_x, start_y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def is_in_valid_area(self, box):
        start_x, start_y, width, height = box
        if self.detect_just_left_side:
            centroid = ((start_x + width // 2), (start_y + height // 2))
            return centroid[0] < self.W // 2
        return True

    def gen_frames(self, input_frame, output_frame, flag):
        # Loop over frames from the video stream.

        while True:
            # Counter for social distance violations.
            self.distance_violation_count = 0

            # TODO wastes CPU time, rework with waits
            # Grab the next frame if available.
            while not flag.value:
                pass
            flag.value = False
            frame[:] = input_frame

            # Convert the frame from BGR to RGB for dlib.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            self.status = "Waiting"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if self.total_frames == 0:
                # set the status and initialize our new set of object trackers
                self.status = "Detecting"
                self.trackers = []

                # convert the frame to a blob and pass the blob through the
                # network and obtain the detections
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)

                start = time.time()
                detections = self.net.forward(self.layer_names)
                end = time.time()
                if self.args["VERBOSE"]:
                    print("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

                boxes, confidences, classids = self.generate_boxes_confidences_class_ids(detections,
                                                                                         self.args["CONFIDENCE"])

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["CONFIDENCE"], self.NMS_THRESH)

                # loop over the detections
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the confidence (i.e., probability) associated
                        # with the prediction
                        confidence = confidences[i]

                        # filter out weak detections by requiring a minimum
                        # confidence
                        if confidence > self.args["CONFIDENCE"] and self.is_in_valid_area(boxes[i]):
                            # extract the index of the class label from the
                            # detections list
                            idx = int(classids[i])

                            # if the class label is not a person, ignore it
                            if CamaraProcessing.CLASSES[idx] != "person":
                                continue

                            startX, startY, width, height = boxes[i]

                            endX = startX + width
                            endY = startY + height

                            # construct a dlib rectangle object from the bounding
                            # box coordinates and then start the dlib correlation
                            # tracker`
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                            tracker.start_track(rgb, rect)

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            self.trackers.append(tracker)

            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            else:
                # loop over the trackers
                for tracker in self.trackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    self.status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))

            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
            if self.v_orientation:
                cv2.line(frame, (self.W // 2, 0), (self.W // 2, self.H), (255, 0, 0), 2)
            else:
                if self.detect_just_left_side:
                    cv2.line(frame, (0, self.H // 2), (self.W // 2, self.H // 2), (255, 0, 0), 2)
                else:
                    cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (255, 0, 0), 2)

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            object_position_data = self.ct.update(rects)
            objects = object_position_data["centroid"]
            points = object_position_data["rect"]

            # get social distancing violations and points of violation
            if self.run_distance_violation:
                violate = self.get_social_distance_violations(objects)
            else:
                violate = []

            # loop over the tracked objects
            for (i, (object_id, centroid)) in enumerate(objects.items()):

                # check to see if a trackable object exists for the current
                # object ID
                to = self.trackable_objects.get(object_id, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(object_id, centroid)

                # otherwise, append new centroid
                else:
                    to.centroids.append(centroid)

                # store the trackable object in our dictionary
                self.trackable_objects[object_id] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                x_start, y_start, x_end, y_end = points[object_id]

                text = "ID {}".format(object_id)
                color = self.COLOR_GREEN

                if i in violate:
                    self.distance_violation_count += 1
                    color = self.COLOR_RED

                cv2.rectangle(frame, (x_start, y_start), (x_start + 40, y_start + 15), color, -1)
                cv2.putText(frame, text, (x_start + 5, y_start + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 1)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 1)

            self.data = {
                "in_direction": self.total_going_in,
                "out_direction": self.total_going_out,
                "counter": len(objects.items()),
                "social_distancing_v": math.ceil(self.distance_violation_count / 2),
                "in_frame_time_avg": round(self.people_in_frame_time_avg, 3),
                "fps": int(self.fps),
            }

            # Publish frame.
            output_frame[:] = frame

            # Show the output frame.
            if self.args["VERBOSE"]:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # Increment frames counter.
            self.total_frames = (self.total_frames + 1) % self.args["SKIP_FRAMES"]

            # Update FPS counter.
            self.fps.update()

    def end_process(self):
        # check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()

        # close any open windows
        cv2.destroyAllWindows()


processReference = []
sources = []
frameShapes = []
inputFrames = []
outputFrames = []
flags = []


@app.route('/camara/<id>')
def camera_stream_route(id):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(show_frame(int(id)), mimetype='multipart/x-mixed-replace; boundary=frame')


def show_frame(id):
    outputFrame = np.frombuffer(outputFrames[id], dtype=np.uint8)
    outputFrame = outputFrame.reshape(frameShapes[id])
    while True:
        if ARGS["FORWARD_CAMERA"]:
            ret, buffer = cv2.imencode('.jpg', outputFrame)
        else:
            ret, buffer = cv2.imencode('.jpg', np.zeros(frameShapes[id], np.uint8))
        frame_ready = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
        time.sleep(1 / 60)  # Sleep 1/(FPS * 2)


@app.route('/')
def index_route():
    """Video streaming home page."""
    return render_template('indexv4.html', len=len(ARGS["CAMARAIDS"]), camaraIDs=ARGS["CAMARAIDS"])


BaseManager.register("socket_manager", SocketIOProcess)


def get_manager():
    m = BaseManager()
    m.start()
    return m


if __name__ == '__main__':
    # Initialize Socket Manager.
    manager = get_manager()
    socketManager = manager.socket_manager(ARGS)

    # TODO change into a wait
    # Wait till Camaras Info Received.
    while not socketManager.get_camera_info():
        pass

    per_camera_info = socketManager.get_camera_info()

    for index, camara in enumerate(per_camera_info):
        cap = cv2.VideoCapture(camara["source"])
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        frameShapes.append(frame.shape)
        cap.release()

        inputFrames.append(
            Array(ctypes.c_uint8, frameShapes[-1][0] * frameShapes[-1][1] * frameShapes[-1][2], lock=False))
        outputFrames.append(
            Array(ctypes.c_uint8, frameShapes[-1][0] * frameShapes[-1][1] * frameShapes[-1][2], lock=False))
        flags.append(Value(ctypes.c_bool, False))
        processReference.append(Process(target=CamaraProcessing, args=(
            index, camara["v_orientation"], camara["run_distance_violation"], camara["detect_just_left_side"],
            camara["last_record"][0], inputFrames[-1], outputFrames[-1], frameShapes[-1], flags[-1], socketManager,
            ARGS)))
        processReference[-1].start()

        sources.append(camara["source"])

    readProcessRef = Process(target=CamaraRead, args=(sources, inputFrames, frameShapes, flags, ARGS))
    readProcessRef.start()

    # TODO don't use waitress directly, this should expose a flask API to use with any server
    from waitress import serve

    app.debug = True
    app.use_reloader = False
    serve(app, host="0.0.0.0", port=8080)
    print("Server 0.0.0.0:8080")
    socketManager.wait()
