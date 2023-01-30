import math
import os
import cv2
import time
import threading
import dlib
import numpy as np
from imutils.video import FPS
from scipy.spatial.distance import cdist

from centroid_tracker import CentroidTracker
from trackable_object import TrackableObject


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
        if self.args["gpu_available"]:
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

        # TODO this was a fix to using a local scope variable inside the process
        # check if it can be scoped to the function, instead of as a member
        # also check if it even works
        self.frame = np.zeros(frame_shape)

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
                                                    time.time() - trackable_object.creation_time,
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

        self.ct = CentroidTracker(max_disappeared_frames=40, max_distance=50, remove_action=remove_action)

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
            if self.args["verbose"]:
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
            D = cdist(np_centroids, np_centroids, metric="euclidean")

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
            self.frame[:] = input_frame

            # Convert the frame from BGR to RGB for dlib.
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if self.W is None or self.H is None:
                (self.H, self.W) = self.frame.shape[:2]

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
                blob = cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)

                start = time.time()
                detections = self.net.forward(self.layer_names)
                end = time.time()
                if self.args["verbose"]:
                    print("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

                boxes, confidences, classids = self.generate_boxes_confidences_class_ids(detections,
                                                                                         self.args["confidence"])

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.args["confidence"], self.NMS_THRESH)

                # loop over the detections
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the confidence (i.e., probability) associated
                        # with the prediction
                        confidence = confidences[i]

                        # filter out weak detections by requiring a minimum
                        # confidence
                        if confidence > self.args["confidence"] and self.is_in_valid_area(boxes[i]):
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
                cv2.line(self.frame, (self.W // 2, 0), (self.W // 2, self.H), (255, 0, 0), 2)
            else:
                if self.detect_just_left_side:
                    cv2.line(self.frame, (0, self.H // 2), (self.W // 2, self.H // 2), (255, 0, 0), 2)
                else:
                    cv2.line(self.frame, (0, self.H // 2), (self.W, self.H // 2), (255, 0, 0), 2)

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

                cv2.rectangle(self.frame, (x_start, y_start), (x_start + 40, y_start + 15), color, -1)
                cv2.putText(self.frame, text, (x_start + 5, y_start + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 1)
                cv2.circle(self.frame, (centroid[0], centroid[1]), 4, color, -1)
                cv2.rectangle(self.frame, (x_start, y_start), (x_end, y_end), color, 1)

            self.data = {
                "in_direction": self.total_going_in,
                "out_direction": self.total_going_out,
                "counter": len(objects.items()),
                "social_distancing_v": math.ceil(self.distance_violation_count / 2),
                "in_frame_time_avg": round(self.people_in_frame_time_avg, 3),
                "fps": int(self.fps),
            }

            # Publish frame.
            output_frame[:] = self.frame

            # Show the output frame.
            if self.args["verbose"]:
                cv2.imshow("Frame", self.frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # Increment frames counter.
            self.total_frames = (self.total_frames + 1) % self.args["skip_frames"]

            # Update FPS counter.
            self.fps.update()

    def end_process(self):
        # check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()

        # close any open windows
        cv2.destroyAllWindows()
