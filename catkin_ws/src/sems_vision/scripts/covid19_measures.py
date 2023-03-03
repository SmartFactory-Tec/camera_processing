#!/usr/bin/env python3
# USAGE
# python3 covid19_measures.py

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import FPS
import threading
import pathlib
import argparse
import time
import os
import rospy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int16
from std_msgs.msg import Float32MultiArray
import dlib
import screeninfo
import imutils
import numpy as np
import cv2
import copy
import sys
sys.path.append(str(pathlib.Path(__file__).parent) + '/../include')
from sems_vision_utils import *

FLT_EPSILON = sys.float_info.epsilon
ARGS= {
    "GPU_AVAILABLE": True,
    "MODELS_PATH": str(pathlib.Path(__file__).parent) + "/../../../../models",
    "CONFIDENCE": 0.5,
    "SHOW_FACES": False,
    "SKIP_FRAMES": 5,
    "SHOW_DISTANCE_VIOLATIONS_COUNTER": False,
}

class Person:
    def __init__(self, xCentroid, yCentroid, x, y, w, h, tracker):
        self.point2D = (xCentroid, yCentroid)
        self.point3D = None
        self.rect = (x, y, w, h)
        self.distanceViolation = False
        self.tracker = tracker
        self.usingMask = None
        self.maskPred = None
        self.depth = None

class CamaraProcessing:
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)
    COLOR_BLACK = (0, 0, 0)

    CLASSES = None
    with open(ARGS["MODELS_PATH"] + '/people/coco.names', 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]

    def __init__(self):
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.callback_depth)
        self.rgb_sub = rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color", Image, self.callback_rgb)
        self.rgb_sub_info = rospy.Subscriber("/zed2/zed_node/rgb/camera_info", CameraInfo, self.callback_rgb_info)
        self.publisherImage = rospy.Publisher("/zed2_/image/compressed", CompressedImage, queue_size = 1)
        self.publisherPeople = rospy.Publisher("/zed2_/people_count", Int16, queue_size = 10)
        self.publisherDistanceViolations = rospy.Publisher("/zed2_/distance_violations", Int16, queue_size = 10)
        self.publisherMaskCorrect = rospy.Publisher("/zed2_/masks_correct", Int16, queue_size = 10)
        self.publisherMaskViolations = rospy.Publisher("/zed2_/masks_violations", Int16, queue_size = 10)
        self.depth_image = []
        self.cv_image_rgb = []
        self.cv_image_rgb_processed = []
        self.cv_image_rgb_drawed = []
        self.cv_image_rgb_info = CameraInfo()
        
        self.frameCounter = 0
        
        self.persons = []
        self.distanceviolations = 0
        self.mask_correct = 0
        self.mask_violations = 0

        self.logo_image = imutils.resize(cv2.imread(str(pathlib.Path(__file__).parent) + "/../images/roborregos_logo.png", -1), width=180)

        # Load Models
        print("[INFO] Loading models...")
        def loadPersonsModel(self):
            weightsPath = ARGS["MODELS_PATH"] + "/people/yolov4.weights"
            cfgPath = ARGS["MODELS_PATH"] + "/people/yolov4.cfg"
            self.peopleNet = cv2.dnn.readNetFromDarknet(cfgPath, weightsPath)
            if ARGS["GPU_AVAILABLE"]:
                # set CUDA as the preferable backend and target
                self.peopleNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.peopleNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
            # Get the output layer names of the model
            layer_names = self.peopleNet.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.peopleNet.getUnconnectedOutLayers()]
        
        def loadFacesModel(self):
            prototxtPath = ARGS["MODELS_PATH"] + "/faces/deploy.prototxt"
            weightsPath = ARGS["MODELS_PATH"] + "/faces/res10_300x300_ssd_iter_140000.caffemodel"
            self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            if ARGS["GPU_AVAILABLE"]:
                # set CUDA as the preferable backend and target
                self.faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        def loadMasksModel(self):
            modelPath = ARGS["MODELS_PATH"] + "/masks/mask_detector.model"
            self.maskNet = load_model(modelPath)

        loadPersonsModel(self)
        loadFacesModel(self)
        loadMasksModel(self)
        print("[INFO] Models Loaded")

        # Frames per second throughput estimator
        self.fps = None
        callFpsThread = threading.Thread(target=self.callFps, args=(), daemon=True)
        callFpsThread.start()

        try:
            self.run()
        except KeyboardInterrupt:
            pass
    
    def callFps(self):	
        if self.fps != None:
            self.fps.shutdown()
            print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
            self.fpsValue = self.fps.fps()

        self.fps = FPS().start()
        
        callFpsThread = threading.Timer(2.0, self.callFps, args=())
        callFpsThread.start()

    def callback_depth(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

    def callback_rgb(self, data):
        try:
            self.cv_image_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_rgb_info(self, data):
        self.cv_image_rgb_info = data
        self.rgb_sub_info.unregister()

    def publish(self):
        img = CompressedImage()
        img.header.stamp = rospy.Time.now()
        img.format = "jpeg"
        img.data = np.array(cv2.imencode('.jpg', self.cv_image_rgb_processed)[1]).tostring()
        
        # Publish frame.
        self.publisherImage.publish(img)
        self.publisherPeople.publish(len(self.persons))
        self.publisherDistanceViolations.publish(self.distanceviolations)
        self.publisherMaskViolations.publish(self.mask_violations)
        self.publisherMaskCorrect.publish(self.mask_correct)

    def run(self):
        cv2.namedWindow("SEMS", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("SEMS", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while not rospy.is_shutdown():
            if len(self.depth_image) == 0 or len(self.cv_image_rgb) == 0:
                continue
            
            self.cv_image_rgb_drawed = self.cv_image_rgb.copy()
            self.cv_image_rgb_processed = imutils.resize(self.cv_image_rgb, width=500)

            # Convert the frame from BGR to RGB for dlib.
            cv_image_dlib = cv2.cvtColor(self.cv_image_rgb_processed, cv2.COLOR_BGR2RGB)

            def detect_and_predict_people(self):
                height, width, channels = self.cv_image_rgb_processed.shape
                blob = cv2.dnn.blobFromImage(self.cv_image_rgb_processed, 1 / 255.0, (320, 320), swapRB=True, crop=False)
                self.peopleNet.setInput(blob)
                outs = self.peopleNet.forward(self.output_layers)

                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                self.persons = []
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = mapRectangle(self.cv_image_rgb_processed.shape, self.cv_image_rgb_drawed.shape, boxes[i])
                        label = str(CamaraProcessing.CLASSES[class_ids[i]])
                        if label == 'person':
                            xmid = int(x + w/2)
                            ymid = int(y + h/2)
                            endX = x + w
                            endY = y + h

                            # construct a dlib rectangle object from the bounding
                            # box coordinates and then start the dlib correlation
                            # tracker`
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(int(x), int(y), int(endX), int(endY))
                            tracker.start_track(cv_image_dlib, rect)

                            self.persons.append(Person(xmid, ymid, x, y, w, h, tracker))

            def track_people(self):
                for person in self.persons:
                    person._tracker.update(cv_image_dlib)
                    pos = person._tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    self.rect = (startX, startY, endX - startX, endY - startY)

            def detect_mask_violations(self):
                def detect_and_predict_masks(self):
                    frame = self.cv_image_rgb_drawed
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (900 , 900),
                        (104.0, 177.0, 123.0))

                    # Obtain face detections
                    self.faceNet.setInput(blob)
                    detections = self.faceNet.forward()

                    faces = []
                    locs = []
                    preds = []

                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]

                        if confidence > ARGS["CONFIDENCE"]:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                    
                            (startX, startY) = (max(0, startX), max(0, startY))
                            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                            # Preprocess Image
                            face = frame[startY:endY, startX:endX]
                            if len(face) == 0 or startX >= endX or startY >= endY:
                                continue
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            face = cv2.resize(face, (224, 224))
                            face = img_to_array(face)
                            face = preprocess_input(face)

                            # Add the face and bounding boxes to their respective list
                            faces.append(face)
                            box = (startX, startY, endX - startX, endY - startY)
                            locs.append(box)

                    if len(faces) > 0:
                        faces = np.array(faces, dtype="float32")
                        preds = self.maskNet.predict(faces, batch_size=32)

                    # return a tuple of the face locations and prediction
                    return (locs, preds)
                
                (locs, preds) = detect_and_predict_masks(self)

                # loop over the detected face locations
                self.mask_correct = 0
                self.mask_violations = 0
                for (box, pred) in zip(locs, preds):
                    (startX, startY, width, height) = box
                    centroid = (startX + 0.5 * width, startY + 0.5 * height)

                    (mask, withoutMask) = pred
                    usingMask = mask > withoutMask

                    for i, iperson in enumerate(self.persons):
                        if iperson.usingMask == None and point_inside_rect(centroid, iperson.rect):
                            iperson.usingMask = usingMask
                            iperson.maskPred = max(mask, withoutMask) * 100
                            if usingMask:
                                self.mask_correct = self.mask_correct + 1 
                            else: 
                                self.mask_violations = self.mask_violations + 1

                    # SHOW MASKS RECT & PROBABILITY - ADDIMG
                    if ARGS["SHOW_FACES"]:
                        cv2.rectangle(self.cv_image_rgb_drawed, (startX, startY), (startX + width, startY + height), color, 2)

            def social_distancing(self):
                self.distance_violations = 0
                distances = []
                for index, person in enumerate(self.persons):
                    person.depth = get_depth(self.cv_image_rgb_processed, self.depth_image, person.point2D)
                    person.point3D = deproject_pixel_to_point(self.cv_image_rgb_info, person.point2D, person.depth)
                    distances.append([])

                for i, iperson in enumerate(self.persons):
                    for j, jperson in enumerate(self.persons):
                        if i != j:
                            distance = calculatedistance(iperson.point3D,jperson.point3D)
                            if distance < float(1.0):
                                self.distance_violations = self.distance_violations + 1
                                iperson.distanceViolation = True
                                jperson.distanceViolation = True

                # SHOW DISTANCE VIOLATIONS COUNTER - ADDIMG
                if ARGS["SHOW_DISTANCE_VIOLATIONS_COUNTER"]:
                    cv2.putText(self.cv_image_rgb_processed, 'Distance Violations:' + str(self.distance_violations), (5,25), cv2.FONT_HERSHEY_PLAIN, 2, CamaraProcessing.COLOR_BLUE, 2)

            def draw_over_frame(self):
                for i, iperson in enumerate(self.persons):
                    x, y, w, h = iperson.rect
                    rectColor = CamaraProcessing.COLOR_RED if iperson.distanceViolation else CamaraProcessing.COLOR_GREEN

                    # SHOW PERSON RECTS - ADDIMG
                    cv2.rectangle(self.cv_image_rgb_drawed, (x, y), (x + w, y + h), rectColor, 2)
                    
                    if iperson.usingMask != None:
                        maskLabel = "Mask" if iperson.usingMask else "No Mask"
                        maskLabel = "{}: {:.2f}%".format(maskLabel, iperson.maskPred)
                        maskLabelColor = CamaraProcessing.COLOR_RED if not iperson.usingMask else CamaraProcessing.COLOR_GREEN
                        (textScale, textHeight) = get_optimal_font_scale(maskLabel, w - 10, cv2.FONT_HERSHEY_PLAIN, 3)

                        # SHOW MASK LABEL - ADDIMG
                        cv2.putText(self.cv_image_rgb_drawed, maskLabel, (x, y + h - textHeight), cv2.FONT_HERSHEY_PLAIN, textScale, maskLabelColor, 3)

                def add_logo(self):
                    padding = (50, 25)
                    (heightImg, widthImg, _) = self.cv_image_rgb_drawed.shape
                    (heightLogo, widthLogo, _) = self.logo_image.shape
                    position = (widthImg - widthLogo - padding[0], padding[1])
                    return add_image(self.cv_image_rgb_drawed, self.logo_image, position)

                add_logo(self)

            if self.frameCounter == 0:
                detect_and_predict_people(self)
                detect_mask_violations(self)
            else:
                track_people(self)

            social_distancing(self)
            draw_over_frame(self)

            cv2.imshow("SEMS", self.cv_image_rgb_drawed)
            cv2.waitKey(1)
            self.publish()

            self.frameCounter = (self.frameCounter + 1) % ARGS["SKIP_FRAMES"]

            self.fps.update()

def main():
    rospy.init_node('Covid19Measures', anonymous=True)
    CamaraProcessing()

if __name__ == '__main__':
    main()
