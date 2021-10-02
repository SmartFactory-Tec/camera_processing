# USAGE
# python covid19_measures.py

# import Dependencies
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pathlib
import argparse
import time
import os
import rospy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int16
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
import math
import copy

ARGS= {
	"GPU_AVAILABLE": True,
    "MODELS_PATH": str(pathlib.Path(__file__) / "../../../../models",
    "CONFIDENCE": 0.5,
}

class Person:
    def __init__(self, xpixel, ypixel, x, y, w, h):
        self.xpixel = xpixel
        self.ypixel = ypixel
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.drawed = False
        # depth_pixel = [self.xpixel, self.ypixel]
        # udist =  depth_frame.get_distance(self.xpixel, self.ypixel)
        # self.point = rs.rs2_deproject_pixel_to_point(color_intrin, depth_pixel,udist*(depth_scale*1000))

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
        self.image_sub = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, self.callback_depth)
        self.rgb_sub = rospy.Subscriber("/zed/zed_node/rgb/image_raw_color", Image, self.callback_rgb)
        self.pc_sub = rospy.Subscriber("/zed/zed_node/point_cloud/cloud_registered", PointCloud2, self.callback_pc)
        self.publisherImage = rospy.Publisher("/zed2/image/compressed", CompressedImage, queue_size = 1)
        self.publisherPeople = rospy.Publisher("/zed2/people_count", Int16, queue_size = 10)
        self.publisherDistanceViolations = rospy.Publisher("/zed2/distance_violations", Int16, queue_size = 10)
        self.publisherMaskCorrect = rospy.Publisher("/zed2/masks_correct", Int16, queue_size = 10)
        self.publisherMaskViolations = rospy.Publisher("/zed2/masks_violations", Int16, queue_size = 10)
        self.depth_image = []
        self.cv_image_rgb = []
        self.cv_image_rgb_processed = []
        self.pointcloud = PointCloud2()
        
        self.persons = []
        self.distanceviolations = 0
        self.mask_correct = 0
        self.mask_violations = 0
        
        # Load Models
        print("[INFO] Loading models...")
        def loadPersonsModel(self):
            weightsPath = ARGS["MODELS_PATH"] + "/people/People/yolov4.weights"
            cfgPath = ARGS["MODELS_PATH"] + "/people/yolov4.cfg"
            self.peopleNet = cv2.dnn.readNet(weightsPath, cfgPath)
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
			self.end_process()
    
    def callFps(self):	
		if self.fps != None:
			self.fps.stop()
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

    def callback_pc(self, data):
        self.pointcloud = data

    def publish(self):
        img = CompressedImage()
        img.header.stamp = rospy.Time.now()
        img.format = "jpeg"
        img.data = np.array(cv2.imencode('.jpg', self.cv_image_rgb)[1]).tostring()
        
        # Publish data.
        self.publisherImage.publish(img)
        self.publisherPeople.publish(len(self.persons))
        self.publisherDistanceViolations.publish(self.distanceviolations)
        self.publisherMaskViolations.publish(self.mask_violations)
        self.publisherMaskCorrect.publish(self.mask_correct)

    def run():
        while not rospy.is_shutdown():
            if len(self.depth_image) == 0 or len(self.cv_image_rgb) == 0:
                continue

            def detect_and_predict_people(self):
                height, width, channels = self.cv_image_rgb.shape
                blob = cv2.dnn.blobFromImage(self.cv_image_rgb, 1 / 255.0, (320, 320), swapRB=True, crop=False)
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
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                self.persons = []
                color_image_multiple = copy.copy(np.asanyarray(color_frame.get_data()))
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        if label == 'person':
                            xmid = int(x + w/2)
                            ymid = int(y + h/2)
                            dist = depth_frame.get_distance(xmid, ymid)
                            diststr = '%.4f' % dist
                            self.persons.append(Person(xmid, ymid, x, y, w, h))
                            cv2.putText(self.cv_image_rgb, diststr, (x, y + 50), font, 2, COLOR_BLUE, 3)
            
            detect_and_predict_people(self)

            def detect_mask_violations(self):
                def detect_and_predict_masks(self):
                    frame = self.cv_image_rgb
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
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
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            face = cv2.resize(face, (224, 224))
                            face = img_to_array(face)
                            face = preprocess_input(face)

                            # Add the face and bounding boxes to their respective list
                            faces.append(face)
                            locs.append((startX, startY, endX, endY))

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
                    # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    if label == "Mask":
                        self.mask_correct = self.mask_correct + 1 
                    else: 
                        self.mask_violations = self.mask_violations + 1
                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(self.cv_image_rgb, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(self.cv_image_rgb, (startX, startY), (endX, endY), color, 2)

            detect_mask_violations(self)

            def social_distancing(self):
                def calculatedistance(point1,point2):
                    return  math.sqrt(
                                math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(
                                    point1[2] - point2[2], 2))

                self.distance_violations = 0
                distances = []
                for index, person in enumerate(self.persons):
                    distances.append = []

                for i, iperson in enumerate(self.persons):
                    for j, jperson in enumerate(self.persons):
                        if i != j:
                            distance = calculatedistance(iperson.point,jperson.point)
                            distances[i].append(distance)
                            distances[j].append(distance)
                for i, iperson in enumerate(self.persons):
                    x = iperson.x
                    y = iperson.y
                    w = iperson.w
                    h = iperson.h
                    for idist in distances[i]:
                        if idist < float(1.0):
                            cv2.rectangle(self.cv_image_rgb, (x, y), (x + w, y + h), COLOR_RED, 2)
                            iperson.drawed = True
                            self.distance_violations = self.distance_violations + 1
                            break
                    if iperson.drawed == False:
                        cv2.rectangle(self.cv_image_rgb, (x, y), (x + w, y + h), COLOR_GREEN, 2)
                            
                cv2.putText(self.cv_image_rgb, 'Distance Violations:' + str(self.distance_violations), (5,25), font, 2, COLOR_BLUE, 3)

            social_distancing(self)

            self.publish()
            
            # Update FPS counter.
			self.fps.update()

def main():
    rospy.init_node('Covid19Measures', anonymous=True)
    CamaraProcessing()

if __name__ == '__main__':
    main()
