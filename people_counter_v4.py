# USAGE
# python3 people_counter_v4.py

# Import Dependencies
from multiprocessing import Process, Array
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from collections import deque
from flask import Flask, render_template, Response
from imutils.video import VideoStream
from imutils.video import FPS
from dotenv import load_dotenv
from scipy.spatial import distance as dist
import os
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import requests
import threading
import json
import ctypes
import math
import socketio
import socket

BACK_AVAILABLE = False
GPU_AVAILABLE = True
VERBOSE = False
CONFIDENCE_ = 0.3
SKIP_FRAMES_ = 25
MAX_FPS = 60

load_dotenv()
app = Flask(__name__)

with open('inputScript_TestVideo.json') as inputScript:
  inputSources = json.load(inputScript)

class Camara:
	API_ENDPOINT = os.getenv("API_ENDPOINT")
	COLOR_RED = (0, 0, 255)
	COLOR_GREEN = (0, 255, 0)
	COLOR_BLACK = (0, 0, 0)
	socialDistanceThreshold = 90
	
	sio = socketio.Client()
	
	# Initialize list of class labels MobileNet SSD was trained to detect
	CLASSES = None
	with open('yolo/coco.names', 'r') as f:
		CLASSES = [line.strip() for line in f.readlines()]
	
	def __init__(self, id, inputSource, inputFrame, frameShape):
		self.id = id
		self.idDb = 0
		self.camaraId = "Camara" + str(id)
		
		# Load Model
		self.net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
		if GPU_AVAILABLE:
			# set CUDA as the preferable backend and target
			print("[INFO] setting preferable backend and target to CUDA...")
			net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

  	# Get the output layer names of the model
		self.layer_names = self.net.getLayerNames()
		self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

		self.inputSource = inputSource
		
		print("[INFO] opening video file...")
		self.vs = cv2.VideoCapture(self.inputSource)
		self.vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

		# initialize the video writer (we'll instantiate later if need be)
		self.writer = None

		# initialize the frame dimensions (we'll set them as soon as we read
		# the first frame from the video)
		self.W = None
		self.H = None

		# Non maxima supression threshold
		self.NMS_THRESH = 0.3

		# Instantiate our centroid tracker, then initialize a list to store
		# each of our dlib correlation trackers, followed by a dictionary to
		# map each unique object ID to a TrackableObject
		self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
		self.trackers = []
		self.trackableObjects = {}

		# initialize the total number of frames processed thus far, along
		# with the total number of objects that have moved either up or down
		self.totalFrames = 0
		self.totalDown = 0
		self.totalUp = 0
		self.status = "Waiting"
		self.fpsValue = 30

		# Counter for social distance violations.
		self.totalDistanceViolations = 0

		self.data = {
			"in_direction": 0,
			"out_direction": 0,
			"counter": 0,
			"social_distancing_v": 0,
			"fps": 0,
		}

		# Start the frames per second throughput estimator
		self.fps = None
		callFpsThread = threading.Thread(target=self.callFps, args=())
		callFpsThread.start()


		if BACK_AVAILABLE:
			self.sioConnected = False
			self.sio.on('connect', self.connectSIO)
			self.sio.on('disconnect', self.disconnectSIO)
			self.sio.on('visionInit', self.visionInitSIO)
			self.sio.connect('http://covid-response-back.herokuapp.com/')

		sharedFrame = np.frombuffer(inputFrame, dtype=np.uint8)
		sharedFrame = sharedFrame.reshape(frameShape)    
		self.gen_frames(sharedFrame)
	
	def connectSIO(self):
		self.sioConnected = True
		print('Connection Established.')
		quantityCamaras = 2
		self.sio.emit('visionInit', quantityCamaras)

	def disconnectSIO(self):
		self.sioConnected = False
	
	def visionInitSIO(self, camaraInfo):
		print('CamaraInfo ', camaraInfo)
		self.idDb = camaraInfo[self.id]['id']
		self.sio.emit('updateCamara', data=(
			self.idDb, 
			'http://' + socket.getfqdn() + ':8080/camara/' + str(self.id)
		))

		callPostThread = threading.Thread(target=self.callPost, args=())
		callPostThread.start()

	def callPost(self):
		callPostThread = threading.Timer(3.0, self.callPost, args=())
		callPostThread.start()
		
		# Sending Camara Data
		if self.data["counter"] != 0 and self.sioConnected:
			self.sio.emit('visionPost', data=(
				self.idDb,
				self.data["in_direction"],
				self.data["out_direction"],
				self.data["counter"],
				self.data["social_distancing_v"],
				self.data["fps"],
			))

	def callFps(self):	
		if self.fps != None:
			self.fps.stop()
			print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
			self.fpsValue = self.fps.fps()

		self.fps = FPS().start()
		
		callFpsThread = threading.Timer(2.0, self.callFps, args=())
		callFpsThread.start()

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
					if D[i, j] < self.socialDistanceThreshold:
						# update our violation set with the indexes of
						# the centroid pairs
						point_violations.add(i)
						point_violations.add(j)
		return point_violations

	def generate_boxes_confidences_classids(self, outs, threshold):
		boxes = []
		confidences = []
		classids = []

		for out in outs:
				for detection in out:
					# Get the scores, classid, and the confidence of the prediction
					scores = detection[5:]
					classid = np.argmax(scores)
					confidence = scores[classid]				

					if confidence > threshold:

						# compute the (x, y)-coordinates of the bounding box
						# for the object
						box = np.array(detection[0:4]) * np.array([self.W, self.H, self.W, self.H])
						(centerX, centerY, width, height) = box.astype("int")

						startX = int(centerX - (width / 2))
						startY = int(centerY - (height / 2))

						# Append to list
						boxes.append([startX, startY, int(width), int(height)])
						confidences.append(float(confidence))
						classids.append(classid)

		return boxes, confidences, classids

	def gen_frames(self, sharedFrame):
		# Loop over frames from the video stream.
		lastTime = 0

		while True:
			# Counter for social distance violations.
			self.totalDistanceViolations = 0

			# Grab the next frame and handle if we are reading from either
			# VideoCapture or VideoStream.
			status, frame = self.vs.read()
			lastTime = time.time()

			if not status:
				break

			# Resize the frame to have a maximum width of 500 pixels (the
			# less data we have, the faster we can process it), then convert
			# the frame from BGR to RGB for dlib.
			frame = imutils.resize(frame, width=500)
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
			if self.totalFrames % SKIP_FRAMES_ == 0:
				# set the status and initialize our new set of object trackers
				self.status = "Detecting"
				self.trackers = []

				# convert the frame to a blob and pass the blob through the
				# network and obtain the detections
				blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
				self.net.setInput(blob)
				
				start = time.time()
				detections = self.net.forward(self.layer_names)
				end = time.time()
				print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

				boxes, confidences, classids = self.generate_boxes_confidences_classids(detections, CONFIDENCE_)

				idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_, self.NMS_THRESH)

				# loop over the detections
				if len(idxs) > 0:
						# loop over the indexes we are keeping
						for i in idxs.flatten():
							# extract the confidence (i.e., probability) associated
							# with the prediction
							confidence = confidences[i]

							# filter out weak detections by requiring a minimum
							# confidence
							if confidence > CONFIDENCE_:
								# extract the index of the class label from the
								# detections list
								idx = int(classids[i])

								# if the class label is not a person, ignore it
								if Camara.CLASSES[idx] != "person":
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
			cv2.line(frame, (self.W//2, 0), (self.W // 2, self.H), (0, 255, 255), 2)

			# use the centroid tracker to associate the (1) old object
			# centroids with (2) the newly computed object centroids
			object_position_data = self.ct.update(rects)
			objects = object_position_data["centroid"]
			points = object_position_data["rect"]

			# get social distancing violations and points of violation
			violate = self.get_social_distance_violations(objects)

			# loop over the tracked objects
			for (i, (objectID, centroid)) in enumerate(objects.items()):
				
				# check to see if a trackable object exists for the current
				# object ID
				to = self.trackableObjects.get(objectID, None)

				# if there is no existing trackable object, create one
				if to is None:
					to = TrackableObject(objectID, centroid)

				# otherwise, there is a trackable object so we can utilize it
				# to determine direction
				else:
					# the difference between the x-coordinate of the current
					# centroid and the mean of previous centroids will tell
					# us in which direction the object is moving (negative for
					# 'left' and positive for 'right')
					x = [c[0] for c in to.centroids]
					direction = centroid[0] - np.mean(x)
					to.centroids.append(centroid)

					# check to see if the object has been counted or not
					if not to.counted:
						# if the direction is negative (indicating the object
						# is moving up) AND the centroid is above the center
						# line, count the object
						if direction < 0 and centroid[0] < self.W // 2:
							self.totalUp += 1
							to.counted = True

						# if the direction is positive (indicating the object
						# is moving down) AND the centroid is below the
						# center line, count the object
						elif direction > 0 and centroid[0] > self.W // 2:
							self.totalDown += 1
							to.counted = True

				# store the trackable object in our dictionary
				self.trackableObjects[objectID] = to

				# draw both the ID of the object and the centroid of the
				# object on the output frame
				x_start, y_start, x_end, y_end = points[objectID]

				text = "ID {}".format(objectID)
				color = self.COLOR_GREEN

				if i in violate:
					self.totalDistanceViolations += 1
					color = self.COLOR_RED

				cv2.rectangle(frame, (x_start, y_start), (x_start + 40, y_start + 15), color, -1)
				cv2.putText(frame, text, (x_start + 5, y_start + 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
				cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 1)

			self.data = {
				"in_direction": self.totalUp,
				"out_direction": self.totalDown,
				"counter": len(objects.items()),
				"social_distancing_v": math.ceil(self.totalDistanceViolations/2),
				"fps": int(self.fpsValue),
			}

			#Publish frame in Shared Array
			sharedFrame[:] = frame

			# show the output frame
			if VERBOSE:
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break

			# increment the total number of frames processed thus far and
			# then update the FPS counter
			self.totalFrames += 1
			
			# Handle Max Fps
			while time.time() - lastTime  < 1 / MAX_FPS:
				pass
				
			self.fps.update()

		
		self.vs = cv2.VideoCapture(self.inputSource)
		self.vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
		self.gen_frames(sharedFrame)

	def end_process(self):
		# check to see if we need to release the video writer pointer
		if self.writer is not None:
			self.writer.release()

		self.vs.release()

		# close any open windows
		cv2.destroyAllWindows()


processReference = []
inputShapes = []
inputFrames = []

@app.route('/camara/<id>')
def camaraStream(id):
	#Video streaming route. Put this in the src attribute of an img tag
	return Response(showFrame(int(id)), mimetype='multipart/x-mixed-replace; boundary=frame')

def showFrame(id):
	inputFrame = np.frombuffer(inputFrames[id], dtype=np.uint8)
	inputFrame = inputFrame.reshape(inputShapes[id])
	while True:
		ret, buffer = cv2.imencode('.jpg', inputFrame)
		frame_ready = buffer.tobytes()
		yield (b'--frame\r\n'
						b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
		time.sleep(1 / 60 ) # Sleep 1/(FPS * 2) 
  	

@app.route('/')
def index():
	"""Video streaming home page."""
	return render_template('index.html', inputSources = inputSources)


if __name__ == '__main__':
  
	refI = 0
	refE = 0
	for location in inputSources:
		refI = 0
		for camara in inputSources[location]["camaras"]:
			cap = cv2.VideoCapture(camara["src"])
			ret, frame = cap.read()
			frame = imutils.resize(frame, width=500)
			inputShapes.append(frame.shape)
			cap.release()

			inputFrames.append(Array(ctypes.c_uint8, inputShapes[-1][0] * inputShapes[-1][1] * inputShapes[-1][2], lock=False))
			processReference.append(Process(target=Camara, args=(refE, camara["src"], inputFrames[-1], inputShapes[-1])))
			processReference[-1].start()

			camara["refI"] = refI
			camara["refE"] = refE
			refI += 1
			refE += 1

	from waitress import serve

	app.debug=True
	app.use_reloader=False
	serve(app, host="0.0.0.0", port=8080)
	print("Server 0.0.0.0:8080")	
