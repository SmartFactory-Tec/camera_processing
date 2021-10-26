# USAGE
# python3 main.py

# Import Dependencies
from multiprocessing import Process, Array, Value
from multiprocessing.managers import BaseManager
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from flask import Flask, render_template, Response
from imutils.video import VideoStream
from imutils.video import FPS
from scipy.spatial import distance as dist
from queue import Queue
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

ARGS= {
	"CAMARAIDS": [8],
	"BACK_ENDPOINT": ["http://sems.back.ngrok.io/", "http://localhost:3001/"][0],
	"NGROK_AVAILABLE": True,
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
		self.camaraIDs = self.args["CAMARAIDS"]
		self.quantityCamaras = len(self.camaraIDs)
		self.camarasInfo = []
		self.sioConnected = False
		self.hasCamarasInfo = False
		self.sio.on('connect', self.connectSIO)
		self.sio.on('disconnect', self.disconnectSIO)
		self.sio.on('visionInit', self.visionInitSIO)
		self.sio.connect(self.args["BACK_ENDPOINT"])
	
	def connectSIO(self):
		print('Connected')
		self.sioConnected = True
		self.sio.emit('visionInit', self.camaraIDs)

	def disconnectSIO(self):
		print('Disconnected')
		self.sioConnected = False
		self.hasCamarasInfo = False
		self.camarasInfo = []
	
	def waitSIO(self):
		self.sio.wait()

	def visionInitSIO(self, camarasInfo):
		self.camarasInfo = camarasInfo
		self.hasCamarasInfo = True
		print('CamaraInfo ', camarasInfo)

	def getCamaraInfo(self, id = None):
		if not self.hasCamarasInfo:
			return False
		
		if not id:
			return self.camarasInfo
		
		return self.camarasInfo[id]
	
	def sendCamaraData(self, id, data):
		if self.sioConnected:
			self.sio.emit('visionPost', data=(
				self.camarasInfo[id]['id'],
				data["in_direction"],
				data["out_direction"],
				data["counter"],
				data["social_distancing_v"],
				data["in_frame_time_avg"],
				data["fps"],
			))

	def setCamaraURL(self, id):
		if self.sioConnected:
			if self.args["NGROK_AVAILABLE"] and self.args["FORWARD_CAMERA"]:
				endpoint = 'http://sems.ngrok.io/camara/'
			elif self.args["FORWARD_CAMERA"]:	
				endpoint = 'http://' + socket.getfqdn() + ':8080/camara/'
			else:
				endpoint = ''

			self.sio.emit('updateCamara', data=(
				self.camarasInfo[id]['id'], 
				endpoint + str(id)
			))

class CamaraRead:
	MAX_FPS = 34
	MAX_SKIP = 3

	def __init__(self, sources, inputFrames, frameShapes, flags, args):
		self.sources = sources
		self.inputFrames = inputFrames
		self.frameShapes = frameShapes
		self.flags = flags
		self.args = args
		for index in range(len(sources)):
			readThread = threading.Thread(target=self.mainLoop, args=(index,), daemon=True)
			readThread.start()
		
		readThread.join()
	
	def mainLoop(self, index):
		source = self.sources[index]
		inputFrame = self.inputFrames[index]
		frameShape = self.frameShapes[index]
		flag = self.flags[index]

		print("[INFO] opening video file...", source)
		vs = cv2.VideoCapture(source)
		vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
		inputFrame_ = np.frombuffer(inputFrame, dtype=np.uint8)
		inputFrame_ = inputFrame_.reshape(frameShape)
		
		q = Queue(maxsize = 0)
		notTakenCounter = 0

		while True:
			lastTime = time.time()
			status, frame = vs.read()

			if not status:
				vs = cv2.VideoCapture(source)
				vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
				continue

			frame = imutils.resize(frame, width=500)

			if notTakenCounter == 0:
				q.put(frame)

			if not flag.value:
				if (q.empty()):
					inputFrame_[:] = frame
					notTakenCounter = 0
				else:
					inputFrame_[:] = q.get()
				flag.value = True

			notTakenCounter = (notTakenCounter + 1) % CamaraRead.MAX_SKIP

			while time.time() - lastTime  < 1 / CamaraRead.MAX_FPS:
				pass

class CamaraProcessing:
	COLOR_RED = (0, 0, 255)
	COLOR_GREEN = (0, 255, 0)
	COLOR_BLACK = (0, 0, 0)
	socialDistanceThreshold = 90
	
	CLASSES = None
	with open('models/people/coco.names', 'r') as f:
		CLASSES = [line.strip() for line in f.readlines()]
	
	def __init__(self, id, v_orientation, run_distance_violation, detect_just_left_side, last_record, inputFrame, outputFrame, frameShape, flag, socketManager, args):
		self.id = id
		self.v_orientation = v_orientation
		self.run_distance_violation = run_distance_violation
		self.detect_just_left_side = detect_just_left_side
		self.camaraId = "Camara" + str(self.id)
		self.socketManager = socketManager
		self.socketManager.setCamaraURL(self.id)
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
		self.peopleTimeAvg = 0
		self.peopleCounter = 0

		# Instantiate our centroid tracker, initialize a list to store
		# each of our dlib correlation trackers and a dictionary to
		# map each unique object ID to a TrackableObject
		self.trackers = []
		self.trackableObjects = {}

		# Instantiate custom removeAction for centroid tracker.
		def removeAction(objectID):
			def getAvg(prev_avg, x, n):
				return (prev_avg * n + x) / (n + 1)

			tmpTO = self.trackableObjects[objectID]
			self.peopleTimeAvg = getAvg(self.peopleTimeAvg, time.time() - tmpTO.startTime, self.peopleCounter)
			self.peopleCounter += 1

			def determineDirection(self, to):
				if self.v_orientation:
					x = [c[0] for c in to.centroids]
					
					if x[len(x) - 1] < (self.W // 2) and x[0] > (self.W // 2):
						self.totalInDir += 1

					elif x[len(x) - 1] > (self.W // 2) and x[0] < (self.W // 2):
						self.totalOutDir += 1
				else:
					y = [c[1] for c in to.centroids]

					if y[len(y) - 1] < (self.H // 2) and y[0] > (self.H // 2):
						self.totalInDir += 1	
					elif y[len(y) - 1] > (self.H // 2) and y[0] < (self.H // 2):
						self.totalOutDir += 1

			determineDirection(self, tmpTO)
			self.overpassPostCondition =  True
			del self.trackableObjects[objectID]

		self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50, removeAction=removeAction)

		# initialize the total number of frames processed thus far, along
		# with the total number of objects that have moved either up or down
		self.totalFrames = 0
		self.totalInDir = last_record["in_direction"]
		self.totalOutDir = last_record["out_direction"]
		self.status = "Waiting"
		self.fpsValue = 0

		# Counter for social distance violations.
		self.totalDistanceViolations = 0

		self.data = {
			"in_direction": self.totalInDir,
			"out_direction": self.totalOutDir,
			"counter": 0,
			"social_distancing_v": 0,
			"in_frame_time_avg": 0,
			"fps": 0,
		}

		# Start the frames per second throughput estimator
		self.fps = None
		callFpsThread = threading.Thread(target=self.callFps, args=(), daemon=True)
		callFpsThread.start()

		# Start data post Thread
		self.overpassPostCondition = False
		callPostThread = threading.Thread(target=self.callPost, args=(), daemon=True)
		callPostThread.start()

		inputFrame_ = np.frombuffer(inputFrame, dtype=np.uint8)
		inputFrame_ = inputFrame_.reshape(frameShape)
		outputFrame_ = np.frombuffer(outputFrame, dtype=np.uint8)
		outputFrame_ = outputFrame_.reshape(frameShape)
		try:
			self.gen_frames(inputFrame_, outputFrame_, flag)
		except KeyboardInterrupt:
			self.end_process()

	def callPost(self):
		callPostThread = threading.Timer(3.0, self.callPost, args=())
		callPostThread.start()
		
		# Sending Camara Data
		if self.data["counter"] != 0 or self.overpassPostCondition:
			self.overpassPostCondition = False
			self.socketManager.sendCamaraData(self.id, self.data)

	def callFps(self):	
		if self.fps != None:
			self.fps.stop()
			if self.args["VERBOSE"]:
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

	def is_in_valid_area(self, box):
		startX, startY, width, height = box
		if self.detect_just_left_side:
			centroid = ((startX + width // 2), (startY + height // 2))
			return centroid[0] < self.W // 2
		return True

	def gen_frames(self, inputFrame_, outputFrame_, flag):
		# Loop over frames from the video stream.

		while True:
			# Counter for social distance violations.
			self.totalDistanceViolations = 0

			# Grab the next frame if available.
			while(not flag.value):
				pass
			flag.value = False
			frame[:] = inputFrame_

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
			if self.totalFrames == 0:
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
				if self.args["VERBOSE"]:
					print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

				boxes, confidences, classids = self.generate_boxes_confidences_classids(detections, self.args["CONFIDENCE"])

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
				cv2.line(frame, (self.W//2, 0), (self.W // 2, self.H), (255, 0, 0), 2)
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
			for (i, (objectID, centroid)) in enumerate(objects.items()):
				
				# check to see if a trackable object exists for the current
				# object ID
				to = self.trackableObjects.get(objectID, None)

				# if there is no existing trackable object, create one
				if to is None:
					to = TrackableObject(objectID, centroid)

				# otherwise, append new centroid
				else:
					to.centroids.append(centroid)

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
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 1)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
				cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 1)

			self.data = {
				"in_direction": self.totalInDir,
				"out_direction": self.totalOutDir,
				"counter": len(objects.items()),
				"social_distancing_v": math.ceil(self.totalDistanceViolations/2),
				"in_frame_time_avg": round(self.peopleTimeAvg, 3),
				"fps": int(self.fpsValue),
			}

			# Publish frame.
			outputFrame_[:] = frame

			# Show the output frame.
			if self.args["VERBOSE"]:
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break

			# Increment frames counter.
			self.totalFrames = (self.totalFrames + 1) % self.args["SKIP_FRAMES"]
				
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
def camaraStream(id):
	#Video streaming route. Put this in the src attribute of an img tag
	return Response(showFrame(int(id)), mimetype='multipart/x-mixed-replace; boundary=frame')

def showFrame(id):
	outputFrame = np.frombuffer(outputFrames[id], dtype=np.uint8)
	outputFrame = outputFrame.reshape(frameShapes[id])
	while True:
		if self.args["FORWARD_CAMERA"]:
			ret, buffer = cv2.imencode('.jpg', outputFrame)
		else:
			ret, buffer = cv2.imencode('.jpg', np.zeros(frameShapes[id], np.uint8))
		frame_ready = buffer.tobytes()
		yield (b'--frame\r\n'
						b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
		time.sleep(1 / 60 ) # Sleep 1/(FPS * 2) 

@app.route('/')
def index():
	"""Video streaming home page."""
	return render_template('indexv4.html', len = len(ARGS["CAMARAIDS"]), camaraIDs = ARGS["CAMARAIDS"])

BaseManager.register("socketManager", SocketIOProcess)
def getManager():
	m = BaseManager()
	m.start()
	return m

if __name__ == '__main__':
	# Initialize Socket Manager.
	manager = getManager()
	socketManager = manager.socketManager(ARGS)

	# Wait till Camaras Info Received.
	while not socketManager.getCamaraInfo():
		pass

	camarasInfo = socketManager.getCamaraInfo()

	for index, camara in enumerate(camarasInfo):
		cap = cv2.VideoCapture(camara["source"])
		ret, frame = cap.read()
		frame = imutils.resize(frame, width=500)
		frameShapes.append(frame.shape)
		cap.release()

		inputFrames.append(Array(ctypes.c_uint8, frameShapes[-1][0] * frameShapes[-1][1] * frameShapes[-1][2], lock=False))
		outputFrames.append(Array(ctypes.c_uint8, frameShapes[-1][0] * frameShapes[-1][1] * frameShapes[-1][2], lock=False))
		flags.append(Value(ctypes.c_bool, False))
		processReference.append(Process(target=CamaraProcessing, args=(index, camara["v_orientation"], camara["run_distance_violation"], camara["detect_just_left_side"], camara["last_record"][0], inputFrames[-1], outputFrames[-1], frameShapes[-1], flags[-1], socketManager, ARGS)))
		processReference[-1].start()
		
		sources.append(camara["source"])

	readProcessRef = Process(target=CamaraRead, args=(sources, inputFrames, frameShapes, flags, ARGS))
	readProcessRef.start()

	from waitress import serve

	app.debug=True
	app.use_reloader=False
	serve(app, host="0.0.0.0", port=8080)
	print("Server 0.0.0.0:8080")	
	socketManager.waitSIO()
