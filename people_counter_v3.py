# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from collections import deque
from flask import Flask, render_template, Response
from imutils.video import VideoStream
from imutils.video import FPS
from dotenv import load_dotenv
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


# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", type=str, default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str, default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.22,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=35,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

load_dotenv()
app = Flask(__name__)

with open('inputScript_TestVideo.json') as inputScript:
  inputSources = json.load(inputScript)

camaras = []

class Camara:
	global args

	API_ENDPOINT = os.getenv("API_ENDPOINT")
	
	# Initialize list of class labels MobileNet SSD was trained to detect
	CLASSES = None
	with open('yolo/coco.names', 'r') as f:
		CLASSES = [line.strip() for line in f.readlines()]
	
	camaraCounter = 0

	def __init__(self, inputSource):
		Camara.camaraCounter += 1
  	
		self.camaraId = "Camara" + str(Camara.camaraCounter)
		
		self.last_frames = deque( maxlen=120 )
		
		# Load Model
		self.net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
  	
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

		self.data = {
			"cantidad" : 0,
			"lugar" : self.camaraId
		}

		# Start the frames per second throughput estimator
		self.fps = None
		self.callFps()
		
		genFrameTread = threading.Thread(target=self.gen_frames, args=())
		genFrameTread.start()

		self.callPost()
	
	def callPost(self):
  		
		callPostThread = threading.Timer(3.0, self.callPost, args=())
		callPostThread.start()

		# Sending post request and saving response as response object.
		if self.data["cantidad"] != 0 :
			requests.post(Camara.API_ENDPOINT, json=self.data)

	def callFps(self):	
		if self.fps != None:
			self.fps.stop()
			print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
			self.fpsValue = self.fps.fps()

		self.fps = FPS().start()
		
		callFpsThread = threading.Timer(2.0, self.callFps, args=())
		callFpsThread.start()

  
	def gen_frames(self):
		# Loop over frames from the video stream.
		while True:
			# Grab the next frame and handle if we are reading from either
			# VideoCapture or VideoStream.
			frame = self.vs.read()
			frame = frame[1]

			if frame is None:
				break

			# Resize the frame to have a maximum width of 500 pixels (the
			# less data we have, the faster we can process it), then convert
			# the frame from BGR to RGB for dlib.
			frame = imutils.resize(frame, width=500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# if the frame dimensions are empty, set them
			if self.W is None or self.H is None:
				(self.H, self.W) = frame.shape[:2]

			# if we are supposed to be writing a video to disk, initialize
			# the writer
			if args["output"] is not None and self.writer is None:
				self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				self.writer = cv2.VideoWriter(args["output"], fourcc, 30,
					(self.W, self.H), True)

			# initialize the current status along with our list of bounding
			# box rectangles returned by either (1) our object detector or
			# (2) the correlation trackers
			self.status = "Waiting"
			rects = []

			# check to see if we should run a more computationally expensive
			# object detection method to aid our tracker
			if self.totalFrames % args["skip_frames"] == 0:
				# set the status and initialize our new set of object trackers
				self.status = "Detecting"
				self.trackers = []

				# convert the frame to a blob and pass the blob through the
				# network and obtain the detections
				blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 127.5, swapRB=True, crop=False)
				self.net.setInput(blob)

				layer_names = self.net.getLayerNames()
				output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

				detections = self.net.forward(output_layers)

				# loop over the detections
				for i in np.arange(0, detections.shape[2]):
					# extract the confidence (i.e., probability) associated
					# with the prediction
					confidence = detections[0, 0, i, 2]

					# filter out weak detections by requiring a minimum
					# confidence
					if confidence > args["confidence"]:
						# extract the index of the class label from the
						# detections list
						idx = int(detections[0, 0, i, 1])

						# if the class label is not a person, ignore it
						if Camara.CLASSES[idx] != "person":
							continue

						# compute the (x, y)-coordinates of the bounding box
						# for the object
						box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
						(startX, startY, endX, endY) = box.astype("int")

						# construct a dlib rectangle object from the bounding
						# box coordinates and then start the dlib correlation
						# tracker
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
			objects = self.ct.update(rects)["centroid"]

			# loop over the tracked objects
			for (objectID, centroid) in objects.items():
				self.data = {
					"cantidad": len(objects.items()),
					"lugar": "Camara 0"
				}
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
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			# construct a tuple of information we will be displaying on the
			# frame
			info = [
				("Der a Izq", self.totalUp),
				("Izq a Der", self.totalDown),
				("Status", self.status),
				("FPS", int(self.fpsValue))
			]

			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			# check to see if we should write the frame to disk
			if self.writer is not None:
				self.writer.write(frame)

			#Publish frame in Server
			self.last_frames.append(frame)

			# show the output frame
			# cv2.imshow("Frame", frame)
			# key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			# if key == ord("q"):
			# 	break

			# increment the total number of frames processed thus far and
			# then update the FPS counter
			self.totalFrames += 1
			self.fps.update()
		
		if "videos" in self.inputSource:
			self.end_process()
		else:
			self.vs = cv2.VideoCapture(self.inputSource)
			self.vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
			self.gen_frames()

	def end_process(self):
		# check to see if we need to release the video writer pointer
		if self.writer is not None:
			self.writer.release()

		self.vs.release()

		# close any open windows
		cv2.destroyAllWindows()

	def show_frames(self):
		while True:
			if(len(self.last_frames)>=1):
				ret, buffer = cv2.imencode('.jpg', self.last_frames[0])
				frame_ready = buffer.tobytes()
				yield (b'--frame\r\n'
								b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
			else:
				ret, buffer = cv2.imencode('.jpg', np.zeros((self.H,self.W,3), np.uint8))
				frame_ready = buffer.tobytes()
				yield (b'--frame\r\n'
								b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
		time.sleep(1 / self.fpsValue * 2)

@app.route('/camara/<id>')
def camaraStream(id):
	global camaras
	#Video streaming route. Put this in the src attribute of an img tag
	return Response(camaras[int(id)].show_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
			camaras.append(Camara(camara["src"]))
			camara["refI"] = refI
			camara["refE"] = refE
			refI += 1
			refE += 1

	print("Server 0.0.0.0:8080")	
	from waitress import serve
	app.debug=True
	app.use_reloader=False
	serve(app, host="0.0.0.0", port=8080)
