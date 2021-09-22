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
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import requests
import threading

GPU_AVAILABLE = True
CONFIDENCE_ = 0.3
NMS_THRESH = 0.3

app = Flask(__name__)
last_frames = deque( maxlen=120 )

# defining the api-endpoint  
API_ENDPOINT = "https://prod-64.westus.logic.azure.com:443/workflows/ff179f5e08284d08b4fcb35a025443a0/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=g9033T7EODjg-A4GivUtOxNsLj08gWHomND-ALQXX1g"

# data to be sent to api 
data = {
	"cantidad" : 0,
	"lugar" : "Camara 0"
} 

def callPost():
	threading.Timer(3.0,callPost).start()
	# sending post request and saving response as response object 
	if(data["cantidad"] != 0 ):
		r = requests.post(API_ENDPOINT, json=data) 
  
#Post Call
#callPost()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = None
with open('yolo/coco.names', 'r') as f:
	CLASSES = [line.strip() for line in f.readlines()]

# load our serialized model from disk
print("[INFO] loading model...")
# Load Model
net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
if GPU_AVAILABLE:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	#input video (file)
	vs = cv2.VideoCapture(args["input"])
	vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
	#input video (local ip)

	#vs = cv2.VideoCapture('http://192.168.1.71:8080/video')

	#input video (public ip) MODIFY THIS LINE
	#vs = cv2.VideoCapture('')

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# Define the frames per second throughput estimator
fps = None

def generate_boxes_confidences_classids( outs, threshold):
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
					box = np.array(detection[0:4]) * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					startX = int(centerX - (width / 2))
					startY = int(centerY - (height / 2))

					# Append to list
					boxes.append([startX, startY, int(width), int(height)])
					confidences.append(float(confidence))
					classids.append(classid)

		return boxes, confidences, classids

def gen_frames():
	global args
	global CLASSES
	global data
	global net
	global vs
	global writer
	global fps
	global totalFrames
	global totalDown
	global totalUp
	global ct
	global trackers
	global trackableObjects
	global W
	global H
	global last_frames
	fps = FPS().start()
	# loop over frames from the video stream
	while True:
		startAll = time.time()
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if args["input"] is not None and frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width=500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
			net.setInput(blob)
			
			start = time.time()
			detections = net.forward(layer_names)
			end = time.time()
			print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

			boxes, confidences, classids = generate_boxes_confidences_classids(detections, CONFIDENCE_)

			idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_, NMS_THRESH)
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
						if CLASSES[idx] != "person":
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
						trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

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
		cv2.line(frame, (W//2, 0), (W // 2, H), (0, 255, 255), 2)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)["centroid"]

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			data = {
				"cantidad": len(objects.items()),
				"lugar": "Camara 0"
			}
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

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
					if direction < 0 and centroid[0] < W // 2:
						totalUp += 1
						to.counted = True

					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, count the object
					elif direction > 0 and centroid[0] > W // 2:
						totalDown += 1
						to.counted = True

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		# construct a tuple of information we will be displaying on the
		# frame
		info = [
			("Izq", totalUp),
			("Der", totalDown),
			("Status", status),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		#Publish frame in Server
		last_frames.append(frame)

		# show the output frame
		# cv2.imshow("Frame", frame)
		# key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		#	break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()
		endAll = time.time()
		print ("[INFO] Iteration took {:6f} seconds".format(endAll - startAll))

	if args["output"] is None and writer is None:
		fps.stop()
		print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		print("\n\n\n\n")
		vs = cv2.VideoCapture(args["input"])
		vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
		gen_frames()
	else:
		end_process()

def end_process():
	global args
	global CLASSES
	global net
	global vs
	global writer
	global fps
	global totalFrames
	global totalDown
	global totalUp
	global ct
	global trackers
	global trackableObjects
	global W
	global H
	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# check to see if we need to release the video writer pointer
	if writer is not None:
		writer.release()

	# if we are not using a video file, stop the camera video stream
	if not args.get("input", False):
		vs.stop()

	# otherwise, release the video file pointer
	else:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()

def show_frames():
	global last_frames
	global H
	global W
	while True:
		if(len(last_frames)>=1):
			ret, buffer = cv2.imencode('.jpg', last_frames[0])
			frame_ready = buffer.tobytes()
			yield (b'--frame\r\n'
							b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
		else:
			ret, buffer = cv2.imencode('.jpg', np.zeros((H,W,3), np.uint8))
			frame_ready = buffer.tobytes()
			yield (b'--frame\r\n'
							b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
					
@app.route('/camara0')
def camara0():
	#Video streaming route. Put this in the src attribute of an img tag
	return Response(show_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
	"""Video streaming home page."""
	return render_template('indexv1.html')


if __name__ == '__main__':
	genFrameTread = threading.Thread(target=gen_frames)
	genFrameTread.start()
	time.sleep(2)
	from waitress import serve
	serve(app, host="0.0.0.0", port=8080)
	