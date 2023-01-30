import os
from multiprocessing import Process, Array, Value
from multiprocessing.managers import BaseManager

from sems_vision.socket_io_process import SocketIOProcess
from sems_vision.centroid_tracker import CentroidTracker
from sems_vision.trackable_object import TrackableObject
from sems_vision.camera_processing import CamaraProcessing
from sems_vision.camera_read import CamaraRead
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
from config import load_config

# TODO follow proper flask guidelines
app = Flask(__name__)






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
        if app.config['forward_camera']:
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
    return render_template('indexv4.html', len=len(app.config['camera_ids']), camaraIDs=app.config['camera_ids'])


BaseManager.register("socket_manager", SocketIOProcess)


def get_manager():
    m = BaseManager()
    m.start()
    return m


if __name__ == '__main__':
    config = load_config()

    app.config.from_mapping(config)

    # Initialize Socket Manager.
    manager = get_manager()
    socketManager = manager.socket_manager(config)

    # TODO change into a wait
    # Wait till Camaras Info Received.
    while not socketManager.get_camera_info():
        pass

    per_camera_info = socketManager.get_camera_info()

    for index, camara in enumerate(per_camera_info):
        # TODO is doing all of this just to get the frame shape necessary?
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
            config)))
        processReference[-1].start()

        sources.append(camara["source"])

    readProcessRef = Process(target=CamaraRead, args=(sources, inputFrames, frameShapes, flags, config))
    readProcessRef.start()

    # TODO don't use waitress directly, this should expose a flask API to use with any server
    from waitress import serve

    app.debug = True
    app.use_reloader = False
    serve(app, host="0.0.0.0", port=8080)
    print("Server 0.0.0.0:8080")
    socketManager.wait()
