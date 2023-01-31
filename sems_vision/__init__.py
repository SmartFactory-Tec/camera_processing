from multiprocessing import Process, Array, Value
from sems_vision.sems_manager import SemsManager
from sems_vision.camera_processing import CamaraProcessing
from sems_vision.camera_read import CamaraRead
from flask import Flask, render_template, Response
import numpy as np
import imutils
import time
import cv2
import ctypes
from config import load_config

# TODO follow proper flask guidelines
app = Flask(__name__)


@app.route('/camara/<id>')
def camera_stream_route(id):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(show_frame(int(id)), mimetype='multipart/x-mixed-replace; boundary=frame')


def show_frame(id):
    outputFrame = np.frombuffer(output_frames[id], dtype=np.uint8)
    outputFrame = outputFrame.reshape(frame_shapes[id])
    while True:
        if app.config['forward_camera']:
            ret, buffer = cv2.imencode('.jpg', outputFrame)
        else:
            ret, buffer = cv2.imencode('.jpg', np.zeros(frame_shapes[id], np.uint8))
        frame_ready = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_ready + b'\r\n')  # concat frame one by one and show result
        time.sleep(1 / 60)  # Sleep 1/(FPS * 2)


@app.route('/')
def index_route():
    """Video streaming home page."""
    return render_template('indexv4.html', len=len(app.config['camera_ids']), camaraIDs=app.config['camera_ids'])


if __name__ == '__main__':
    config = load_config()

    app.config.from_mapping(config)

    # Initialize shared sems state anager.
    manager = SemsManager()

    manager.start()

    # TODO check if using shared state instead of global variables works
    socket_manager = manager.SocketIOProcess(config)
    process_reference = manager.dict()
    sources = manager.dict()
    frame_shapes = manager.dict()
    input_frames = manager.dict()
    output_frames = manager.dict()
    flags = manager.dict()

    # TODO change into a wait
    # Wait till Camaras Info Received.
    while not socket_manager.get_camera_info():
        pass

    per_camera_info = socket_manager.get_camera_info()

    for index, camara in enumerate(per_camera_info):
        # TODO is doing all of this just to get the frame shape necessary?
        cap = cv2.VideoCapture(camara["source"])
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        frame_shapes.append(frame.shape)
        cap.release()

        input_frames.append(
            Array(ctypes.c_uint8, frame_shapes[-1][0] * frame_shapes[-1][1] * frame_shapes[-1][2], lock=False))
        output_frames.append(
            Array(ctypes.c_uint8, frame_shapes[-1][0] * frame_shapes[-1][1] * frame_shapes[-1][2], lock=False))
        flags.append(Value(ctypes.c_bool, False))
        process_reference.append(Process(target=CamaraProcessing, args=(
            index, camara["v_orientation"], camara["run_distance_violation"], camara["detect_just_left_side"],
            camara["last_record"][0], input_frames[-1], output_frames[-1], frame_shapes[-1], flags[-1], socket_manager,
            config)))
        process_reference[-1].start()

        sources.append(camara["source"])

    readProcessRef = Process(target=CamaraRead, args=(sources, input_frames, frame_shapes, flags, config))
    readProcessRef.start()

    # TODO don't use waitress directly, this should expose a flask API to use with any server
    from waitress import serve

    app.debug = True
    app.use_reloader = False
    serve(app, host="0.0.0.0", port=8080)
    print("Server 0.0.0.0:8080")
    socket_manager.wait()
