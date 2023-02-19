import time
import cv2
import numpy as np
from flask import Blueprint, Response, current_app
from sems_vision.frame_store import FrameStore
from dataclasses import dataclass


@dataclass
class Camera:
    id: int
    name: str
    connection_string: str
    location_text: str
    location_id: str


def construct_camera_blueprint(frame_store: FrameStore) -> Blueprint:
    bp = Blueprint('camera', __name__, url_prefix='/camera')

    @bp.route('/<id>')
    def camera_stream_route(id):
        # Video streaming route. Put this in the src attribute of an img tag
        return Response(show_frame(int(id)), mimetype='multipart/x-mixed-replace; boundary=frame')

    def show_frame(frame_id):
        output_frame = frame_store.get_output_frame_ref(frame_id)

        while True:
            if current_app.config['forward_camera']:
                ret, buffer = cv2.imencode('.jpg', output_frame)
            else:
                ret, buffer = cv2.imencode('.jpg', np.zeros(output_frame.shape, output_frame.dtype))

            encoded_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')  # concat frame one by one and show result
            time.sleep(1 / 60)  # Sleep 1/(FPS * 2)

    return bp
