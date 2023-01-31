from flask import Blueprint, Response

bp = Blueprint('camera', __name__, url_prefix='/camera')


@bp.route('/<id>')
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
