import socket

import socketio


class SocketIOProcess:
    sio = socketio.Client()

    def __init__(self, args):
        self.args = args
        self.camera_ids = self.args["camera_ids"]
        self.camera_count = len(self.camera_ids)
        self.per_camera_info = []
        self.is_connected = False
        self.has_camera_info = False

        self.sio.on('connect', self.connect)
        self.sio.on('disconnect', self.disconnect)
        self.sio.on('visionInit', self.init_vision)
        self.sio.connect(self.args["back_endpoint"])

    def connect(self):
        print('Connected')
        self.is_connected = True
        self.sio.emit('visionInit', self.camera_ids)

    def disconnect(self):
        print('Disconnected')
        self.is_connected = False
        self.has_camera_info = False
        self.per_camera_info = []

    def wait(self):
        self.sio.wait()

    def init_vision(self, per_camera_info):
        self.per_camera_info = per_camera_info
        self.has_camera_info = True
        print('CamaraInfo ', per_camera_info)

    def get_camera_info(self, camera_id=None):
        if not self.has_camera_info:
            return False

        if not camera_id:
            return self.per_camera_info

        return self.per_camera_info[camera_id]

    def send_camera_data(self, camera_id, data):
        if self.is_connected:
            self.sio.emit('visionPost', data=(
                self.per_camera_info[camera_id]['camera_id'],
                data["in_direction"],
                data["out_direction"],
                data["counter"],
                data["social_distancing_v"],
                data["in_frame_time_avg"],
                data["fps"],
            ))

    def set_camera_url(self, camera_id):
        if self.is_connected:
            if self.args["ngrok_available"] and self.args["forward_camera"]:
                endpoint = 'http://sems.ngrok.io/camara/'
            elif self.args["forward_camera"]:
                endpoint = 'http://' + socket.getfqdn() + ':8080/camara/'
            else:
                endpoint = ''

            self.sio.emit('updateCamara', data=(
                self.per_camera_info[camera_id]['camera_id'],
                endpoint + str(camera_id)
            ))
