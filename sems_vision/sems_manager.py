from multiprocessing.managers import BaseManager
from socket_io_process import SocketIOProcess


class SemsManager(BaseManager):
    pass


SemsManager.register('dict', dict)
SemsManager.register('SocketIOProcess', SocketIOProcess)
