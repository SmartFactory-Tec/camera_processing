from multiprocessing.managers import BaseManager
from sems_vision.socket_io_process import SocketIOProcess
from sems_vision.frame_store import FrameStore


class SemsStateManager(BaseManager):
    pass


SemsStateManager.register('FrameStore', FrameStore)
SemsStateManager.register('list', list)
SemsStateManager.register('dict', dict)
SemsStateManager.register('SocketIOProcess', SocketIOProcess)