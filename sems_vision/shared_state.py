from multiprocessing import Event
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from multiprocessing.managers import BaseManager, SharedMemoryManager
from sems_vision.socket_io_process import SocketIOProcess


class SharedFrame:
    def __init__(self, shape: tuple, dtype: type, shared_mem: SharedMemory):
        self._shape = shape
        self._dtype = dtype
        self._shared_mem = shared_mem

        self.new_frame_event = Event()

    def get_ndarray(self) -> np.ndarray:
        return

    def new_frame_is_set(self):
        return self.new_frame_event.is_set()

    def set_new_frame(self):
        self.new_frame_event.set()

    def unset_new_frame(self):
        self.new_frame_event.clear()

    def wait_for_new_frame(self, timeout=None):
        self.new_frame_event.wait(timeout)


class SemsStateManager(SharedMemoryManager):
    pass


# SemsStateManager.register('SharedFrame', SharedFrame)
