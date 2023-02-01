from multiprocessing.shared_memory import SharedMemory
import numpy as np
from typing import Dict


class FrameStore:
    MEMORY_PREFIX = 'SEMS_VISION_'

    def __init__(self):
        self.frame_shapes: Dict[str, tuple] = {}
        self.frame_dtypes: Dict[str, np.dtype] = {}

    def _create_shared_ndarray(self, mem_name, shape: tuple, dtype: np.dtype) -> np.ndarray:
        if mem_name in self.frame_shapes or mem_name in self.frame_dtypes:
            raise FrameExistsError()

        self.frame_shapes[mem_name] = shape
        self.frame_dtypes[mem_name] = dtype

        d_size = np.dtype(dtype).itemsize * np.prod(shape)
        shared_mem = SharedMemory(create=True, size=d_size, name=mem_name)

        return np.ndarray(shape=shape, dtype=dtype, buffer=shared_mem.buf)

    def register_input_frame(self, frame_id: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """
        Creates a shared memory numpy array to store a frame and returns a reference to it
        @param frame_id: integer ID to identify this frame's shared memory
        @param shape: shape of the shared memory array
        @param dtype: dtype of the shared memory array
        @return: ndarray wrapper that uses this shared memory
        """
        memory_name = self.MEMORY_PREFIX + 'IN_FRAME_' + str(frame_id)
        return self._create_shared_ndarray(memory_name, shape, dtype)

    def register_output_frame(self, frame_id: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """
        Creates a shared memory numpy array to store a frame and returns a reference to it
        @param frame_id: integer ID to identify this frame's shared memory
        @param shape: shape of the shared memory array
        @param dtype: dtype of the shared memory array
        @return: ndarray wrapper that uses this shared memory
        """
        memory_name = self.MEMORY_PREFIX + 'OUT_FRAME_' + str(frame_id)
        return self._create_shared_ndarray(memory_name, shape, dtype)

    def get_input_frame_ref(self, frame_id: int) -> np.ndarray:
        """
        Returns a reference to a previously created shared memory ndarray storing a frame
        @param frame_id: id of the frame's shared memory
        @return: ndarray wrapper that uses this shared memory
        """
        memory_name = self.MEMORY_PREFIX + 'IN_FRAME_' + str(frame_id)

        if memory_name not in self.frame_shapes or memory_name not in self.frame_dtypes:
            raise FrameDoesNotExistError

        shared_mem = SharedMemory(name=memory_name)

        frame = np.ndarray(self.frame_shapes[memory_name], dtype=self.frame_dtypes[memory_name], buffer=shared_mem.buf)

        return frame

    def get_output_frame_ref(self, frame_id: int) -> np.ndarray:
        """
        Returns a reference to a previously created shared memory ndarray storing a frame
        @param frame_id: id of the frame's shared memory
        @return: ndarray wrapper that uses this shared memory
        """
        memory_name = self.MEMORY_PREFIX + 'OUT_FRAME_' + str(frame_id)

        if memory_name not in self.frame_shapes or memory_name not in self.frame_dtypes:
            raise FrameDoesNotExistError

        shared_mem = SharedMemory(name=memory_name)

        frame = np.ndarray(self.frame_shapes[memory_name], dtype=self.frame_dtypes[memory_name], buffer=shared_mem.buf)

        return frame


class FrameExistsError(Exception):
    pass


class FrameDoesNotExistError(Exception):
    pass
