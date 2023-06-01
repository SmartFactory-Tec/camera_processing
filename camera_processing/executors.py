import random
from typing import Callable

import cv2

from .frame_packet import FramePacketGenerator


def pipeline_executor(source: FramePacketGenerator) -> Callable:
    """
    Creates a pipeline executor that accepts frames as fast as possible until end of stream
    :param source: pipeline source
    :return: executor function
    """

    def executor():
        nonlocal source
        for _ in source:
            continue

    return executor


def imshow_pipeline_executor(source: FramePacketGenerator, unique_id: int) -> Callable:
    """
    Creates a pipeline executor that accepts frames as fast as possible and shows them using
    cv2's imshow
    :param source: pipeline source
    :return:  executor function
    """

    def executor():
        nonlocal source
        window_id = f'imshow_executor{unique_id}'
        for packet in source:
            cv2.imshow(window_id, packet.frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                return

        cv2.destroyWindow(window_id)

    return executor
