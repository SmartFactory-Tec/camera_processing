from typing import Callable

import cv2

from sems_vision import FramePacketGenerator


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


def imshow_pipeline_executor(source: FramePacketGenerator) -> Callable:
    """
    Creates a pipeline executor that accepts frames as fast as possible and shows them using
    cv2's imshow
    :param source: pipeline source
    :return:  executor function
    """

    def executor():
        nonlocal source
        for packet in source:
            cv2.imshow('imshow_executor', packet.frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    return executor
