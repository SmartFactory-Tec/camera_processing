import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional
from math import floor
from camera_processing.frame_packet import FramePacketGenerator
from camera_processing.detection import Detection, BoundingBox


class YoloV8DetectionProcessor:
    def __init__(self, confidence_threshold: float, nms_threshold: float, process_height: Optional[int] = 640):
        self.__model = YOLO('yolov8n.pt')

        self.__process_height: Optional[int] = process_height

        self.nms_threshold = nms_threshold

        self.confidence_threshold = confidence_threshold

    def process(self, source: FramePacketGenerator, skip_frames=4, detections_key='detections'):
        frame_counter = 1
        for packet in source:
            frame_counter += 1
            frame_counter %= skip_frames
            if frame_counter != 0:
                yield packet
                continue

            frame: np.ndarray = packet.frame
            resize_factor: Optional[float] = None

            # resize if set to do so
            if self.__process_height is not None:
                resize_factor = self.__process_height / frame.shape[0]
                frame = cv2.resize(frame, (
                    floor(frame.shape[1] * resize_factor), floor(frame.shape[0] * resize_factor)),
                                   interpolation=cv2.INTER_LINEAR)

            predictions = self.__model.predict(source=frame, verbose=False)[0].boxes

            bounding_boxes: list[BoundingBox] = []
            confidences: list[float] = []

            for prediction in predictions:
                if prediction.cls.item() != 0:
                    continue
                bb = prediction.xyxy[0]
                bounding_boxes.append(
                    (floor(bb[0].item()), floor(bb[1].item()), floor(bb[2].item()), floor(bb[3].item())))
                confidences.append(prediction.conf.item())

            if resize_factor is not None:
                bounding_boxes = list([tuple((floor(coord / resize_factor) for coord in bb)) for bb in bounding_boxes])

            nms_bbs = list([(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]) for bb in bounding_boxes])
            # filter detections via the Non-Maximal Suppression algorithm
            filtered_detection_indices = cv2.dnn.NMSBoxes(nms_bbs, confidences, self.confidence_threshold,
                                                          self.nms_threshold)

            detections: list[Detection] = []

            # loop over the indexes we are keeping
            for idx in filtered_detection_indices:
                detection = Detection(bounding_boxes[idx], confidences[idx])
                detections.append(detection)

            packet.values[detections_key] = detections

            yield packet
