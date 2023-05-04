import cv2
import numpy as np
from math import floor
from sems_vision.frame_packet import FramePacketGenerator
from sems_vision.detection import Detection, BoundingBox
import pandas as pd


class YoloV5DetectingProcessor:
    def __init__(self, confidence_threshold: float, nms_threshold: float, process_height: int | None = 300):
        self.__model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

        self.__process_height: int | None = process_height

        self.nms_threshold = nms_threshold

        self.confidence_threshold = confidence_threshold

    def process(self, source: FramePacketGenerator, skip_frames=4, detections_key='detections'):
        frame_counter = 1
        for packet in source:
            frame_counter %= skip_frames
            if frame_counter == 0:
                yield packet
            frame_counter += 1

            frame: np.ndarray = packet.frame
            resize_factor: float | None = None

            # resize if set to do so
            if self.__process_height is not None:
                resize_factor = self.__process_height / frame.shape[0]
                frame = cv2.resize(frame, (
                    floor(frame.shape[1] * resize_factor), floor(frame.shape[0] * resize_factor)),
                                          interpolation=cv2.INTER_AREA)

            detections_dataframe: pd.DataFram = self.__model([frame]).pandas().xyxy

            bounding_boxes: list[BoundingBox] = []
            confidences: list[float] = []
            for idx, detection in enumerate(detections_dataframe):
                people_detected = detection[detection['class'] == 0]

                x = people_detected['xmin'].tolist()
                y = people_detected['ymin'].tolist()
                w = (people_detected['xmax'] - people_detected['xmin']).tolist()
                h = (people_detected['ymax'] - people_detected['ymin']).tolist()

                if resize_factor is not None:
                    x = [coord / resize_factor for coord in x]
                    y = [coord / resize_factor for coord in y]
                    w = [coord / resize_factor for coord in w]
                    h = [coord / resize_factor for coord in h]

                bounding_boxes = list(zip(x, y, w, h))
                confidences = people_detected['confidence'].tolist()

            # TODO check if its better to run this before or after removing all non person detections
            # filter detections via the Non Maximal Suppression algorithm
            filtered_detection_indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.confidence_threshold,
                                                          self.nms_threshold)

            # transform them from x1 y1 w h to x1 y1 x2 y2
            bounding_boxes = list([(bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]) for bb in bounding_boxes])

            for bounding_box in bounding_boxes:
                cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0))

            detections: list[Detection] = []

            # loop over the indexes we are keeping
            for idx in filtered_detection_indices:
                bounding_box = bounding_boxes[idx]
                confidence = confidences[idx]
                detection = Detection(bounding_box, confidence)
                detections.append(detection)

            packet.values[detections_key] = detections

            yield packet
