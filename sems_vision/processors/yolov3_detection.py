from math import floor

import cv2
import numpy as np

from sems_vision.frame_packet import FramePacketGenerator
from sems_vision.detection import Detection, BoundingBox


class YoloV3DetectionProcessor:
    PERSON_CLASS_ID = 0

    def __init__(self, confidence_threshold: float, nms_threshold: float):
        self._net = cv2.dnn.readNetFromDarknet('models/people/yolov3.cfg', 'models/people/yolov3.weights')
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.nms_threshold = nms_threshold

        self._output_layer_names = self._net.getUnconnectedOutLayersNames()

        self.confidence_threshold = confidence_threshold

    def process(self, source: FramePacketGenerator, skip_frames=4, detections_key='detections'):
        frame_counter = 1
        for packet in source:
            frame_counter %= skip_frames
            if frame_counter == 0:
                yield packet
            frame_counter += 1

            frame = packet.frame
            frame_shape = frame.shape
            resize_factor = 300 / frame.shape[0]
            scaled_frame = cv2.resize(frame, (floor(frame.shape[0] * resize_factor), floor(frame.shape[1] * resize_factor)), interpolation=cv2.INTER_AREA)
            scaled_shape = scaled_frame.shape


            detections: list[Detection] = []

            blob = cv2.dnn.blobFromImage(scaled_frame, 1 / 255.0, scaled_shape, swapRB=True, crop=False)

            self._net.setInput(blob)

            network_out = self._net.forward(self._output_layer_names)

            bounding_boxes: list[BoundingBox] = []
            confidences: list[float] = []

            # TODO there should only be one layer so no need for a loop, verify
            for output in network_out:
                for raw_detections in output:
                    # Get the scores, class_id, and the confidence of the prediction
                    scores = raw_detections[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold and class_id == YoloV3DetectionProcessor.PERSON_CLASS_ID:
                        # compute the (x, y)-coordinates of the bounding box
                        # for the object
                        box = np.array(raw_detections[0:4]) * np.array(
                            [frame_shape[0], frame_shape[1], frame_shape[0], frame_shape[1]])
                        (center_x, center_y, width, height) = box.astype("int")

                        x1 = int(center_x - (width / 2))
                        y1 = int(center_y - (height / 2))

                        bounding_box = (x1, y1, width, height)

                        # Append to list
                        bounding_boxes.append(bounding_box)
                        confidences.append(float(confidence))

            # TODO check if its better to run this before or after removing all non person detections
            # filter detections via the Non Maximal Suppression algorithm
            filtered_detection_indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.confidence_threshold,
                                                          self.nms_threshold)

            # transform them from x1 y1 w h to x1 y1 x2 y2
            bounding_boxes = [(bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]) for bb in bounding_boxes]

            # loop over the indexes we are keeping
            for idx in filtered_detection_indices:
                bounding_box = bounding_boxes[idx]
                confidence = confidences[idx]
                detection = Detection(bounding_box, confidence)
                detections.append(detection)

            packet.values[detections_key] = detections

            yield packet
