from dataclasses import dataclass

import cv2


@dataclass
class Detection:
    ltwh: list[int, int, int, int]
    confidence: float
    class_name: str


class Detector:
    def __init__(
        self, confidence_threshold: float, nms_threshold: float, objects: list[str]
    ):
        self._init_class_names()
        self._init_net()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.objects = objects
        self.object_to_color = {"person": (0, 255, 0), "vehicle": (255, 0, 0)}

    def _init_class_names(self):
        class_names_path = "./weights/coco.names"
        with open(class_names_path, "rt") as f:
            self.classNames: list[str] = f.read().rstrip("\n").split("\n")

    def _init_net(self):
        config_path = "./weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        weights_path = "./weights/frozen_inference_graph.pb"

        self.net = cv2.dnn_DetectionModel(weights_path, config_path)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def get_detections(self, frame) -> list[Detection]:
        class_ids, confs, bbox = self.net.detect(
            frame,
            confThreshold=self.confidence_threshold,
            nmsThreshold=self.nms_threshold,
        )
        detections = []
        for class_id, confidence, box in zip(class_ids, confs, bbox):
            class_name = self.classNames[class_id - 1]
            if class_name in self.objects and confidence > self.confidence_threshold:
                x1, y1, w, h = box
                detection = Detection(
                    ltwh=[x1, y1, w, h], confidence=confidence, class_name=class_name
                )
                detections.append(detection)

        return detections

    def convert_class_name_to_color(self, object: str) -> tuple[int, int, int]:
        object = "vehicle" if object not in self.object_to_color else object
        return self.object_to_color.get(object)
    
    def convert_class_id_to_color(self, class_id: int) -> tuple[int, int, int]:
        class_name = self.classNames[class_id]
        return self.convert_class_name_to_color(class_name)

    def plot_detections(
        self, detections: list[Detection], frame: cv2.typing.MatLike
    ) -> cv2.typing.MatLike:
        for detection in detections:
            color = self.convert_class_name_to_color(detection.class_name)
            cv2.rectangle(frame, detection.ltwh, color, 2)

            cv2.putText(
                frame,
                f"{detection.class_name} {detection.confidence:.2f}",
                (detection.ltwh[0], detection.ltwh[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame

    @staticmethod
    def convert_detections_to_raw_list(
        detections: list[Detection],
    ) -> list[tuple[list[int, int, int, int], float, int]]:
        return [
            (detection.ltwh, detection.confidence, detection.class_name)
            for detection in detections
        ]
