import cv2
import numpy as np
from sort import SortTracker
from detection import Detector


def detect(
    confidence_threshold: float = 0.5,
    nms_threshold: float = 0.2,
    max_age: int = 10,
    min_hits: int = 3,
    iou_threshold: float = 0.2,
    objects: list[str] = ["person", "car"],
    cap_path: str | int = 0,
) -> str:
    try:
        cap = cv2.VideoCapture(cap_path)
        if not cap.isOpened():
            print("Ошибка: не удалось открыть видео поток")
            exit()

        detector = Detector(
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            objects=objects,
        )

        tracker = SortTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracking_frame = frame.copy()
            detections = detector.get_detections(frame)
            detections_frame = detector.plot_detections(detections, frame.copy())

            if detections:
                det_array = []
                for det in detections:
                    x, y, w, h = det.ltwh
                    confidence = det.confidence
                    class_id = 0 if det.class_name == "person" else 1
                    det_array.append([x, y, x + w, y + h, class_id, confidence])
                
                # Обновление трекера
                tracked_objects = tracker.update(np.array(det_array), None)

                # Отрисовка результатов трекинга
                for track in tracked_objects:
                    x1, y1, x2, y2, track_id, class_id, confidence = track
                    color = detector.convert_class_id_to_color(int(class_id))
                    cv2.rectangle(tracking_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(
                        tracking_frame,
                        f"ID: {int(track_id)}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )

            cv2.imshow("Трекинг", tracking_frame)
            cv2.imshow("Детекиция", detections_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    detect()
