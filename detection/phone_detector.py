from ultralytics import YOLO


class PhoneDetector:
    """
    Детектор смартфона на основе YOLO.
    Не работает с камерой самостоятельно.
    Получает готовый кадр и возвращает информацию
    об обнаруженном смартфоне.
    """

    PHONE_CLASS_NAME = "cell phone"

    def __init__(self, model_path="yolo11n.pt", confidence=0.50):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Анализирует один кадр.

        Возвращает:
            {
                "detected": True/False,
                "confidence": число,
                "box": (x1, y1, x2, y2)
            }
        """

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        best_detection = None

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                if class_name != self.PHONE_CLASS_NAME:
                    continue

                confidence = float(box.conf[0])

                coordinates = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coordinates)

                if (
                    best_detection is None
                    or confidence > best_detection["confidence"]
                ):
                    best_detection = {
                        "detected": True,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2)
                    }

        if best_detection is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "box": None
            }

        return best_detection