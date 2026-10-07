class GazeDetector:
    """
    Определяет приблизительное направление взгляда
    по landmarks глаз, полученным MediaPipe Face Landmarker.

    Результат:
        direction:
            LEFT
            CENTER
            RIGHT
            UP
            DOWN
            UNKNOWN
    """

    # Индексы MediaPipe Face Landmarker
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_IRIS = 468

    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_IRIS = 473

    def __init__(
        self,
        horizontal_threshold=0.35,
        vertical_threshold=0.35
    ):
        self.horizontal_threshold = horizontal_threshold
        self.vertical_threshold = vertical_threshold

    @staticmethod
    def _distance(a, b):
        dx = a.x - b.x
        dy = a.y - b.y
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _horizontal_position(
        iris,
        outer,
        inner
    ):
        width = abs(outer.x - inner.x)

        if width < 0.0001:
            return 0.5

        return (
            iris.x - min(outer.x, inner.x)
        ) / width

    @staticmethod
    def _vertical_position(
        iris,
        top,
        bottom
    ):
        height = abs(bottom.y - top.y)

        if height < 0.0001:
            return 0.5

        return (
            iris.y - min(top.y, bottom.y)
        ) / height

    def detect(self, face_landmarks):
        """
        Анализирует landmarks одного лица.

        Возвращает:
        {
            "direction": "LEFT/RIGHT/CENTER/UP/DOWN/UNKNOWN",
            "horizontal": число,
            "vertical": число
        }
        """

        if face_landmarks is None:
            return {
                "direction": "UNKNOWN",
                "horizontal": 0.5,
                "vertical": 0.5
            }

        required_indices = [
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM,
            self.LEFT_IRIS,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM,
            self.RIGHT_IRIS
        ]

        if len(face_landmarks) <= max(required_indices):
            return {
                "direction": "UNKNOWN",
                "horizontal": 0.5,
                "vertical": 0.5
            }

        left_iris = face_landmarks[self.LEFT_IRIS]
        right_iris = face_landmarks[self.RIGHT_IRIS]

        left_horizontal = self._horizontal_position(
            left_iris,
            face_landmarks[self.LEFT_EYE_OUTER],
            face_landmarks[self.LEFT_EYE_INNER]
        )

        right_horizontal = self._horizontal_position(
            right_iris,
            face_landmarks[self.RIGHT_EYE_OUTER],
            face_landmarks[self.RIGHT_EYE_INNER]
        )

        left_vertical = self._vertical_position(
            left_iris,
            face_landmarks[self.LEFT_EYE_TOP],
            face_landmarks[self.LEFT_EYE_BOTTOM]
        )

        right_vertical = self._vertical_position(
            right_iris,
            face_landmarks[self.RIGHT_EYE_TOP],
            face_landmarks[self.RIGHT_EYE_BOTTOM]
        )

        horizontal = (
            left_horizontal + right_horizontal
        ) / 2

        vertical = (
            left_vertical + right_vertical
        ) / 2

        # Сначала проверяем вертикальное отклонение.
        if vertical < self.vertical_threshold:
            direction = "UP"

        elif vertical > 1.0 - self.vertical_threshold:
            direction = "DOWN"

        # Затем горизонтальное.
        elif horizontal < self.horizontal_threshold:
            direction = "LEFT"

        elif horizontal > 1.0 - self.horizontal_threshold:
            direction = "RIGHT"

        else:
            direction = "CENTER"

        return {
            "direction": direction,
            "horizontal": round(horizontal, 3),
            "vertical": round(vertical, 3)
        }

    def detect_from_face_result(self, face_result):
        """
        Получает результат FaceDetector.

        Если лицо одно — анализирует его взгляд.
        Если лиц нет или их несколько — возвращает UNKNOWN.
        """

        if face_result["face_count"] != 1:
            return {
                "direction": "UNKNOWN",
                "horizontal": 0.5,
                "vertical": 0.5
            }

        landmarks = face_result["landmarks"][0]

        return self.detect(landmarks)
