import time


class ViolationManager:
    """
    Подтверждает нарушения перед записью в журнал.

    Обрабатывает:
    - отсутствие лица;
    - несколько лиц;
    - смартфон;
    - длительное отклонение взгляда;
    - длительное отклонение головы.
    """

    FACE_MISSING_CONFIRMATION = 2.0
    MULTIPLE_FACES_CONFIRMATION = 1.0
    PHONE_CONFIRMATION = 1.5

    GAZE_CONFIRMATION = 2.0
    HEAD_CONFIRMATION = 2.0

    def __init__(self, logger):
        self.logger = logger

        self.face_missing_start = None
        self.multiple_faces_start = None
        self.phone_start = None
        self.gaze_start = None
        self.head_start = None

        self.face_missing_logged = False
        self.multiple_faces_logged = False
        self.phone_logged = False
        self.gaze_logged = False
        self.head_logged = False

        self.total_violations = 0

    def process(
        self,
        face_result,
        phone_result,
        gaze_result,
        head_result
    ):
        """
        Анализирует результаты детекторов
        и записывает подтверждённые нарушения.
        """

        now = time.monotonic()

        self._process_face_missing(
            face_result,
            now
        )

        self._process_multiple_faces(
            face_result,
            now
        )

        self._process_phone(
            phone_result,
            now
        )

        self._process_gaze(
            face_result,
            gaze_result,
            now
        )

        self._process_head(
            face_result,
            head_result,
            now
        )

    def _record(
        self,
        event_type,
        confidence=None,
        details=None
    ):
        self.logger.log_event(
            event_type=event_type,
            confidence=confidence,
            details=details
        )

        self.total_violations += 1

    def _process_face_missing(
        self,
        face_result,
        now
    ):
        face_count = face_result["face_count"]

        if face_count == 0:

            if self.face_missing_start is None:
                self.face_missing_start = now

            duration = now - self.face_missing_start

            if (
                duration >= self.FACE_MISSING_CONFIRMATION
                and not self.face_missing_logged
            ):

                self._record(
                    event_type="FACE_NOT_DETECTED",
                    details={
                        "duration_seconds": round(
                            duration,
                            2
                        )
                    }
                )

                self.face_missing_logged = True

        else:

            self.face_missing_start = None
            self.face_missing_logged = False

    def _process_multiple_faces(
        self,
        face_result,
        now
    ):
        face_count = face_result["face_count"]

        if face_count >= 2:

            if self.multiple_faces_start is None:
                self.multiple_faces_start = now

            duration = now - self.multiple_faces_start

            if (
                duration >= self.MULTIPLE_FACES_CONFIRMATION
                and not self.multiple_faces_logged
            ):

                self._record(
                    event_type="MULTIPLE_FACES",
                    details={
                        "face_count": face_count,
                        "duration_seconds": round(
                            duration,
                            2
                        )
                    }
                )

                self.multiple_faces_logged = True

        else:

            self.multiple_faces_start = None
            self.multiple_faces_logged = False

    def _process_phone(
        self,
        phone_result,
        now
    ):
        if phone_result["detected"]:

            if self.phone_start is None:
                self.phone_start = now

            duration = now - self.phone_start

            if (
                duration >= self.PHONE_CONFIRMATION
                and not self.phone_logged
            ):

                self._record(
                    event_type="SMARTPHONE_DETECTED",
                    confidence=round(
                        phone_result["confidence"],
                        3
                    ),
                    details={
                        "duration_seconds": round(
                            duration,
                            2
                        ),
                        "source": "camera"
                    }
                )

                self.phone_logged = True

        else:

            self.phone_start = None
            self.phone_logged = False

    def _process_gaze(
        self,
        face_result,
        gaze_result,
        now
    ):
        """
        Фиксирует длительное отклонение взгляда.

        При отсутствии ровно одного лица
        анализ взгляда не считается нарушением.
        """

        if face_result["face_count"] != 1:

            self.gaze_start = None
            self.gaze_logged = False
            return

        direction = gaze_result["direction"]

        is_deviated = direction in {
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN"
        }

        if is_deviated:

            if self.gaze_start is None:
                self.gaze_start = now

            duration = now - self.gaze_start

            if (
                duration >= self.GAZE_CONFIRMATION
                and not self.gaze_logged
            ):

                self._record(
                    event_type="GAZE_DEVIATION",
                    details={
                        "direction": direction,
                        "duration_seconds": round(
                            duration,
                            2
                        )
                    }
                )

                self.gaze_logged = True

        else:

            self.gaze_start = None
            self.gaze_logged = False

    def _process_head(
        self,
        face_result,
        head_result,
        now
    ):
        """
        Фиксирует длительное отклонение положения головы.
        """

        if face_result["face_count"] != 1:

            self.head_start = None
            self.head_logged = False
            return

        direction = head_result["direction"]

        is_deviated = direction in {
            "LEFT",
            "RIGHT",
            "UP",
            "DOWN"
        }

        if is_deviated:

            if self.head_start is None:
                self.head_start = now

            duration = now - self.head_start

            if (
                duration >= self.HEAD_CONFIRMATION
                and not self.head_logged
            ):

                self._record(
                    event_type="HEAD_POSE_DEVIATION",
                    details={
                        "direction": direction,
                        "yaw": head_result["yaw"],
                        "pitch": head_result["pitch"],
                        "roll": head_result["roll"],
                        "duration_seconds": round(
                            duration,
                            2
                        )
                    }
                )

                self.head_logged = True

        else:

            self.head_start = None
            self.head_logged = False

    def get_total_violations(self):
        return self.total_violations
