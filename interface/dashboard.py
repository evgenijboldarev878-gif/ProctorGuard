import cv2
import numpy as np


class ProctorDashboard:

    WINDOW_NAME = "ProctorGuard"

    PANEL_WIDTH = 320
    KEYBOARD_HEIGHT = 250

    def __init__(self):
        self.total_violations = 0

    def show(
        self,
        frame,
        face_result,
        phone_result,
        gaze_result,
        head_result,
        total_violations,
        keyboard_events
    ):
        self.total_violations = total_violations

        frame_height, frame_width = frame.shape[:2]

        # ==========================================
        # ПРАВАЯ ПАНЕЛЬ СО СТАТУСОМ
        # ==========================================

        panel = np.zeros(
            (frame_height, self.PANEL_WIDTH, 3),
            dtype=np.uint8
        )

        cv2.putText(
            panel,
            "PROCTORGARD",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.line(
            panel,
            (20, 60),
            (self.PANEL_WIDTH - 20, 60),
            (120, 120, 120),
            1
        )

        face_count = face_result["face_count"]

        if face_count == 0:
            face_text = "FACE: NOT DETECTED"
            face_color = (0, 0, 255)
        elif face_count == 1:
            face_text = "FACE: 1"
            face_color = (0, 255, 0)
        else:
            face_text = f"FACES: {face_count}"
            face_color = (0, 165, 255)

        self._text(
            panel,
            face_text,
            100,
            face_color
        )

        if phone_result["detected"]:
            confidence = phone_result["confidence"]

            x1, y1, x2, y2 = phone_result["box"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"SMARTPHONE {confidence:.2f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            phone_text = "SMARTPHONE: DETECTED"
            phone_color = (0, 255, 255)

        else:
            phone_text = "SMARTPHONE: NOT DETECTED"
            phone_color = (0, 255, 0)

        self._text(
            panel,
            phone_text,
            145,
            phone_color,
            scale=0.55
        )

        gaze_direction = gaze_result["direction"]

        self._text(
            panel,
            f"GAZE: {gaze_direction}",
            190,
            (255, 255, 255)
        )

        head_direction = head_result["direction"]

        if head_direction == "CENTER":
            head_color = (0, 255, 0)
        elif head_direction == "UNKNOWN":
            head_color = (160, 160, 160)
        else:
            head_color = (0, 165, 255)

        self._text(
            panel,
            f"HEAD: {head_direction}",
            235,
            head_color
        )

        yaw = head_result["yaw"]
        pitch = head_result["pitch"]
        roll = head_result["roll"]

        self._text(
            panel,
            f"YAW:   {yaw:.1f}",
            290,
            (220, 220, 220),
            scale=0.6
        )

        self._text(
            panel,
            f"PITCH: {pitch:.1f}",
            325,
            (220, 220, 220),
            scale=0.6
        )

        self._text(
            panel,
            f"ROLL:  {roll:.1f}",
            360,
            (220, 220, 220),
            scale=0.6
        )

        cv2.line(
            panel,
            (20, 390),
            (self.PANEL_WIDTH - 20, 390),
            (120, 120, 120),
            1
        )

        violation_color = (
            (0, 255, 0)
            if self.total_violations == 0
            else (0, 165, 255)
        )

        self._text(
            panel,
            f"VIOLATIONS: {self.total_violations}",
            435,
            violation_color
        )

        # ==========================================
        # РАМКИ ЛИЦ
        # ==========================================

        for face_landmarks in face_result["landmarks"]:
            xs = [
                int(landmark.x * frame_width)
                for landmark in face_landmarks
            ]

            ys = [
                int(landmark.y * frame_height)
                for landmark in face_landmarks
            ]

            if xs and ys:
                x1 = max(min(xs), 0)
                y1 = max(min(ys), 0)
                x2 = min(max(xs), frame_width - 1)
                y2 = min(max(ys), frame_height - 1)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        # ==========================================
        # ОБЪЕДИНЯЕМ КАМЕРУ + ПРАВУЮ ПАНЕЛЬ
        # ==========================================

        top_section = np.hstack(
            (frame, panel)
        )

        total_width = top_section.shape[1]

        # ==========================================
        # НИЖНЯЯ ПАНЕЛЬ KEYBOARD EVENTS
        # ==========================================

        keyboard_panel = np.zeros(
            (
                self.KEYBOARD_HEIGHT,
                total_width,
                3
            ),
            dtype=np.uint8
        )

        cv2.putText(
            keyboard_panel,
            "KEYBOARD EVENTS",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.line(
            keyboard_panel,
            (20, 55),
            (total_width - 20, 55),
            (120, 120, 120),
            1
        )

        events = keyboard_events[-8:]

        y = 90

        if not events:
            cv2.putText(
                keyboard_panel,
                "No monitored keyboard events",
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (150, 150, 150),
                2
            )

        else:
            for event in reversed(events):
                timestamp = event["time"]
                event_name = event["event"]

                cv2.putText(
                    keyboard_panel,
                    f"{timestamp}    {event_name}",
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                y += 28

        # ==========================================
        # ФИНАЛЬНОЕ ОКНО
        # ==========================================

        combined = np.vstack(
            (top_section, keyboard_panel)
        )

        cv2.imshow(
            self.WINDOW_NAME,
            combined
        )

    @staticmethod
    def _text(
        image,
        text,
        y,
        color,
        scale=0.65
    ):
        cv2.putText(
            image,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2
        )

    def wait_key(self):
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return "q"

        return None

    def close(self):
        cv2.destroyAllWindows()
