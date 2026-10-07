import sys
import time
from pathlib import Path

from camera.camera_manager import CameraManager
from detection.face_detector import FaceDetector
from detection.phone_detector import PhoneDetector
from detection.gaze_detector import GazeDetector
from detection.head_pose_detector import HeadPoseDetector
from monitoring.event_logger import EventLogger
from monitoring.violation_manager import ViolationManager
from security.keyboard_guard import KeyboardGuard
from interface.keyboard_window import KeyboardEventsWindow


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent

    return str(base_path / relative_path)


class ProctorEngine:
    def __init__(self):
        self.camera = CameraManager()

        self.face_detector = FaceDetector(
            model_path=resource_path("models/face_landmarker.task"),
            num_faces=5
        )

        self.phone_detector = PhoneDetector(
            model_path=resource_path("yolo11n.pt"),
            confidence=0.25
        )

        self.gaze_detector = GazeDetector()
        self.head_pose_detector = HeadPoseDetector()

        self.logger = EventLogger(log_directory="logs")

        self.violation_manager = ViolationManager(
            logger=self.logger
        )

        self.keyboard_window = KeyboardEventsWindow()

        self.keyboard_guard = KeyboardGuard(
            logger=self.logger,
            event_callback=self.keyboard_window.add_event
        )

        self.start_time = None

    def start(self):
        self.camera.start()
        self.keyboard_guard.start()
        self.start_time = time.monotonic()

    def process_frame(self):
        frame = self.camera.read()

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        face_result = self.face_detector.detect(
            frame,
            timestamp_ms
        )

        phone_result = self.phone_detector.detect(frame)

        gaze_result = self.gaze_detector.detect_from_face_result(
            face_result
        )

        head_result = {
            "direction": "UNKNOWN",
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }

        if face_result["face_count"] == 1:
            height, width = frame.shape[:2]

            head_result = self.head_pose_detector.detect(
                face_result["landmarks"][0],
                width,
                height
            )

        self.violation_manager.process(
            face_result=face_result,
            phone_result=phone_result,
            gaze_result=gaze_result,
            head_result=head_result
        )

        return {
            "frame": frame,
            "face": face_result,
            "phone": phone_result,
            "gaze": gaze_result,
            "head": head_result,
            "violations": self.violation_manager.get_total_violations(),
            "keyboard_events": self.keyboard_window.get_events()
        }

    def stop(self):
        self.keyboard_guard.stop()
        self.camera.release()
        self.face_detector.close()
        self.keyboard_window.close()
