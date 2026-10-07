import cv2
import mediapipe as mp


class FaceDetector:
    """
    Детектор лиц на основе MediaPipe Face Landmarker.

    Определяет:
    - количество лиц в кадре;
    - наличие лица;
    - координаты landmarks каждого лица.
    """

    def __init__(
        self,
        model_path="models/face_landmarker.task",
        num_faces=5,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.model_path = model_path

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )

    def detect(self, frame, timestamp_ms):
        """
        Анализирует один кадр камеры.

        Возвращает:
        {
            "face_detected": True/False,
            "face_count": количество лиц,
            "landmarks": landmarks лиц
        }
        """

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        face_count = len(result.face_landmarks)

        return {
            "face_detected": face_count > 0,
            "face_count": face_count,
            "landmarks": result.face_landmarks
        }

    def close(self):
        """Освобождает ресурсы MediaPipe."""
        self.landmarker.close()
