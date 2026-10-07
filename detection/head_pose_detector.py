import cv2
import numpy as np


class HeadPoseDetector:
    """
    Определяет положение головы по landmarks MediaPipe Face Landmarker.

    yaw   - поворот влево/вправо
    pitch - наклон вверх/вниз
    roll  - наклон головы вбок
    """

    NOSE_TIP = 1
    CHIN = 152

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    def __init__(
        self,
        yaw_threshold=15.0,
        pitch_threshold=15.0
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1)
            ],
            dtype=np.float64
        )

    def detect(
        self,
        face_landmarks,
        frame_width,
        frame_height
    ):

        required_indices = [
            self.NOSE_TIP,
            self.CHIN,
            self.LEFT_EYE_OUTER,
            self.RIGHT_EYE_OUTER,
            self.MOUTH_LEFT,
            self.MOUTH_RIGHT
        ]

        if (
            face_landmarks is None
            or len(face_landmarks) <= max(required_indices)
        ):
            return {
                "direction": "UNKNOWN",
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            }

        image_points = np.array(
            [
                self._point(
                    face_landmarks[self.NOSE_TIP],
                    frame_width,
                    frame_height
                ),
                self._point(
                    face_landmarks[self.CHIN],
                    frame_width,
                    frame_height
                ),
                self._point(
                    face_landmarks[self.LEFT_EYE_OUTER],
                    frame_width,
                    frame_height
                ),
                self._point(
                    face_landmarks[self.RIGHT_EYE_OUTER],
                    frame_width,
                    frame_height
                ),
                self._point(
                    face_landmarks[self.MOUTH_LEFT],
                    frame_width,
                    frame_height
                ),
                self._point(
                    face_landmarks[self.MOUTH_RIGHT],
                    frame_width,
                    frame_height
                )
            ],
            dtype=np.float64
        )

        focal_length = float(frame_width)

        camera_matrix = np.array(
            [
                [focal_length, 0, frame_width / 2],
                [0, focal_length, frame_height / 2],
                [0, 0, 1]
            ],
            dtype=np.float64
        )

        distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64
        )

        success, rotation_vector, _ = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "direction": "UNKNOWN",
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            }

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        angles = self._rotation_matrix_to_angles(
            rotation_matrix
        )

        yaw = angles["yaw"]
        pitch = angles["pitch"]
        roll = angles["roll"]

        direction = self._get_direction(
            yaw,
            pitch
        )

        return {
            "direction": direction,
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2)
        }

    @staticmethod
    def _point(landmark, width, height):
        return (
            landmark.x * width,
            landmark.y * height
        )

    @staticmethod
    def _rotation_matrix_to_angles(rotation_matrix):

        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2
            + rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:

            x_angle = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )

            y_angle = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            z_angle = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )

        else:

            x_angle = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )

            y_angle = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            z_angle = 0.0

        pitch = np.degrees(x_angle)
        yaw = np.degrees(y_angle)
        roll = np.degrees(z_angle)

        # Нормализация углов в диапазон -90...90.
        # Это устраняет ложные значения около 180°
        # при нормальном положении головы.

        if pitch > 90:
            pitch -= 180

        if pitch < -90:
            pitch += 180

        if yaw > 90:
            yaw -= 180

        if yaw < -90:
            yaw += 180

        if roll > 90:
            roll -= 180

        if roll < -90:
            roll += 180

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }

    def _get_direction(self, yaw, pitch):

        if yaw < -self.yaw_threshold:
            return "LEFT"

        if yaw > self.yaw_threshold:
            return "RIGHT"

        if pitch < -self.pitch_threshold:
            return "UP"

        if pitch > self.pitch_threshold:
            return "DOWN"

        return "CENTER"
