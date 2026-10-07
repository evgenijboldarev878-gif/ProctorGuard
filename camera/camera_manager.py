import cv2


class CameraManager:
    """
    Управляет веб-камерой ProctorGuard.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera = None

    def start(self):
        self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Не удалось открыть камеру.")

    def read(self):
        if self.camera is None:
            raise RuntimeError("Камера не запущена.")

        success, frame = self.camera.read()

        if not success:
            raise RuntimeError("Не удалось получить кадр с камеры.")

        return frame

    def release(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
