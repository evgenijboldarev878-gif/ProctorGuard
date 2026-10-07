import time
import cv2

from detection.phone_detector import PhoneDetector
from monitoring.event_logger import EventLogger


PHONE_CONFIRMATION_TIME = 1.5
MIN_CONFIDENCE = 0.50


print("Запуск системы мониторинга телефона...")

phone_detector = PhoneDetector(
    model_path="yolo11n.pt",
    confidence=MIN_CONFIDENCE
)

event_logger = EventLogger()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ОШИБКА: камера не открылась.")
    raise SystemExit


print("Камера запущена.")
print("Покажи телефон перед камерой.")
print("Для выхода нажми Q.")


phone_start_time = None
violation_logged = False


while True:

    success, frame = camera.read()

    if not success:
        print("ОШИБКА: не удалось получить кадр.")
        break

    detection = phone_detector.detect(frame)

    if detection["detected"]:

        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["box"]

        # Рисуем рамку телефона
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
            0.7,
            (0, 255, 255),
            2
        )

        # Начинаем отсчёт времени присутствия телефона
        if phone_start_time is None:
            phone_start_time = time.monotonic()

        duration = time.monotonic() - phone_start_time

        cv2.putText(
            frame,
            f"Detected: {duration:.1f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Фиксируем нарушение только один раз
        if (
            duration >= PHONE_CONFIRMATION_TIME
            and not violation_logged
        ):
            event = event_logger.log_event(
                event_type="SMARTPHONE_DETECTED",
                confidence=round(confidence, 3),
                details={
                    "duration_seconds": round(duration, 2),
                    "source": "camera"
                }
            )

            print("!!! НАРУШЕНИЕ ЗАПИСАНО !!!")
            print(event)

            violation_logged = True

    else:

        # Телефон исчез из кадра
        phone_start_time = None
        violation_logged = False

        cv2.putText(
            frame,
            "SMARTPHONE: NOT DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "ProctorGuard - Phone Monitoring",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("Мониторинг завершён.")