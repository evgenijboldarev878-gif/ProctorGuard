import cv2

from phone_detector import PhoneDetector


print("Загрузка детектора телефона...")

detector = PhoneDetector(
    model_path="../yolo11n.pt",
    confidence=0.50
)

print("Детектор телефона загружен.")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ОШИБКА: камера не открылась.")
    raise SystemExit

print("Камера запущена.")
print("Нажми Q для выхода.")

while True:
    success, frame = camera.read()

    if not success:
        print("ОШИБКА: не удалось получить кадр.")
        break

    detection = detector.detect(frame)

    if detection["detected"]:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        text = f"SMARTPHONE {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    cv2.imshow("ProctorGuard - Phone Detector Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Тест завершён.")