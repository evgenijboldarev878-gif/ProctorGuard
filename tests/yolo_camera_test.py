import cv2
from ultralytics import YOLO

print("Загрузка модели YOLO...")

model = YOLO("yolo11n.pt")

print("Модель загружена.")
print("Запуск камеры...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ОШИБКА: камера не открылась.")
    exit()

print("Камера запущена.")
print("Для выхода нажми Q.")

while True:
    success, frame = camera.read()

    if not success:
        print("ОШИБКА: не удалось получить кадр.")
        break

    # YOLO анализирует текущий кадр
    results = model(frame, verbose=False)

    # Рисуем найденные объекты
    annotated_frame = results[0].plot()

    # Показываем результат
    cv2.imshow("ProctorGuard - YOLO Test", annotated_frame)

    # Выход по Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Камера остановлена.")