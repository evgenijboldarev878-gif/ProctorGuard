from event_logger import EventLogger


print("Создание журнала...")

logger = EventLogger()

event = logger.log_event(
    event_type="SMARTPHONE_DETECTED",
    confidence=0.70,
    details={
        "test": True,
        "message": "Тестовая запись журнала"
    }
)

print("Событие успешно записано.")
print(event)
print("Файл журнала:", logger.log_file)