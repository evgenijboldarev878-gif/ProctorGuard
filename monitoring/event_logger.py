from datetime import datetime
from pathlib import Path
import json


class EventLogger:
    """
    Журнал событий прокторинга.

    Каждое событие сохраняется в отдельную строку JSON.
    Это позволяет позже легко сформировать отчёт.
    """

    def __init__(self, log_directory="logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_directory / "proctor_events.jsonl"

    def log_event(
        self,
        event_type,
        confidence=None,
        details=None
    ):
        """
        Записывает одно событие.

        event_type:
            тип события, например SMARTPHONE_DETECTED

        confidence:
            уверенность модели, если она есть

        details:
            дополнительная информация
        """

        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": event_type,
            "confidence": confidence,
            "details": details or {}
        }

        with self.log_file.open(
            "a",
            encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False
                ) + "\n"
            )

        return event
