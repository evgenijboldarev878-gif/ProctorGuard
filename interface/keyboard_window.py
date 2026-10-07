import time


class KeyboardEventsWindow:
    """
    Хранилище событий клавиатуры для отображения
    внутри основного окна ProctorGuard.
    """

    def __init__(self, max_events=8):
        self.max_events = max_events
        self.events = []

    def add_event(self, event_name):
        timestamp = time.strftime("%H:%M:%S")

        self.events.append({
            "time": timestamp,
            "event": event_name
        })

        if len(self.events) > self.max_events:
            self.events.pop(0)

    def get_events(self):
        return list(self.events)

    def close(self):
        pass
