from pynput import keyboard


class KeyboardGuard:
    """
    Отслеживает только контролируемые комбинации клавиш.

    Обычный ввод текста не записывается.
    """

    MONITORED_KEYS = {
        "alt_tab": "ALT+TAB",
        "alt_f4": "ALT+F4",
        "ctrl_c": "CTRL+C",
        "ctrl_v": "CTRL+V",
        "ctrl_x": "CTRL+X",
        "ctrl_a": "CTRL+A",
        "print_screen": "PRINT_SCREEN",
        "windows": "WINDOWS"
    }

    CTRL_COMBINATIONS = {
        "\x03": "ctrl_c",
        "\x16": "ctrl_v",
        "\x18": "ctrl_x",
        "\x01": "ctrl_a"
    }

    def __init__(self, logger, event_callback=None):
        self.logger = logger
        self.event_callback = event_callback

        self.alt_pressed = False
        self.ctrl_pressed = False

        self.listener = None

    def _key_name(self, key):
        if key in {
            keyboard.Key.alt,
            keyboard.Key.alt_l,
            keyboard.Key.alt_r
        }:
            return "alt"

        if key in {
            keyboard.Key.ctrl,
            keyboard.Key.ctrl_l,
            keyboard.Key.ctrl_r
        }:
            return "ctrl"

        if key == keyboard.Key.print_screen:
            return "print_screen"

        if key in {
            keyboard.Key.cmd,
            keyboard.Key.cmd_l,
            keyboard.Key.cmd_r
        }:
            return "windows"

        if key == keyboard.Key.tab:
            return "tab"

        if key == keyboard.Key.f4:
            return "f4"

        try:
            return key.char.lower()
        except (AttributeError, TypeError):
            return None

    def _log(self, event_name):
        self.logger.log_event(
            event_type="KEYBOARD_EVENT",
            details={"key": event_name}
        )

        if self.event_callback is not None:
            self.event_callback(event_name)

    def on_press(self, key):
        name = self._key_name(key)

        if name == "alt":
            self.alt_pressed = True
            return

        if name == "ctrl":
            self.ctrl_pressed = True
            return

        if name == "windows":
            self._log(self.MONITORED_KEYS["windows"])
            return

        if name == "print_screen":
            self._log(self.MONITORED_KEYS["print_screen"])
            return

        if self.alt_pressed:
            if name == "tab":
                self._log(self.MONITORED_KEYS["alt_tab"])
                return

            if name == "f4":
                self._log(self.MONITORED_KEYS["alt_f4"])
                return

        if self.ctrl_pressed:
            try:
                char = key.char
            except (AttributeError, TypeError):
                char = None

            if char in self.CTRL_COMBINATIONS:
                combination_name = self.CTRL_COMBINATIONS[char]
                self._log(self.MONITORED_KEYS[combination_name])
                return

            if name in {
                "c",
                "v",
                "x",
                "a"
            }:
                combination_name = {
                    "c": "ctrl_c",
                    "v": "ctrl_v",
                    "x": "ctrl_x",
                    "a": "ctrl_a"
                }[name]

                self._log(self.MONITORED_KEYS[combination_name])

    def on_release(self, key):
        name = self._key_name(key)

        if name == "alt":
            self.alt_pressed = False

        elif name == "ctrl":
            self.ctrl_pressed = False

    def start(self):
        if self.listener is not None:
            return

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )

        self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
