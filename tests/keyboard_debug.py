from pynput import keyboard

print("Нажимай клавиши. Для выхода нажми ESC.")

def on_press(key):
    print("PRESS:", repr(key))

def on_release(key):
    print("RELEASE:", repr(key))

    if key == keyboard.Key.esc:
        return False

with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    listener.join()
