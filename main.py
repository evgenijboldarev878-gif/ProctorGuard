from proctor_engine import ProctorEngine
from interface.dashboard import ProctorDashboard


print("====================================")
print("       PROCTORGARD ЗАПУЩЕН")
print("====================================")
print("Нажми Q для выхода.")


engine = ProctorEngine()
dashboard = ProctorDashboard()

try:
    engine.start()

    while True:

        data = engine.process_frame()

        frame = data["frame"]
        face_result = data["face"]
        phone_result = data["phone"]
        gaze_result = data["gaze"]
        head_result = data["head"]
        violations = data["violations"]
        keyboard_events = data["keyboard_events"]

        dashboard.show(
            frame=frame,
            face_result=face_result,
            phone_result=phone_result,
            gaze_result=gaze_result,
            head_result=head_result,
            total_violations=violations,
            keyboard_events=keyboard_events
        )

        if dashboard.wait_key() == "q":
            break

finally:

    engine.stop()
    dashboard.close()

    print("====================================")
    print("       PROCTORGARD ОСТАНОВЛЕН")
    print("====================================")
