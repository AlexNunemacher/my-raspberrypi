import time
import requests
import numpy as np


IP = "192.168.1.2"
STREAM_URL = f"http://{IP}:5800"

MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS_PATH = "coco_labels.txt"

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

TARGET_OBJECTS = ["person"]
CONFIDENCE_THRESHOLD = 0.50
ALERT_COOLDOWN = 15

def send_telegram_alert(image, message):
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message}

    try:
        requests.post(url, data=data, files=files, timeout=5)
    except Exception:
        pass

def main():
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    labels = read_label_file(LABELS_PATH)

    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        return

    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        _, scale = common.set_resized_input(
            interpreter, 
            (frame.shape[1], frame.shape[0]),
            lambda size: cv2.resize(frame, size)
        )

        interpreter.invoke()
        objs = detect.get_objects(interpreter, CONFIDENCE_THRESHOLD, scale)

        detected_targets = []

        for obj in objs:
            label_name = labels.get(obj.id, "Unknown")
            bbox = obj.bbox

            cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {obj.score:.2f}",
                        (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label_name in TARGET_OBJECTS:
                detected_targets.append(label_name)

        current_time = time.time()
        if detected_targets and (current_time - last_alert_time > ALERT_COOLDOWN):
            summary = ", ".join(set(detected_targets))
            alert_msg = f"Security Alert! Detected: {summary}"
            send_telegram_alert(frame, alert_msg)
            last_alert_time = current_time

        cv2.imshow("Security Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()