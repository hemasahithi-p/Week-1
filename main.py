# main.py
from ultralytics import YOLO
import cv2
import numpy as np
import math
import imutils
from playsound import playsound
import threading
import time
import os

# Config
VIDEO_SOURCE = 0                 # 0 = webcam, or 'videos/road.mp4'
OUTPUT_PATH = 'outputs/result.avi'
CONF_THRESHOLD = 0.35
TARGET_CLASSES = ['person', 'car', 'motorbike', 'bus', 'truck']  # focus
ALERT_COOLDOWN = 3.0             # seconds between audio alerts

os.makedirs('outputs', exist_ok=True)

# Load YOLO model (this will auto-download if needed)
model = YOLO('yolov8n.pt')

last_alert_time = 0.0

def play_alert():
    # run sound in separate thread to avoid blocking
    def _play():
        try:
            playsound('alert.wav')
        except Exception as e:
            print("Audio play error:", e)
    threading.Thread(target=_play, daemon=True).start()

def estimate_proximity(box_w, frame_w):
    """
    Simple proximity heuristic: larger bounding box width => closer object.
    We normalize box_w by frame width.
    Returns a score 0..1 where 1 is very close.
    """
    rel = box_w / float(frame_w)
    # sharpen behavior (tunable)
    score = min(1.0, (rel - 0.08) / (0.4 - 0.08))  # map rel from [0.08..0.4] -> [0..1]
    score = max(0.0, score)
    return score

def main():
    global last_alert_time
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Cannot open video source:", VIDEO_SOURCE)
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: resize for speed
        frame = imutils.resize(frame, width=960)
        f_h, f_w = frame.shape[:2]

        # Run YOLO (stream=True to iterate results)
        results = model(frame, stream=True, conf=CONF_THRESHOLD, verbose=False)

        # default overlay
        warning_displayed = False

        for r in results:
            # boxes container
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                label = model.names[cls_idx]

                if label not in TARGET_CLASSES:
                    continue

                # draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 200, 20), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 200, 20), 2)

                # simple proximity check using bounding box width
                box_w = (x2 - x1)
                proximity = estimate_proximity(box_w, f_w)
                # threshold to trigger alert (tune as needed)
                if proximity > 0.75:
                    # display warning
                    cv2.putText(frame, "⚠ COLLISION WARNING ⚠", (int(f_w*0.2), 50),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                    warning_displayed = True

        # If warning is on, optionally play audio (with cooldown to avoid rapid repeats)
        if warning_displayed:
            now = time.time()
            if now - last_alert_time > ALERT_COOLDOWN:
                # ensure 'alert.wav' exists in project root (or skip if not)
                if os.path.exists('alert.wav'):
                    play_alert()
                last_alert_time = now

        # save & show frame
        out.write(frame)
        cv2.imshow("ADAS-Lite", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # press 's' to save current frame image
            cv2.imwrite(f'outputs/snapshot_{int(time.time())}.jpg', frame)
            print("Snapshot saved.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
