import os
import time
import cv2
from playsound import playsound
from ultralytics import YOLO

def detect_objects_in_video(video_path):
    os.makedirs("./frames_detectados", exist_ok=True)

    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("detected_objects_video2.avi", fourcc, 20.0, (width, height))

    screenshot_count = 1
    last_trigger_time = 0
    cooldown = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    label = classes[int(cls[pos])]
                    if label.lower() in ["knife", "gun"]:
                        now = time.time()
                        if now - last_trigger_time > cooldown:
                            playsound("beep.wav", block=False)
                            last_trigger_time = now

                    xmin, ymin, xmax, ymax = detection
                    label_text = f"{label} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]) * 40 % 255, 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label_text, (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

        cv2.imshow("Real-time detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

