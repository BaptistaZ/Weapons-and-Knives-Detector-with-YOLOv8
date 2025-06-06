import cv2
import time
import os
from ultralytics import YOLO
from playsound import playsound

def detect_objects_from_camera():

    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    cap = cv2.VideoCapture(0)

    screenshot_count = 1
    last_trigger_time = 0
    cooldown = 1

    if not cap.isOpened():
        print("Error opening camera.")
        return

    print("Camera on. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
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
                            filename = f"./frames_detectados/camera_{screenshot_count:03d}.jpg"
                            cv2.imwrite(filename, frame)
                            screenshot_count += 1
                            last_trigger_time = now

                    xmin, ymin, xmax, ymax = detection
                    label_text = f"{label} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]) * 40 % 255, 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label_text, (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Real-time detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera off.")
