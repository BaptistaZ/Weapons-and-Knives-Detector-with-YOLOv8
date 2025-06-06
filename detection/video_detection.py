import os
import time
import cv2
from playsound import playsound
from ultralytics import YOLO

def detect_objects_in_video(video_path, enable_sound=True):
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
    detection_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        results = yolo_model(frame)
        danger_detected = False

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    label = classes[int(cls[pos])]
                    if label.lower() in ["knife", "gun"]:
                        danger_detected = True
                        now = time.time()
                        if now - last_trigger_time > cooldown:
                            if enable_sound:
                                playsound("beep.wav", block=False)
                            cv2.imwrite(f"./frames_detectados/frame_{screenshot_count:03d}.jpg", frame)
                            screenshot_count += 1
                            detection_count += 1
                            last_trigger_time = now

                    xmin, ymin, xmax, ymax = detection
                    label_text = f"{label} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]) * 40 % 255, 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label_text, (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Informação extra no ecrã
        estado = "DANGER DETECTED" if danger_detected else "SAFE"
        estado_color = (0, 0, 255) if danger_detected else (0, 255, 0)

        cv2.putText(frame, f"State: {estado}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, estado_color, 2)

        cv2.putText(frame, f"Detections: {detection_count}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, f"Time: {frame_time:.2f}s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, f"File: {os.path.basename(video_path)}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.putText(frame, "Press 'q' to exit", (width - 230, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        out.write(frame)

        cv2.imshow("Real-time detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
