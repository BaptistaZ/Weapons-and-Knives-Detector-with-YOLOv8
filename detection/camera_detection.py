import cv2
import time
import os
from ultralytics import YOLO
from playsound import playsound

def detect_objects_from_camera(enable_sound=True):
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    cap = cv2.VideoCapture(0)

    os.makedirs("./frames_detectados", exist_ok=True)

    screenshot_count = 1
    last_trigger_time = 0
    cooldown = 0.5

    perigo_ativo = False
    perigo_deteccoes_consecutivas = 0
    seguro_deteccoes_consecutivas = 0
    LIMIAR_PERIGO = 3
    LIMIAR_SEGURO = 5

    if not cap.isOpened():
        print("Error opening camera.")
        return

    print("Camera on. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera.")
            break

        results = yolo_model(frame, verbose=False)
        deteccao_perigosa = False

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    label = classes[int(cls[pos])]
                    if label.lower() in ["knife", "gun"]:
                        deteccao_perigosa = True
                        now = time.time()
                        if now - last_trigger_time > cooldown:
                            if enable_sound:
                                playsound("beep.wav", block=False)
                            cv2.imwrite(f"./frames_detectados/camera_{screenshot_count:03d}.jpg", frame)
                            screenshot_count += 1
                            last_trigger_time = now

                    xmin, ymin, xmax, ymax = detection
                    label_text = f"{label} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]) * 40 % 255, 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label_text, (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if deteccao_perigosa:
            perigo_deteccoes_consecutivas += 1
            seguro_deteccoes_consecutivas = 0
        else:
            perigo_deteccoes_consecutivas = 0
            seguro_deteccoes_consecutivas += 1

        if perigo_deteccoes_consecutivas >= LIMIAR_PERIGO:
            perigo_ativo = True
        elif seguro_deteccoes_consecutivas >= LIMIAR_SEGURO:
            perigo_ativo = False

        estado_texto = "DANGER DETECTED" if perigo_ativo else "SAFE"
        estado_cor = (0, 0, 255) if perigo_ativo else (0, 255, 0)
        cv2.putText(frame, estado_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, estado_cor, 2)

        cv2.imshow("Real-time detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera off.")
