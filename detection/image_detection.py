import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def detect_objects_in_photo(image_path):
    image_orig = cv2.imread(image_path)
    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')

    results = yolo_model(image_orig)

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                xmin, ymin, xmax, ymax = detection
                color = (0, int(cls[pos]) * 40 % 255, 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Detection Result")
    plt.axis("off")
    plt.show()

    return image_orig

