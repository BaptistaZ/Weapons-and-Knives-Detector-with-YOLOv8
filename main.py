from detection.image_detection import detect_objects_in_photo
from detection.video_detection import detect_objects_in_video
from detection.camera_detection import detect_objects_from_camera

if __name__ == "__main__":
    # Imagem

    # detect_objects_in_photo("media/facaCasaVerde.jpg")

    # Vídeo

    # Com som
     detect_objects_in_video("./media/videoFaca.mp4", enable_sound=True)
    # Sem som (modo stealth)
    # detect_objects_in_video("./media/videoFaca.mp4", enable_sound=False)

    # Câmera Ligada

    # Com som (modo normal)
    # detect_objects_from_camera(enable_sound=True)
    # Sem som (modo silencioso)
    # detect_objects_from_camera(enable_sound=False)

