from detection.image_detection import detect_objects_in_photo
from detection.video_detection import detect_objects_in_video
from detection.camera_detection import detect_objects_from_camera

if __name__ == "__main__":
    # Imagem
    # detect_objects_in_photo("media/facaCasaVerde.jpg")

    # Vídeo
    # detect_objects_in_video("media/videoFacaVermelha.mp4")

    # Câmera Ligada
    detect_objects_from_camera()
