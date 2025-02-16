from video_processing import VideoProcessor
from face_recognition_module import load_known_faces
from report_module import generate_report


def main():
    video_path = "data/videos/video_fornecido.mp4"         # Vídeo de entrada
    output_path = "data/videos/output_processado.mp4"         # Vídeo de saída processado
    images_folder = "data/images"                             # Pasta com imagens de rostos conhecidos
    report_path = "reports/report.json"                       # Relatório final (JSON)

    # Carrega os rostos conhecidos
    known_face_encodings, known_face_names = load_known_faces(images_folder)

    # Cria a instância do processador de vídeo
    processor = VideoProcessor(
        video_path, output_path, known_face_encodings, known_face_names,
        frame_skip=2,         # Processa 1 a cada 2 frames para performance
        resize_factor=1.0
    )

    # Processa o vídeo e coleta os dados:
    # total_frames, anomaly_count, face_data, emotion_data, activity_data
    total_frames, anomaly_count, face_data, emotion_data, activity_data = processor.process_video()

    # Gera o relatório final com estatísticas agregadas por pessoa
    generate_report(face_data, emotion_data, activity_data, total_frames, anomaly_count, report_path)

    print("Processing completed.")
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()
