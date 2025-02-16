import cv2
from face_recognition_module import recognize_faces
from emotion_analysis_module import analyze_emotions
from activity_detection_module import detect_activities, draw_landmarks


class VideoProcessor:
    def __init__(self, video_path, output_path, known_face_encodings, known_face_names, frame_skip=2, resize_factor=1.0):
        self.video_path = video_path
        self.output_path = output_path
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.frame_skip = frame_skip
        self.resize_factor = resize_factor

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Erro ao abrir o vídeo: {self.video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.resize_factor)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.resize_factor)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))

        total_frames = 0
        anomaly_count = 0
        face_data = []      # Lista de detecções faciais: {"face_id", "name", "location"}
        emotion_data = []   # Lista: {"face_id", "label"}
        activity_data = []  # Lista: {"face_id", "activities": [lista de atividades]}
        face_id_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            if self.resize_factor != 1.0:
                frame = cv2.resize(frame, (frame_width, frame_height))

            # Processa somente 1 a cada frame_skip
            if total_frames % self.frame_skip != 0:
                out.write(frame)
                cv2.imshow('Processed Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Reconhecimento facial
            faces = recognize_faces(frame, self.known_face_encodings, self.known_face_names)
            face_locations = [face["location"] for face in faces]
            # Análise de emoções para os rostos detectados
            emotions = analyze_emotions(frame, face_locations)

            # Atribui um face_id a cada face e armazena os dados
            for i, face in enumerate(faces):
                face["face_id"] = face_id_counter
                face_data.append(face)
                emotion_label = emotions[i] if i < len(emotions) else "Desconhecido"
                emotion_data.append({"face_id": face_id_counter, "label": emotion_label})
                face_id_counter += 1

            # Detecção de atividades (usando MediaPipe Pose)
            activities_list, anomaly_detected, pose_results = detect_activities(frame)
            if anomaly_detected:
                anomaly_count += 1

            # Associação: se o número de rostos igualar o de poses detectadas, vincula cada face à sua atividade;
            # caso contrário, atribui "Indefinido"
            if len(faces) == len(activities_list):
                for i, face in enumerate(faces):
                    activity_data.append({"face_id": face["face_id"], "activities": activities_list[i]})
            else:
                for face in faces:
                    activity_data.append({"face_id": face["face_id"], "activities": ["Indefinido"]})

            # Desenha retângulos, nomes e emoções para cada rosto
            for i, face in enumerate(faces):
                top, right, bottom, left = face["location"]
                name = face["name"]
                emotion_text = emotions[i] if i < len(emotions) else "Desconhecido"

                padding = 10
                top = max(0, top - padding)
                right = min(frame.shape[1], right + padding)
                bottom = min(frame.shape[0], bottom + padding)
                left = max(0, left - padding)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, emotion_text, (left + 6, top - 6), font, 0.8, (0, 0, 255), 1)

            # Exibe as atividades no frame para os rostos processados (apenas as deste ciclo)
            y_offset = 50
            for ad in activity_data[-len(faces):]:
                activities_str = ", ".join(ad["activities"])
                cv2.putText(frame, activities_str, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                y_offset += 25

            # Desenha os landmarks dos esqueleto (pose) na imagem
            draw_landmarks(frame, pose_results)

            out.write(frame)
            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Vídeo processado e salvo em:", self.output_path)
        return total_frames, anomaly_count, face_data, emotion_data, activity_data


if __name__ == "__main__":
    print("Este é o módulo video_processing. Execute 'main.py' para iniciar a aplicação.")
