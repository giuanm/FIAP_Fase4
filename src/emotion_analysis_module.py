from deepface import DeepFace


def analyze_emotions(frame, face_locations):
    """
    Analisa as expressões emocionais dos rostos detectados no frame.
    :param frame: Imagem (frame) do vídeo.
    :param face_locations: Lista de tuplas (top, right, bottom, left) para cada rosto.
    :return: Lista de strings com a emoção dominante para cada rosto.
    """
    emotions = []
    for (top, right, bottom, left) in face_locations:
        top = max(0, top)
        right = min(frame.shape[1], right)
        bottom = min(frame.shape[0], bottom)
        left = max(0, left)

        face_frame = frame[top:bottom, left:right]
        if face_frame.size == 0:
            emotions.append("Desconhecido")
            continue

        try:
            analysis = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list) and len(analysis) > 0:
                emotion = analysis[0].get('dominant_emotion', 'Desconhecido')
            else:
                emotion = analysis.get('dominant_emotion', 'Desconhecido')
            emotions.append(emotion)
        except Exception as e:
            print(f"Erro ao analisar emoções: {e}")
            emotions.append("Desconhecido")
    return emotions


if __name__ == "__main__":
    print("Este é o módulo emotion_analysis_module. Execute 'main.py' para iniciar a aplicação.")
