import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Índices dos keypoints relevantes
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def calculate_angle(point1, point2, point3):
    if point1 is None or point2 is None or point3 is None:
        return 0
    radians = np.arctan2(point3.y - point2.y, point3.x - point2.x) - \
              np.arctan2(point1.y - point2.y, point1.x - point2.x)
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def detect_activities(frame):
    """
    Detecta atividades para todas as pessoas na cena.
    Retorna:
       - activities_list: lista (um item por pessoa) com as atividades detectadas (ex.: ["Braco Levantado", "Pessoa Sentada"])
       - anomaly_detected: True se pelo menos uma pessoa tiver Braco levantado
       - results: objeto do MediaPipe para desenho
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True

    activities_list = []
    anomaly_detected = False

    if results.pose_landmarks:
        # Suporte para múltiplas detecções (se results.pose_landmarks for uma lista)
        if isinstance(results.pose_landmarks, list):
            all_landmarks = results.pose_landmarks
        else:
            all_landmarks = [results.pose_landmarks]

        for landmarks in all_landmarks:
            person_activities = []
            # Detecção de Braco levantado: verifica se qualquer pulso está acima do respectivo ombro
            try:
                left_shoulder = landmarks.landmark[LEFT_SHOULDER]
                right_shoulder = landmarks.landmark[RIGHT_SHOULDER]
                left_wrist = landmarks.landmark[LEFT_WRIST]
                right_wrist = landmarks.landmark[RIGHT_WRIST]
                if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
                    person_activities.append("Braco Levantado")
                    anomaly_detected = True
            except Exception as e:
                print(f"Erro ao detectar Braco levantado: {e}")

            # Detecção de sentado ou em Pe
            try:
                left_hip = landmarks.landmark[LEFT_HIP]
                right_hip = landmarks.landmark[RIGHT_HIP]
                left_knee = landmarks.landmark[LEFT_KNEE]
                right_knee = landmarks.landmark[RIGHT_KNEE]
                left_ankle = landmarks.landmark[LEFT_ANKLE]
                right_ankle = landmarks.landmark[RIGHT_ANKLE]

                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                sitting_threshold = 130
                standing_threshold = 160

                if left_knee_angle < sitting_threshold and right_knee_angle < sitting_threshold:
                    person_activities.append("Pessoa Sentada")
                elif left_knee_angle > standing_threshold and right_knee_angle > standing_threshold:
                    person_activities.append("Pessoa Em Pe")
            except Exception as e:
                print(f"Erro na detecção de sentado/em Pe: {e}")

            activities_list.append(person_activities)
    else:
        activities_list = []

    return activities_list, anomaly_detected, results


def draw_landmarks(frame, results):
    """Desenha os landmarks e conexões do esqueleto no frame."""
    if results and results.pose_landmarks:
        if isinstance(results.pose_landmarks, list):
            for landmarks in results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
        else:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )


if __name__ == "__main__":
    print("Este é o módulo activity_detection_module. Execute 'main.py' para iniciar a aplicação.")
