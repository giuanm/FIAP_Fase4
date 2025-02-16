# src/face_recognition_module.py
import face_recognition
import os


def load_known_faces(images_folder):
    """
    Carrega imagens de rostos conhecidos e gera os encodings.
    :param images_folder: Pasta contendo as imagens.
    :return: Tuple (known_face_encodings, known_face_names).
    """
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(images_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            image_path = os.path.join(images_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0][:-1]
                known_face_names.append(name)
            else:
                print(f"Aviso: Nenhum rosto detectado na imagem: {filename}")
    return known_face_encodings, known_face_names


def recognize_faces(frame, known_face_encodings, known_face_names):
    """
    Detecta e identifica rostos no frame.
    :param frame: Imagem (frame) do vídeo.
    :param known_face_encodings: Lista de encodings de rostos conhecidos.
    :param known_face_names: Lista de nomes correspondentes.
    :return: Lista de dicionários com chaves "name" e "location" (top, right, bottom, left).
    """
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    faces = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        faces.append({
            "name": name,
            "location": face_location  # (top, right, bottom, left)
        })
    return faces


if __name__ == "__main__":
    print("Este é o módulo face_recognition_module. Execute 'main.py' para iniciar a aplicação.")
