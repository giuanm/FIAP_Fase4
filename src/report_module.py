# src/report_module.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def generate_report(face_data, emotion_data, activity_data, total_frames, anomaly_count, report_path):
    """
    Gera o relatório final agregando estatísticas por pessoa.
    :param face_data: Lista de detecções faciais (cada item com "face_id", "name" e "location").
    :param emotion_data: Lista de dicionários {"face_id", "label"}.
    :param activity_data: Lista de dicionários {"face_id", "activities": [lista de atividades]}.
    :param total_frames: Total de frames processados.
    :param anomaly_count: Número total de anomalias detectadas.
    :param report_path: Caminho para salvar o relatório (JSON).
    """
    # Agrupando os dados por pessoa (usando o campo "name")
    person_stats = {}
    for face in face_data:
        name = face.get("name", "Desconhecido")
        if name not in person_stats:
            person_stats[name] = {
                "face_detections": 0,
                "emotions": {},
                "activities": {}
            }
        person_stats[name]["face_detections"] += 1

    # Agregando as emoções por pessoa, utilizando o face_id para relacionar
    for ed in emotion_data:
        face_id = ed["face_id"]
        face = next((f for f in face_data if f["face_id"] == face_id), None)
        if face:
            name = face.get("name", "Desconhecido")
            emotion = ed["label"]
            person_stats[name]["emotions"][emotion] = person_stats[name]["emotions"].get(emotion, 0) + 1

    # Agregando as atividades por pessoa
    for ad in activity_data:
        face_id = ad["face_id"]
        activities = ad["activities"] if isinstance(ad["activities"], list) else [ad["activities"]]
        face = next((f for f in face_data if f["face_id"] == face_id), None)
        if face:
            name = face.get("name", "Desconhecido")
            for act in activities:
                person_stats[name]["activities"][act] = person_stats[name]["activities"].get(act, 0) + 1

    report = {
        "total_frames": total_frames,
        "anomaly_count": anomaly_count,
        "total_face_detections": len(face_data),
        "total_person_count": len(person_stats),
        "identified_persons": list(person_stats.keys()),
        "person_statistics": person_stats
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"Relatório salvo em: {report_path}")

    plot_statistics(person_stats, os.path.dirname(report_path))


def plot_statistics(person_stats, output_dir):
    # Agregando estatísticas gerais de emoções e atividades
    overall_emotions = {}
    overall_activities = {}
    for stats in person_stats.values():
        for emo, count in stats["emotions"].items():
            overall_emotions[emo] = overall_emotions.get(emo, 0) + count
        for act, count in stats["activities"].items():
            overall_activities[act] = overall_activities.get(act, 0) + count

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(overall_emotions.keys()), y=list(overall_emotions.values()), palette="coolwarm")
    plt.title("Distribuição de Emoções")
    plt.xlabel("Emoção")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.barplot(x=list(overall_activities.keys()), y=list(overall_activities.values()), palette="viridis")
    plt.title("Distribuição de Atividades")
    plt.xlabel("Atividade")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45)

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "statistics.png")
    plt.savefig(graph_path)
    print(f"Gráficos salvos em: {graph_path}")
    plt.close()


if __name__ == "__main__":
    # # Dados simulados para teste
    # face_data = [
    #     {"face_id": 0, "name": "Valentina", "location": (10, 100, 50, 10)},
    #     {"face_id": 1, "name": "Nick", "location": (60, 150, 100, 60)}
    # ]
    # emotion_data = [
    #     {"face_id": 0, "label": "happy"},
    #     {"face_id": 1, "label": "neutral"},
    #     {"face_id": 0, "label": "happy"}
    # ]
    # activity_data = [
    #     {"face_id": 0, "activities": ["Pessoa Sentada", "Braco Levantado"]},
    #     {"face_id": 1, "activities": ["Pessoa Em Pe"]}
    # ]
    # generate_report(face_data, emotion_data, activity_data, total_frames=200, anomaly_count=1, report_path="reports/report.json")
    print("Este é o módulo report_module. Execute 'main.py' para iniciar a aplicação.")
