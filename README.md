## Tech Challenge - Video Analysis with Face Recognition, Emotion Detection, and Activity Detection

**[English Version Below]**

Este projeto é uma aplicação de análise de vídeo desenvolvida como parte do Tech Challenge da Fase 4 do curso "IA para Devs" da FIAP - Pós Tech. O sistema utiliza técnicas de visão computacional e aprendizado de máquina para:

*   **Detectar rostos** em um vídeo.
*   **Identificar pessoas** com base em um conjunto de imagens de referência.
*   **Analisar as expressões emocionais** (feliz, triste, neutro, com raiva, com medo, surpreso, com nojo) dos rostos detectados.
*   **Detectar atividades** simples, como "Braços Levantados" e "Pessoa Em Pé/Sentada", usando análise de pose (esqueleto).
*   **Gerar um relatório** com estatísticas sobre os frames processados, anomalias detectadas, atividades e emoções.

A aplicação foi desenvolvida em Python e utiliza as seguintes bibliotecas principais:

*   **OpenCV (cv2):** Para processamento de vídeo (leitura, exibição, escrita) e desenho de elementos gráficos (retângulos, texto).
*   **face_recognition:** Para detecção e identificação de rostos. (Baseada em dlib).
*   **dlib:** Usada internamente pela face_recognition para detecção de faces e *facial landmarks*.
*   **DeepFace:** Para análise de expressões emocionais.
*   **MediaPipe:** Para detecção de pose (esqueleto) e inferência de atividades.
*   **Poetry:** Para gerenciamento de dependências e ambiente virtual.
*   **NumPy:** Para operações numéricas (usada internamente por muitas bibliotecas).
*   **tqdm:** Para exibir barras de progresso (opcional, mas útil).
*    **TensorFlow/tf-keras:** Usada como backend pelo DeepFace.
*   **Gensim:** Para sumarização de texto
*   **Transformers:** Para sumarização de texto

## Tecnologia

Aqui estão as tecnologias usadas neste projeto:

*   Python 3.11
*   OpenCV
*   face_recognition (dlib)
*   DeepFace
*   MediaPipe
*   Poetry
*   NumPy
*   tqdm
* TensorFlow/tf-keras
* Gensim
* Transformers

## Serviços Utilizados

*   GitHub

## Começando

**Pré-requisitos:**

*   Python 3.11 (ou superior, mas 3.11 foi testado).
*   Poetry (para gerenciamento de dependências).
*   Visual Studio Build Tools com a carga de trabalho "Desktop development with C++" (necessário para compilar o `dlib` no Windows).  *Não* é necessário o Visual Studio *IDE* completo.
*   CMake (certifique-se de adicioná-lo ao PATH durante a instalação).
*   Git (opcional, mas recomendado para clonar o repositório).

**Instalação:**

1.  **Clone o repositório (opcional):**
    ```bash
    git clone <URL do seu repositório>
    cd <nome da pasta do projeto>
    ```

2.  **Instale o Poetry (se ainda não tiver):**  Siga as instruções em [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).

3.  **Instale as dependências com Poetry:**
    ```bash
    poetry install
    ```
    Este comando cria o ambiente virtual e instala todas as bibliotecas necessárias, com as versões corretas, definidas no arquivo `pyproject.toml`.

4.  **Prepare os Dados:**
    *   Crie uma pasta `data/videos` e coloque um vídeo de teste (`.mp4`) dentro dela.  Nomeie o vídeo como `video_fornecido.mp4` (ou altere o nome no arquivo `src/main.py`).
    *   Crie uma pasta `data/images` e coloque imagens de rostos conhecidos dentro dela. Cada imagem deve conter *apenas um* rosto, e o nome do arquivo deve ser o nome da pessoa (por exemplo, `joao.jpg`, `maria.png`).

5.  **Execute a Aplicação:**
    *   Ative o ambiente virtual do Poetry:
        ```bash
        poetry shell
        ```
    *   Execute o script principal:
        ```bash
        python src/main.py
        ```

    A aplicação irá processar o vídeo, exibir uma janela com os resultados (retângulos nos rostos, nomes, emoções e keypoints) e gerar um relatório (`report.txt`) na pasta `reports/`.  Pressione 'q' na janela de vídeo para encerrar a execução.

## How to Use

1.  **Prepare os Dados:** Coloque o vídeo de teste em `data/videos` e as imagens de rostos conhecidos em `data/images`.
2.  **Execute:** Execute `python src/main.py` dentro do ambiente virtual do Poetry.
3.  **Visualize:** Observe a janela de vídeo com as detecções e o relatório gerado em `reports/report.txt`.
4.  **Encerre:** Pressione 'q' na janela de vídeo para encerrar.

## Estrutura do Projeto

```
video-analysis-project/
├── .venv/            # Ambiente virtual (gerenciado pelo Poetry)
├── pyproject.toml    # Arquivo de configuração do projeto (Poetry)
├── poetry.lock      # Arquivo com as versões exatas das dependências (Poetry)
├── src/              # Código-fonte
│   ├── main.py       # Script principal
│   ├── video_processing.py  # Processamento de vídeo
│   ├── face_recognition_module.py   # Reconhecimento facial
│   ├── emotion_analysis_module.py  # Análise de emoções
│   ├── activity_detection_module.py # Detecção de atividades
│   └── report_module.py     # Geração de relatório
├── data/             # Dados (vídeos e imagens)
│   ├── videos/       # Vídeos de teste
│   └── images/       # Imagens de rostos conhecidos
├── reports/          # Relatórios gerados
└── README.md         # Este arquivo
```
## Features
    - Face Detection.
    - Face Recognition.
    - Análise de emoções
    - Detecção de pose.

## Limitações e problemas conhecidos

* A precisão da análise de emoções pode ser afetada por iluminação ruim, baixa resolução de vídeo e expressões faciais sutis/ambíguas.
* A detecção de atividade "Braços levantados" é um exemplo básico e pode precisar de ajustes nos limites de ângulo para diferentes ângulos de câmera e tipos de corpo.
* A lógica de detecção "Sentado/em pé" é básica e depende da visibilidade da parte inferior do corpo.
* A detecção de anomalias está atualmente vinculada a "Braços levantados" e é um exemplo muito simplificado.

## Melhorias futuras

* Implementar detecção mais robusta de sentar/ficar em pé.
* Melhorar a detecção de anomalias analisando a velocidade dos movimentos de pontos-chave.
* Adicionar mais detecções de atividade (por exemplo, "andar", "correr", "pular").
* Ajustar o modelo de detecção de emoções para melhor precisão.
* Adicionar uma Interface Gráfica do Usuário (GUI) simples para uso mais fácil.
* Melhorar o manuseio de várias pessoas na cena para detecção de atividade.
  
## Contribuições:

Contribuições são bem-vindas! Se você encontrar um bug ou quiser sugerir uma melhoria, abra um issue ou envie um pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You'll need to create a LICENSE file if you choose to use the MIT License).

## Contact

Giuan Miranda - [giuanm@live.com](mailto:giuanm@live.com)


------------------------------------------------------------------------------------------------------------------------------------

**English Version:**

## Tech Challenge - Video Analysis with Face Recognition, Emotion Detection, and Activity Detection

This project is a video analysis application developed as part of the Tech Challenge of Phase 4 of the "AI for Devs" course at FIAP - Pós Tech. The system uses computer vision and machine learning techniques to:

*   **Detect faces** in a video.
*   **Identify people** based on a set of reference images.
*   **Analyze facial expressions** (happy, sad, neutral, angry, fear, surprised, disgust) of the detected faces.
*   **Detect simple activities**, such as "Arms Raised" and "Person Sitting/Standing", using pose estimation (skeleton).
*   **Generate a report** with statistics on processed frames, detected anomalies, activities, and emotions.

The application is developed in Python and uses the following main libraries:

*   **OpenCV (cv2):** For video processing (reading, displaying, writing) and drawing graphical elements (rectangles, text).
*   **face_recognition:** For face detection and identification. (Based on dlib).
*   **dlib:** Used internally by face_recognition for face detection and facial landmarks.
*   **DeepFace:** For emotion analysis.
*   **MediaPipe:** For pose detection (skeleton) and activity inference.
*   **Poetry:** For dependency management and virtual environment.
*   **NumPy:** For numerical operations (used internally by many libraries).
*   **tqdm:** To display progress bars (optional, but useful).
*    **TensorFlow/tf-keras:** Used as backend pelo DeepFace.
*   **Gensim:** Para sumarização de texto
*   **Transformers:** Para sumarização de texto

## Technology

Here are the technologies used in this project:

*   Python 3.11
*   OpenCV
*   face_recognition (dlib)
*   DeepFace
*   MediaPipe
*   Poetry
*   NumPy
*   tqdm
* TensorFlow/tf-keras
* Gensim
* Transformers

## Services Used

*   GitHub

## Getting Started

**Prerequisites:**

*   Python 3.11 (or higher, but 3.11 was tested).
*   Poetry (for dependency management).
*   Visual Studio Build Tools with the "Desktop development with C++" workload (necessary to compile `dlib` on Windows).  You do *not* need the full Visual Studio IDE.
*   CMake (make sure to add it to the PATH during installation).
*   Git (optional, but recommended for cloning the repository).

**Installation:**

1.  **Clone the repository (optional):**
    ```bash
    git clone <URL of your repository>
    cd <project folder name>
    ```

2.  **Install Poetry (if you don't have it yet):** Follow the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation).

3.  **Install dependencies with Poetry:**
    ```bash
    poetry install
    ```
    This command creates the virtual environment and installs all necessary libraries, with the correct versions, defined in the `pyproject.toml` file.

4.  **Prepare the Data:**
    *   Create a `data/videos` folder and place a test video (`.mp4`) inside it. Name the video `video_fornecido.mp4` (or change the name in the `src/main.py` file).
    *   Create a `data/images` folder and place images of known faces inside it. Each image should contain *only one* face, and the filename should be the name of the person (e.g., `john.jpg`, `mary.png`).

5.  **Run the Application:**
    *   Activate the Poetry virtual environment:
        ```bash
        poetry shell
        ```
    *   Run the main script:
        ```bash
        python src/main.py
        ```

    The application will process the video, display a window with the results (rectangles on faces, names, emotions, and keypoints), and generate a report (`report.txt`) in the `reports/` folder. Press 'q' in the video window to stop the execution.

## How to Use

1.  **Prepare Data:** Place the test video in `data/videos` and known face images in `data/images`.
2.  **Run:** Execute `python src/main.py` within the Poetry virtual environment.
3.  **View:** Observe the video window with detections and the generated report in `reports/report.txt`.
4.  **Terminate:** Press 'q' in the video window to stop.

## Project Structure
(Same structure as in portuguese)

## Features
    - Face Detection.
    - Face Recognition.
    - Emotion Analysis.
    - Pose Detection.
    - Activity Detection (Arms Raised, Sitting/Standing - basic).
    - Report Generation (total frames, anomalies, activities, emotions).

## Limitations and Known Issues

*   The emotion analysis accuracy can be affected by poor lighting, low video resolution, and subtle/ambiguous facial expressions.
*   The "Arms Raised" activity detection is a basic example and may need adjustments to the angle thresholds for different camera angles and body types.
*   The "Sitting/Standing" detection logic is basic and relies on the visibility of the lower body.  It may require further refinement.
*   Anomaly detection is currently rudimentary (tied to "Arms Raised").
*   The MediaPipe warning "Using NORM_RECT without IMAGE_DIMENSIONS..." can be addressed in future improvements, but doesn't prevent the current functionality.

## Future Improvements (Optional)

*   Implement more robust sitting/standing detection.
*   Improve anomaly detection by analyzing the speed of keypoint movements (not just arm position).
*   Add more activity detections (e.g., "walking," "running," "jumping").
*   Fine-tune the emotion detection model for better accuracy (requires a labeled dataset of facial expressions).
*   Add a simple Graphical User Interface (GUI) for easier use (advanced).
*   Improve the handling of multiple people for activity detection to clearly associate activities with *specific* individuals even when they are close together or occlude each other.
*  Create unit tests for the different modules.

## Contributing

Contributions are welcome! If you find a bug or want to suggest an improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You'll need to create a LICENSE file if you choose to use the MIT License).  The MIT License is a very permissive open-source license.

## Contact

Giuan Miranda - [giuanm@live.com](mailto:giuanm@live.com)
