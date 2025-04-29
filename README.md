# GradeMate

This project aims to revolutionize exam grading by developing GradeMate, an AIpowered automated grading system for exam papers. It addresses the challenge of
manual, time-consuming, and inconsistent exam evaluation processes in educational
institutions. The motivation behind this work is to enhance grading efficiency,
fairness, and scalability.
We utilized NLP techniques, LLMs (Large Language Models), and RAG (RetrievalAugmented Generation) methodologies. The system architecture includes a
frontend dashboard, backend grading engine, and custom RAG pipelines.
Results show that GradeMate achieves high semantic grading accuracy compared to
manual grading, drastically reducing grading time and offering consistent
evaluation across different answer styles.
In conclusion, GradeMate successfully demonstrates the feasibility of AI-driven
descriptive exam evaluation and opens avenues for future work, such as expanding
support for subjective creativity assessment and multilingual grading.

## Tech Stack

- **Full Stack Development**: Python, Django
- **AI Models**: GroqCloud, Deepseek, Gemini, Llama, Google AI Studio
- **Embeddings**: Nomic.ai
- **RAG Frameworks**: LangChain, LangGraph
- **Vector Databases**: Chroma

## Project Structure

```
GradeMate/
├── ai/                     # AI models and related scripts
├── app/                    # Django application files
├── assets/                 # README.md assets
├── docs/                   # Documentation and resources
├── downloaded_models/      # Pre-trained models
├── extraction/             # Data extraction utilities
├── mvp/                    # Minimum Viable Product implementation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Containerization setup
└── README.md               # Project overview and setup instructions
```

##  Installation & Setup

### Via Repository Clone
Follow these steps to set up the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/safwanhamza/GradeMate.git
cd GradeMate
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Apply Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Run the Development Server

```bash
python manage.py runserver
```

Access the application at [http://localhost:8000](http://localhost:8000).

### Via Docker

To containerize the application using Docker:

### 1. Build the Docker Image

```bash
docker build -t grademate-app
```

### 2. Run the Docker Container

```bash
docker run -d -p 8000:8000 grademate-app
```


## Documentation
Documentation is available in the [docs/](docs/) directory.
