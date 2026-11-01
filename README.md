# Customer Support RAG Copilot

An AI-powered customer support assistant that answers user questions using a knowledge base of customer support FAQs.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents and generate context-grounded answers.

## Features

- Document ingestion and processing
- FAQ-based knowledge retrieval
- Gemini embeddings
- ChromaDB vector storage
- Groq-powered LLM responses
- FastAPI backend
- Streamlit chat interface
- Source references in responses
- Fallback responses for unsupported questions
- API key authentication
- Retrieval and answer evaluation
- Pytest-based testing

## Architecture

```text
             User
               |
               v
        Streamlit Frontend
               |
               v
          FastAPI API
               |
         API Authentication
               |
               v
          RAG Pipeline
               |
        User Question
               |
               v
        Gemini Embeddings
               |
               v
          ChromaDB
               |
       Retrieve Relevant FAQs
               |
               v
        Retrieved Context
               |
               v
          Groq LLM
               |
               v
       Generated Answer
               |
               v
      Answer + Source Files
               |
               v
        Streamlit Frontend
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | LLM and RAG integration |
| Gemini | Text embeddings |
| ChromaDB | Vector database |
| Groq | LLM response generation |
| FastAPI | Backend API |
| Streamlit | Frontend interface |
| Pydantic | Request validation |
| Pytest | Automated testing |

## Project Structure

```text
customer_support_RAG_Copilot/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── ingestion.py
│   ├── rag.py
│   └── evaluate_answers.py
│
├── Frontend/
│   └── streamlit_app.py
│
├── data/
│   ├── support_faq.txt
│   ├── evaluation_dataset.json
│   ├── answer_evaluation_dataset.json
│   └── answer_evaluation_report.json
│
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   └── test_api.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer_support_RAG_Copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

GROQ_MODEL=openai/gpt-oss-120b
APP_API_KEY=your_application_api_key
```

Never commit your actual `.env` file or expose API keys publicly.

## Running the Application

### 1. Run document ingestion

```bash
python -m app.ingestion
```

This processes the FAQ document, creates embeddings, and stores them in ChromaDB.

### 2. Start the FastAPI backend

```bash
python -m uvicorn app.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### 3. Start the Streamlit frontend

Open another terminal:

```bash
python -m streamlit run Frontend/streamlit_app.py
```

Frontend:

http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/ask` | Ask a support question |

### Example request

```json
{
  "question": "What is the refund policy?"
}
```

The `/ask` endpoint requires an `X-API-Key` request header.

## RAG Workflow

1. Load customer support documents.
2. Split and process FAQ content.
3. Generate embeddings using Gemini.
4. Store embeddings in ChromaDB.
5. Convert the user's question into an embedding.
6. Retrieve relevant FAQ documents.
7. Pass retrieved context to the Groq model.
8. Generate a context-grounded response.
9. Return the answer and its sources.

## Evaluation and Testing

Run automated tests:

```bash
python -m pytest -v
```

The project includes:

- Retrieval evaluation using a question dataset.
- Answer evaluation using a word-overlap baseline metric.
- API endpoint tests.
- RAG pipeline tests.
- Authentication and validation tests.

**Note:** Word overlap is a basic lexical evaluation metric. It does not independently establish answer correctness or faithfulness.

## Security

- API keys are stored in environment variables.
- `.env` is excluded from Git.
- The `/ask` endpoint uses API key authentication.
- Input questions are validated using Pydantic.

## Future Improvements

- Deploy the backend and frontend.
- Improve answer evaluation with semantic and faithfulness metrics.
- Add production monitoring and observability.

## Author

SANDEEP

Aspiring GenAI and Agentic AI Engineer