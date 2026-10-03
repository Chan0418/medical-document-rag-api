# Medical Document RAG API

This is a work-in-progress educational project for building a medical-document Retrieval-Augmented Generation (RAG) system. The project focuses on extracting text from medical PDFs using an API, splitting the extracted text into chunks, creating embeddings, retrieving relevant source sections and generating grounded answers with citations.

The goal is to learn how APIs, document extraction, embeddings, retrieval and language models can work together in a medical-document question-answering pipeline.

## Project Status

Work in progress. This repository contains a base project structure and starter code for the API and RAG pipeline.

## Main Features Planned

- Upload a medical PDF to an extraction API
- Extract text and page metadata from the PDF
- Split extracted text into chunks
- Generate embeddings for document chunks
- Retrieve relevant chunks based on a medical question
- Generate grounded answers using retrieved sources
- Include citations and medical safety behaviour

## Repository Structure

```text
medical-document-rag-api/
├── api/
│   └── extract_api.py
├── src/
│   ├── rag_pipeline.py
│   └── config.py
├── notebooks/
│   └── medical_rag_learning_template.ipynb
├── docs/
│   └── medical_rag_api_guide.pdf
├── sample_documents/
│   └── README.md
├── outputs/
│   └── README.md
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## How It Works

```text
Medical PDF
   ↓
Extraction API
   ↓
Extracted text and page metadata
   ↓
Chunking
   ↓
Embeddings
   ↓
Retrieval
   ↓
Grounded answer with citations
```

## Setup

Clone the repository:

```bash
git clone https://github.com/your-username/medical-document-rag-api.git
cd medical-document-rag-api
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```text
OPENAI_API_KEY=your_api_key_here
```

## Running the Extraction API

```bash
uvicorn api.extract_api:app --reload --port 8000
```

The extraction endpoint will be available at:

```text
http://127.0.0.1:8000/extract
```

## Disclaimer

This project is an educational prototype. It is not intended to diagnose, prescribe, replace a clinician, provide emergency support or make personalised medical decisions. Answers should be grounded only in the supplied documents and should recommend consulting qualified healthcare professionals for personal medical concerns.

## License

This repository is provided for academic and portfolio purposes. See the `LICENSE` file for details.
