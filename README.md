# GenAI RAG Document Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system for querying large PDF documents.

## Features
- PDF ingestion
- Text extraction from PDF files
- Chunking
- Embeddings with SentenceTransformers
- Vector search
- Retrieval-Augmented Question Answering
- FastAPI interface

## Tech Stack
- Python
- PyPDF
- SentenceTransformers
- ChromaDB
- FastAPI
- Docker

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
RAG_PDF/
│
├── app/               # Core application logic
├── data/              # Input PDF files and vector store
├── notebooks/         # Experiments and analysis
├── requirements.txt   # Project dependencies
├── README.md          # Project documentation
└── .gitignore         # Files ignored by Git
```

## Current Status
This project is currently under development.  
The first implemented step is PDF ingestion and text extraction.

## Next Steps
- Extract text from PDF documents
- Clean and chunk text
- Generate embeddings
- Store embeddings in a vector database
- Implement retrieval
- Build question-answering pipeline
- Expose the system through FastAPI