# RAG Document Assistant

A modular Retrieval-Augmented Generation (RAG) pipeline for semantic search over unstructured documents.

This project implements a clean and extensible architecture for document ingestion, text processing, embedding generation, and similarity-based retrieval, forming the foundation for production-ready RAG systems.

---

## Overview

The system processes documents and enables semantic querying by retrieving the most relevant text segments based on vector similarity.

### Pipeline

1. **Ingestion**  
   - Extract raw text from PDF documents  

2. **Processing**  
   - Clean and normalize text  
   - Split text into overlapping chunks  

3. **Embedding**  
   - Convert text chunks into dense vector representations using transformer-based models  

4. **Retrieval**  
   - Compute similarity between a query and indexed chunks  
   - Return the most relevant context  

The current implementation focuses on building a robust and modular retrieval pipeline.  
Future iterations will extend this into a full RAG system with generation and evaluation capabilities.

---

## Features

- PDF text extraction using PyPDF  
- Text cleaning and normalization  
- Overlapping chunking strategy for context preservation  
- Embedding generation with SentenceTransformers  
- Cosine similarity-based retrieval  
- Modular architecture designed for extensibility  

---

## Architecture

The project is organized into clearly separated components:

```bash
app/
  core/
    config.py        # Global configuration and constants
    models.py        # Core data structures (Document, Chunk, RetrievedChunk)
    logging.py       # Logging configuration

  ingestion/
    ingest.py        # PDF loading and extraction
    text_cleaner.py  # Text preprocessing

  indexing/
    chunking.py      # Chunk creation with overlap
    embeddings.py    # Embedding model and encoding logic

  retrieval/
    retrieval.py     # Similarity computation and top-k retrieval

  main.py            # End-to-end pipeline execution
```

---

## Tech Stack

- Python  
- PyPDF  
- SentenceTransformers  
- NumPy  

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the pipeline:

```bash
python -m app.main
```

### Example Query

```text
What is a mammal?
```

### Output

- Top-k most relevant text chunks  
- Similarity scores for each retrieved result  

---

## Design Principles

- **Modularity** – Each stage of the pipeline is isolated and reusable  
- **Clarity** – Minimal abstractions, explicit data flow  
- **Extensibility** – Designed to support vector databases, APIs, and LLM integration  
- **Reproducibility** – Deterministic processing pipeline  

---

## Roadmap

- Persistent vector storage (FAISS / ChromaDB)  
- Multi-document support and collections  
- FastAPI backend for querying  
- LLM-based answer generation (full RAG)  
- Retrieval vs generation evaluation  
- Streamlit interface for interaction  
- Dockerization and deployment  

---

## License

This project is licensed under the MIT License.
