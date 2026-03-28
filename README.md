# RAG Document Assistant

Production-minded Retrieval-Augmented Generation (RAG) system for document question answering over PDF files.

This project implements a modular pipeline for processing documents, generating embeddings, performing semantic retrieval, and serving answers based on relevant context.

---

## Features

- PDF text extraction using PyPDF
- Text cleaning and preprocessing
- Overlapping chunking strategy
- Embedding generation with SentenceTransformers
- Semantic search via cosine similarity
- Modular and extensible pipeline design

---

## Project Overview

The system follows a standard RAG pipeline:

1. **Ingestion** – Load and extract text from PDF documents  
2. **Processing** – Clean and split text into overlapping chunks  
3. **Embedding** – Convert text chunks into vector representations  
4. **Retrieval** – Find the most relevant chunks for a given query  

> The current version focuses on building a clean and modular retrieval pipeline.  
> Future iterations will include vector databases, LLM-based answer generation, and evaluation.

---

## Tech Stack

- Python
- PyPDF
- SentenceTransformers
- NumPy

---

## Project Structure

```
app/
  config.py        # Configuration and constants
  ingest.py        # PDF loading and text cleaning
  chunking.py      # Text chunking logic
  embeddings.py    # Embedding model and encoding
  retrieval.py     # Similarity and retrieval logic
  main.py          # Entry point

data/
  animals_sample_book.pdf

tests/
```

---

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

---

## Run

```bash
python -m app.main
```

---

## Example

Query:

```
What is a mammal?
```

Output:
- Top-k most relevant text chunks
- Similarity scores for each retrieved chunk

---

## Roadmap

- Persistent vector database (ChromaDB / FAISS)
- Query interface (CLI + API)
- LLM-based answer generation (full RAG)
- RAG vs non-RAG comparison
- Evaluation framework for answer quality
- Streamlit demo interface
- Dockerization

---

## License

This project is licensed under the MIT License.