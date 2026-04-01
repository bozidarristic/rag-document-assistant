from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"

DEFAULT_PDF_PATH = DATA_DIR / "animals_sample_book.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"