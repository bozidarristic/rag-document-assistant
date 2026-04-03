import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from app.core.models import Document


def _build_pdf_document_id(path: Path) -> str:
    resolved_path = str(path.resolve())
    path_hash = hashlib.sha1(resolved_path.encode("utf-8")).hexdigest()[:12]

    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).strip("_").lower()
    if not safe_stem:
        safe_stem = "document"

    return f"pdf::{safe_stem}::{path_hash}"


def load_pdf(pdf_path: str | Path) -> Document:
    path = Path(pdf_path)
    reader = PdfReader(str(path))

    pages_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    content = "\n".join(pages_text)

    document_id = _build_pdf_document_id(path)

    return Document(
        source_id=document_id,
        source_type="pdf",
        content=content,
        metadata={
            "source_type": "pdf",
            "source_name": path.name,
            "file_name": path.name,
            "file_path": str(path.resolve()),
        },
    )