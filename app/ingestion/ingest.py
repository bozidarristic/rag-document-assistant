from pathlib import Path

from pypdf import PdfReader

from app.core.models import Document


def load_pdf(pdf_path: str | Path) -> Document:
    path = Path(pdf_path)
    reader = PdfReader(str(path))

    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")

    content = "\n".join(pages_text)

    return Document(
        source_id=path.stem,
        source_type="pdf",
        content=content,
        metadata={
            "file_name": path.name,
            "path": str(path),
        },
    )