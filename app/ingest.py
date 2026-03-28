import re
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    all_pages_text: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is not None and text.strip():
            all_pages_text.append(text)
        else:
            print(f"Warning: page {page_number} returned empty text.")

    return "\n".join(all_pages_text)


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()