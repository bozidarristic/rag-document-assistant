from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "animals_sample_book.pdf"


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)

    print(f"Number of pages: {len(reader.pages)}")

    all_pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is not None and text.strip():
            all_pages_text.append(text)
        else:
            print(f"Warning: page {page_number} returned empty text.")

    full_text = "\n".join(all_pages_text)
    return full_text


def main():
    print(f"Resolved PDF path: {PDF_PATH}")

    if not PDF_PATH.exists():
        print(f"File not found: {PDF_PATH}")
        return

    text = extract_text_from_pdf(PDF_PATH)

    print("\n--- FIRST 2000 CHARACTERS ---\n")
    print(text[:2000])

    print("\n--- STATS ---")
    print(f"Total characters: {len(text)}")
    print(f"Total words: {len(text.split())}")


if __name__ == "__main__":
    main()