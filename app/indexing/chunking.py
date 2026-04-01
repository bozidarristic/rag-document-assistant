from app.core.models import Chunk, Document


def chunk_text(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    text = document.content
    chunks: list[Chunk] = []

    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_value = text[start:end]

        chunks.append(
            Chunk(
                chunk_id=f"{document.source_id}_chunk_{chunk_index}",
                document_id=document.source_id,
                text=chunk_text_value,
                start_char=start,
                end_char=end,
                metadata=document.metadata.copy(),
            )
        )

        if end == len(text):
            break

        start += chunk_size - overlap
        chunk_index += 1

    return chunks