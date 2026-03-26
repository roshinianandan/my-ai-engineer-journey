import os
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text.strip()


def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a plain text file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.

    chunk_size: number of characters per chunk
    overlap: number of characters shared between consecutive chunks

    Why overlap? So that sentences split across chunk boundaries
    are not lost — both chunks contain the boundary content.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to end at a sentence boundary for cleaner chunks
        if end < len(text):
            # Look for the last period, newline or space before the cut
            for boundary in [". ", "\n", " "]:
                boundary_pos = text.rfind(boundary, start, end)
                if boundary_pos != -1:
                    end = boundary_pos + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # move back by overlap for next chunk

    return chunks


def load_documents(docs_folder: str) -> list:
    """
    Load all .pdf and .txt files from a folder.
    Returns a list of dicts with filename, text, and chunks.
    """
    documents = []
    supported = [".pdf", ".txt"]

    if not os.path.exists(docs_folder):
        print(f"Folder not found: {docs_folder}")
        return []

    files = [f for f in os.listdir(docs_folder)
             if any(f.endswith(ext) for ext in supported)]

    if not files:
        print(f"No PDF or TXT files found in {docs_folder}")
        return []

    for filename in files:
        filepath = os.path.join(docs_folder, filename)
        ext = os.path.splitext(filename)[1].lower()

        print(f"Loading: {filename}")

        if ext == ".pdf":
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)

        chunks = chunk_text(text)
        documents.append({
            "filename": filename,
            "filepath": filepath,
            "text": text,
            "chunks": chunks,
            "num_chunks": len(chunks)
        })

        print(f"  Extracted {len(text)} characters → {len(chunks)} chunks")

    return documents