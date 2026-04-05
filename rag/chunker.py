import os
from pypdf import PdfReader

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file page by page."""
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


# ── CHUNKING STRATEGIES ───────────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Strategy 1: Fixed-size chunking with overlap.
    Fast and simple. Good baseline for most documents.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def chunk_sentence(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list:
    """
    Strategy 2: Sentence-aware chunking.
    Groups complete sentences together — no sentence is ever cut in half.
    Much better for preserving meaning than fixed-size cutting.
    overlap: number of sentences shared between consecutive chunks.
    """
    if not NLTK_AVAILABLE:
        print("NLTK not available — falling back to fixed chunking.")
        return chunk_fixed(text)

    sentences = sent_tokenize(text)
    chunks = []
    start = 0

    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def chunk_paragraph(text: str, max_chunk_size: int = 800) -> list:
    """
    Strategy 3: Paragraph-aware chunking.
    Splits on double newlines (natural paragraph boundaries).
    Merges short paragraphs together until reaching max_chunk_size.
    Best for structured documents like reports and articles.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= max_chunk_size:
            current += (" " if current else "") + para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def chunk_text(text: str, strategy: str = "sentence", **kwargs) -> list:
    """
    Unified chunking interface. Choose strategy by name.
    strategy: 'fixed' | 'sentence' | 'paragraph'
    """
    strategies = {
        "fixed": chunk_fixed,
        "sentence": chunk_sentence,
        "paragraph": chunk_paragraph
    }
    fn = strategies.get(strategy, chunk_sentence)
    return fn(text, **kwargs)


def load_documents(docs_folder: str, strategy: str = "sentence") -> list:
    """
    Load all PDF and TXT files from a folder.
    Returns list of dicts with filename, text, chunks, and metadata.
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
        name = os.path.splitext(filename)[0]

        print(f"Loading: {filename}")

        if ext == ".pdf":
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)

        chunks = chunk_text(text, strategy=strategy)

        # Auto-generate metadata from filename
        # e.g. "ml_basics_beginner.txt" → topic=ml_basics, level=beginner
        parts = name.lower().replace("-", "_").split("_")
        level = "general"
        for lvl in ["beginner", "intermediate", "advanced"]:
            if lvl in parts:
                level = lvl
                break

        documents.append({
            "filename": filename,
            "filepath": filepath,
            "name": name,
            "text": text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "metadata": {
                "source": filename,
                "level": level,
                "file_type": ext.replace(".", ""),
                "strategy": strategy
            }
        })

        print(f"  Strategy: {strategy} | "
              f"Characters: {len(text)} | "
              f"Chunks: {len(chunks)}")

    return documents