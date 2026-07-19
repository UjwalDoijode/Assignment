"""PDF parsing with pdfplumber - extracts text and tables."""
import pdfplumber
from typing import Iterator
from pathlib import Path


class PDFChunk:
    """A chunk of content from a PDF."""
    
    def __init__(self, content: str, page: int, section: str):
        self.content = content
        self.page = page
        self.section = section  # "text" or "table_N"


def parse_pdf(pdf_path: Path) -> Iterator[PDFChunk]:
    """
    Parse a PDF and yield chunks of text and tables.
    
    Tables are converted to pipe-delimited markdown.
    Text is chunked at ~400 words with 60-word overlap.
    
    Args:
        pdf_path: Path to the PDF file
        
    Yields:
        PDFChunk objects containing content, page number, and section type
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract tables first
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if table:
                    # Convert table to markdown
                    markdown_table = _table_to_markdown(table)
                    if markdown_table.strip():
                        yield PDFChunk(
                            content=markdown_table,
                            page=page_num,
                            section=f"table_{table_idx + 1}"
                        )
            
            # Extract text
            text = page.extract_text()
            if text:
                # Chunk the text
                for chunk in _chunk_text(text, chunk_size=400, overlap=60):
                    if chunk.strip():
                        yield PDFChunk(
                            content=chunk,
                            page=page_num,
                            section="text"
                        )


def _table_to_markdown(table: list[list]) -> str:
    """Convert a table (list of lists) to pipe-delimited markdown."""
    if not table or not table[0]:
        return ""
    
    lines = []
    
    # Header row
    header = table[0]
    lines.append("| " + " | ".join(str(cell or "") for cell in header) + " |")
    
    # Separator
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    
    # Data rows
    for row in table[1:]:
        if row:
            lines.append("| " + " | ".join(str(cell or "") for cell in row) + " |")
    
    return "\n".join(lines)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> Iterator[str]:
    """
    Split text into overlapping chunks based on word count.
    
    Args:
        text: Text to chunk
        chunk_size: Target words per chunk
        overlap: Words to overlap between chunks
        
    Yields:
        Text chunks
    """
    words = text.split()
    
    if len(words) <= chunk_size:
        yield text
        return
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        yield " ".join(chunk_words)
        
        if end >= len(words):
            break
        
        start = end - overlap


def get_first_n_pages_text(pdf_path: Path, n: int = 3) -> str:
    """
    Extract text from the first N pages of a PDF.
    
    Used for generating collection descriptions during ingestion.
    
    Args:
        pdf_path: Path to PDF file
        n: Number of pages to extract
        
    Returns:
        Combined text from first N pages
    """
    text_parts = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:n]:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    
    return "\n\n".join(text_parts)
