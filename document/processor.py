import asyncio
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from PyPDF2 import PdfReader

class DocumentProcessor:
    """Parse and chunk documents asynchronously."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, max_workers: int = 3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_jobs = {}

    async def process_file_async(self, file_path: str, job_id: str) -> Dict:
        """Process a single file asynchronously."""
        self.processing_jobs[job_id] = {"status": "processing", "progress": 0}

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(self.executor, self._extract_text, file_path)
            chunks = self._chunk_text(text)

            self.processing_jobs[job_id] = {
                "status": "completed",
                "chunks": chunks,
                "chunk_count": len(chunks),
                "progress": 100
            }
            return chunks
        except Exception as e:
            self.processing_jobs[job_id] = {
                "status": "failed",
                "error": str(e)
            }
            raise

    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF or TXT file."""
        path = Path(file_path)

        if path.suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif path.suffix in [".txt", ".md"]:
            return path.read_text()
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        text = []
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)

    def _chunk_text(self, text: str) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "chunk_id": len(chunks),
                "word_count": len(chunk_words)
            })

        return chunks

    def get_job_status(self, job_id: str) -> Dict:
        """Get processing job status."""
        return self.processing_jobs.get(job_id, {"status": "unknown"})
