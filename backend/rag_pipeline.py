"""
OmniSense RAG Pipeline — Component-based Document Ingestion & Retrieval.

Components:
  1. SmartDocumentLoader  — Format Router + OCR Fallback
  2. VietnameseTextSplitter — Chunk text without breaking Vietnamese words
  3. RAGPipeline           — Embedding + ChromaDB Vector Store + Query
"""

import os
import re
import glob
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF

# Optional dependencies — guarded imports
_HAS_DOCX = False
try:
    from docx import Document as DocxDocument
    _HAS_DOCX = True
except ImportError:
    pass

_HAS_OCR = False
try:
    from pdf2image import convert_from_path
    import pytesseract
    _HAS_OCR = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ── Paths (absolute, based on this file's location) ────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = PROJECT_ROOT / "data" / "rag_docs"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# ── Standardized embedding model (lightweight, CPU-friendly) ───
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ================================================================
# Component 1 & 2: Smart Document Loader (with OCR fallback)
# ================================================================

class SmartDocumentLoader:
    """Reads .txt, .docx, and .pdf files. Falls back to OCR for scanned PDFs."""

    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load plain-text file with common encoding fallbacks."""
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        log.error(f"Không đọc được mã hoá ký tự của {file_path}")
        return ""

    @staticmethod
    def load_docx(file_path: str) -> str:
        if not _HAS_DOCX:
            log.error("Thư viện python-docx chưa được cài. Chạy: pip install python-docx")
            return ""
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    @staticmethod
    def _ocr_pdf(file_path: str) -> str:
        """
        Pure Python OCR for scanned PDFs using RapidOCR + ONNXRuntime.
        Zero external desktop apps (.exe / Tesseract / Poppler) required!
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            import numpy as np
            import cv2

            ocr_engine = RapidOCR()
            filename = os.path.basename(file_path)
            log.info(f"🔍 Đang dùng RapidOCR (Pure ONNX) bóc chữ PDF scan: {filename}...")

            doc = fitz.open(file_path)
            total_pages = len(doc)
            parts = []

            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=150)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                result, _ = ocr_engine(img_bgr)
                if result:
                    page_text = "\n".join([line[1] for line in result if line[1].strip()])
                    if page_text.strip():
                        parts.append(page_text)
                        log.info(f"  📖 OCR Trang {i+1}/{total_pages}: {len(page_text)} ký tự")

            doc.close()
            return "\n\n".join(parts)

        except Exception as e:
            log.error(f"Lỗi RapidOCR: {e}")
            return ""

    @staticmethod
    def load_pdf(file_path: str) -> str:
        doc = fitz.open(file_path)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        text = "\n".join(pages_text)

        # If text is very short for the number of pages → likely scanned
        avg_chars = len(text.strip()) / max(len(pages_text), 1)
        if avg_chars < 30:
            ocr_text = SmartDocumentLoader._ocr_pdf(file_path)
            if ocr_text.strip():
                return ocr_text

        return text

    @staticmethod
    def load(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            ".txt": SmartDocumentLoader.load_txt,
            ".doc": SmartDocumentLoader.load_docx,
            ".docx": SmartDocumentLoader.load_docx,
            ".pdf": SmartDocumentLoader.load_pdf,
        }
        loader = loaders.get(ext)
        if loader is None:
            log.warning(f"Định dạng {ext} không được hỗ trợ.")
            return ""
        return loader(file_path)


# ================================================================
# Component 3: Vietnamese Text Splitter (no LangChain dependency)
# ================================================================

class VietnameseTextSplitter:
    """
    Splits text into chunks of ~chunk_size characters with overlap,
    breaking at sentence/paragraph boundaries to preserve Vietnamese meaning.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Separators ordered by priority (paragraphs > sentences > words)
        self._separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive newlines
        text = re.sub(r"[ \t]+", " ", text)       # collapse whitespace

        chunks = self._recursive_split(text, 0)
        # Filter out empty/tiny chunks
        return [c.strip() for c in chunks if len(c.strip()) > 20]

    def _recursive_split(self, text: str, sep_idx: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if sep_idx >= len(self._separators):
            # Hard split as last resort
            return self._hard_split(text)

        sep = self._separators[sep_idx]
        parts = text.split(sep)

        if len(parts) == 1:
            # This separator didn't help, try the next one
            return self._recursive_split(text, sep_idx + 1)

        chunks = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the part itself is too large, recursively split it
                if len(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            result.append(overlap_text + chunks[i])
        return result


def get_optimal_device() -> str:
    """Tự động kiểm tra phần cứng: Dùng CUDA nếu có GPU NVIDIA, ngược lại dùng CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            log.info("⚡ Đã phát hiện GPU (NVIDIA CUDA) -> Khởi chạy RAG Embedding trên GPU!")
            return "cuda"
    except Exception:
        pass
    log.info("💻 Không có GPU CUDA khả dụng -> Chạy RAG Embedding trên CPU.")
    return "cpu"


class CustomSentenceTransformerEF(chromadb.EmbeddingFunction):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        device = get_optimal_device()
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = self.model.encode(list(input), convert_to_numpy=True)
        return [[float(x) for x in emb] for emb in embeddings]


# ================================================================
# Component 4: RAG Pipeline (Embed + Store + Query)
# ================================================================

class RAGPipeline:
    """Singleton-friendly pipeline for document ingestion and retrieval."""

    def __init__(self):
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        DB_DIR.mkdir(parents=True, exist_ok=True)

        self.splitter = VietnameseTextSplitter(chunk_size=800, chunk_overlap=200)

        self.ef = CustomSentenceTransformerEF(model_name=EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            "omnisense_rag", embedding_function=self.ef
        )

    # ── Ingestion ──────────────────────────────────────────────
    def process_file(self, file_path: str) -> bool:
        """Extract text → chunk → embed → store in ChromaDB."""
        filename = os.path.basename(file_path)
        log.info(f"📄 Đang xử lý: {filename}")

        text = SmartDocumentLoader.load(file_path)
        if not text.strip():
            log.warning(f"Không đọc được nội dung từ {filename}.")
            return False

        chunks = self.splitter.split_text(text)
        if not chunks:
            log.warning(f"Không tách được chunk từ {filename}.")
            return False

        # Remove old chunks for this file first (idempotent upsert)
        try:
            self.collection.delete(where={"source": filename})
        except Exception:
            pass  # collection might be empty

        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        self.collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
        log.info(f"✅ Đã thêm {len(chunks)} chunks từ {filename} vào database.")
        return True

    def process_all(self) -> None:
        """Scan and process all supported files in the docs folder."""
        files = []
        for ext in ("*.pdf", "*.txt", "*.docx", "*.doc"):
            files.extend(glob.glob(str(DOC_DIR / ext)))

        if not files:
            log.info("Không tìm thấy tài liệu nào trong " + str(DOC_DIR))
            return

        for f in files:
            self.process_file(f)

    # ── Removal ────────────────────────────────────────────────
    def remove_file(self, filename: str) -> None:
        """Delete all chunks of a file from ChromaDB."""
        log.info(f"🗑️ Xóa {filename} khỏi vector database...")
        try:
            self.collection.delete(where={"source": filename})
        except Exception as e:
            log.error(f"Lỗi khi xóa {filename}: {e}")
        log.info(f"Đã xóa dữ liệu vector của {filename}.")

    # ── Retrieval ──────────────────────────────────────────────
    def query(self, query_text: str, k: int = 5, distance_threshold: float = 0.85) -> str:
        """
        Retrieve relevant document chunks using Hybrid Vector + Keyword Search.
        Supports both accented native PDFs and unaccented OCR scanned PDFs.
        """
        retrieved_docs = []

        # 1. Vector Search
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                include=["documents", "distances"],
            )
            if results and results.get("documents") and results["documents"][0]:
                for doc, dist in zip(results["documents"][0], results["distances"][0]):
                    if dist <= distance_threshold:
                        retrieved_docs.append(doc)
        except Exception as e:
            log.error(f"Lỗi truy xuất Vector RAG: {e}")

        # 2. Exact Regex Keyword Matching cho tài liệu OCR không dấu (vd: tin chi, hoc phan)
        try:
            import unicodedata
            unaccented_q = "".join(
                c for c in unicodedata.normalize("NFKD", query_text) if not unicodedata.combining(c)
            ).lower()

            # Nếu truy vấn chứa từ khóa tín chỉ (kể cả có dấu lẫn không dấu)
            if re.search(r'\btín\s+chỉ\b|\btin\s+chi\b', query_text, re.IGNORECASE):
                all_data = self.collection.get()
                if all_data and all_data.get("documents"):
                    credit_chunks = [
                        d for d in all_data["documents"]
                        if re.search(r'\btin\s+chi\b|\btín\s+chỉ\b', d, re.IGNORECASE)
                    ]
                    if credit_chunks:
                        # Ưu tiên các đoạn chứa định nghĩa "2. Tin chi la..."
                        def_chunks = [c for c in credit_chunks if "dai luong" in c.lower() or "khoi luong" in c.lower() or "gio tin chi" in c.lower()]
                        other_credit = [c for c in credit_chunks if c not in def_chunks]
                        prioritized = def_chunks + other_credit
                        retrieved_docs = prioritized[:3] + [d for d in retrieved_docs if d not in prioritized]
        except Exception as e:
            log.error(f"Lỗi truy xuất Exact Keyword RAG: {e}")

        # Loại bỏ các đoạn thông tin liên hệ chung nếu đã có kết quả RAG chuyên biệt
        if len(retrieved_docs) > 1:
            clean_docs = [d for d in retrieved_docs if "facebook.com/UET.VNUH" not in d]
            if clean_docs:
                retrieved_docs = clean_docs

        return "\n\n".join(retrieved_docs)


# ── Singleton accessor for use in app.py ───────────────────────
_pipeline_instance: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Lazy singleton — model is loaded only once."""
    global _pipeline_instance
    if _pipeline_instance is None:
        log.info("Khởi tạo RAG Pipeline (lần đầu)...")
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


# ── CLI entrypoint ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.process_all()
