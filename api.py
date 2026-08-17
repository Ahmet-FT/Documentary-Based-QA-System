"""
api.py
======
FastAPI backend — Kaynak Göstermeli Soru-Cevap Sistemi.

Mevcut qa_engine, retriever, llm, vectorstore modüllerini
REST API olarak sunar. Bileşenler startup'ta bir kez yüklenir.

Çalıştırma:
    python main.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Proje kök dizini ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ingestion_pipeline import IngestionPipeline
from app.retriever import Retriever
from app.llm import LLMManager
from app.qa_engine import QAEngine
from app.embeddings import EmbeddingManager
from app.vectorstore import VectorStore
from app.config import settings


# ── Sabitler (.env'den yüklenir) ─────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_MB = settings.MAX_FILE_SIZE_MB
MAX_TEXT_LENGTH = settings.MAX_TEXT_LENGTH
MAX_QUERY_LENGTH = settings.MAX_QUERY_LENGTH


# ── Pydantic modelleri ───────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0
    source_filter: Optional[Union[str, List[str]]] = None
    temperature: float = settings.OLLAMA_TEMPERATURE
    model: str = settings.OLLAMA_MODEL


class TextIndexRequest(BaseModel):
    text: str
    doc_name: str = "metin_girisi.txt"
    chunk_mode: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 128


# ── Singleton sistem bileşenleri ─────────────────────────────────

_pipeline: Optional[IngestionPipeline] = None
_retriever: Optional[Retriever] = None
_qa: Optional[QAEngine] = None
_indexed_files: List[str] = []


def _get_system():
    """Sistem bileşenlerini döndürür, gerekirse başlatır."""
    global _pipeline, _retriever, _qa
    if _qa is None:
        print("  [*] Sistem bileskenleri yukleniyor...")
        _pipeline = IngestionPipeline()
        _retriever = Retriever(vector_store=_pipeline.vector_store)
        _llm = LLMManager()
        _qa = QAEngine(retriever=_retriever, llm_manager=_llm)
        print("  [OK] Sistem hazir.")
    return _pipeline, _retriever, _qa


# ── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlangıcında sistem bileşenlerini yükler."""
    _get_system()
    yield


# ── FastAPI uygulaması ───────────────────────────────────────────

app = FastAPI(
    title="Kaynak Göstermeli Soru-Cevap Sistemi",
    lifespan=lifespan,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ── Statik dosya sunumu ──────────────────────────────────────────

@app.get("/")
async def index():
    """Ana sayfa — frontend HTML."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={}, status_code=204)


# ── API: Sağlık kontrolü ────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Ollama ve VectorStore durumunu kontrol eder."""
    try:
        _, _, qa = _get_system()
        return qa.health_check()
    except Exception as e:
        return {"error": str(e), "ollama_server": False,
                "ollama_model": False, "model_name": "?",
                "vectorstore_chunks": -1}


# ── API: Dosya yükleme ve indeksleme ────────────────────────────

@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    chunk_mode: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(128),
):
    """Yüklenen dosyaları indeksler."""
    pipeline, _, _ = _get_system()

    pipeline.chunker.mode = type(pipeline.chunker.mode)(chunk_mode)
    pipeline.chunker.chunk_size = chunk_size
    pipeline.chunker.chunk_overlap = chunk_overlap

    results = []
    for uf in files:
        # ── Dosya türü kontrolü ──
        ext = os.path.splitext(uf.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "display_name": uf.filename,
                "error": f"Desteklenmeyen dosya turu: '{ext}'. "
                         f"Desteklenen: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            })
            continue

        content = await uf.read()

        # ── Boş dosya kontrolü ──
        if not content or len(content) == 0:
            results.append({"display_name": uf.filename, "error": "Dosya bos."})
            continue

        # ── Dosya boyutu kontrolü ──
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            results.append({
                "display_name": uf.filename,
                "error": f"Dosya cok buyuk (maks {MAX_FILE_SIZE_MB} MB).",
            })
            continue

        suffix = "." + uf.filename.rsplit(".", 1)[-1] if "." in uf.filename else ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            stats = pipeline.ingest(tmp_path, show_progress=False,
                                    original_name=uf.filename)
            stats["display_name"] = uf.filename
            results.append(stats)
            if uf.filename not in _indexed_files:
                _indexed_files.append(uf.filename)
        except Exception as e:
            results.append({"display_name": uf.filename, "error": str(e)})
        finally:
            os.unlink(tmp_path)

    return {"results": results, "indexed_files": _indexed_files}


# ── API: Metin yapıştırma ile indeksleme ─────────────────────────

@app.post("/api/upload-text")
async def upload_text(req: TextIndexRequest):
    """Yapıştırılan metni dosya olarak indeksler."""
    # ── Boş metin kontrolü ──
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin bos olamaz.")

    # ── Çok uzun metin kontrolü ──
    if len(req.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Metin cok uzun ({len(req.text):,} karakter). "
                   f"Maksimum: {MAX_TEXT_LENGTH:,} karakter.",
        )

    pipeline, _, _ = _get_system()

    pipeline.chunker.mode = type(pipeline.chunker.mode)(req.chunk_mode)
    pipeline.chunker.chunk_size = req.chunk_size
    pipeline.chunker.chunk_overlap = req.chunk_overlap

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(req.text)
        tmp_path = tmp.name

    try:
        stats = pipeline.ingest(tmp_path, show_progress=False,
                                original_name=req.doc_name)
        stats["display_name"] = req.doc_name
        if req.doc_name not in _indexed_files:
            _indexed_files.append(req.doc_name)
        return {"result": stats, "indexed_files": _indexed_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── API: Soru-cevap ─────────────────────────────────────────────

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """Kullanıcı sorusunu cevaplar — kaynak göstermeli."""
    # ── Boş soru kontrolü ──
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Soru bos olamaz.")

    # ── Çok uzun soru kontrolü ──
    if len(req.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Soru cok uzun ({len(req.query)} karakter). "
                   f"Maksimum: {MAX_QUERY_LENGTH} karakter.",
        )

    _, _, qa = _get_system()

    # ── Doküman yüklenmemiş kontrolü ──
    chunk_count = qa.health_check().get("vectorstore_chunks", 0)
    if chunk_count <= 0:
        raise HTTPException(
            status_code=400,
            detail="Henuz dokuman yuklenmedi. Once 'Dokuman Yukle' "
                   "sekmesinden dokuman yukleyin.",
        )

    qa.llm.model = req.model
    qa.llm.temperature = req.temperature

    health = qa.health_check()
    if not health["ollama_server"]:
        raise HTTPException(
            status_code=503,
            detail="Ollama sunucusu çalışmıyor. Lütfen Ollama'yı başlatın.",
        )
    if not health["ollama_model"]:
        raise HTTPException(
            status_code=503,
            detail=f"'{req.model}' modeli bulunamadı. "
                   f"Terminal'de: ollama pull {req.model}",
        )

    try:
        result = qa.ask(
            query=req.query,
            top_k=req.top_k,
            min_score=req.min_score,
            source_filter=req.source_filter,
            temperature=req.temperature,
        )

        retrieval_data = []
        for r in result.retrieval_results[:len(result.sources)]:
            retrieval_data.append({
                "rank": r.rank,
                "text": r.text,
                "source": r.source,
                "page": str(r.page),
                "chunk_index": r.chunk_index,
                "score": r.score,
            })

        return {
            **result.to_dict(),
            "retrieval_details": retrieval_data,
        }
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── API: Dosya listesi ──────────────────────────────────────────

@app.get("/api/files")
async def list_files():
    """İndekslenen dosyaların listesini ChromaDB'den döndürür."""
    pipeline, _, _ = _get_system()
    db_files = pipeline.vector_store.get_indexed_files()
    # Dosya başına chunk sayısı bilgisi ekle
    files_detail = []
    for f in db_files:
        chunk_count = pipeline.vector_store.get_file_chunk_count(f)
        files_detail.append({"name": f, "chunks": chunk_count})
    return {"files": db_files, "files_detail": files_detail}


# ── API: Tek doküman silme ───────────────────────────────────────

@app.post("/api/files/delete")
async def delete_file(payload: dict):
    """
    Belirli bir dokümanın tüm chunk'larını ChromaDB'den siler.

    Body: {"file_name": "rapor.pdf"}
    """
    file_name = payload.get("file_name", "").strip()
    if not file_name:
        raise HTTPException(status_code=400, detail="Dosya adı boş olamaz.")

    pipeline, _, _ = _get_system()
    deleted = pipeline.vector_store.delete_by_source(file_name)

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"'{file_name}' adlı doküman bulunamadı.",
        )

    # Bellek listesinden de kaldır
    if file_name in _indexed_files:
        _indexed_files.remove(file_name)

    return {
        "status": "ok",
        "file": file_name,
        "deleted_chunks": deleted,
        "message": f"'{file_name}' silindi ({deleted} chunk).",
    }


# ── API: Veritabanı sıfırlama ───────────────────────────────────

@app.post("/api/reset")
async def reset_database():
    """ChromaDB koleksiyonunu sıfırlar."""
    global _pipeline, _retriever, _qa, _indexed_files

    em = EmbeddingManager()
    vs = VectorStore(embed_manager=em, reset=True)
    _pipeline = IngestionPipeline(vector_store=vs)
    _retriever = Retriever(vector_store=vs)
    _llm = LLMManager()
    _qa = QAEngine(retriever=_retriever, llm_manager=_llm)
    _indexed_files = []

    return {"status": "ok", "message": "Veritabanı sıfırlandı."}


# ── API: İstatistikler ──────────────────────────────────────────

@app.get("/api/stats")
async def stats():
    """Sistem istatistikleri."""
    pipeline, _, qa = _get_system()
    h = qa.health_check()
    db_files = pipeline.vector_store.get_indexed_files()
    return {
        "vectorstore_chunks": h["vectorstore_chunks"],
        "indexed_files_count": len(db_files),
        "files": db_files,
        "ollama_server": h["ollama_server"],
        "ollama_model": h["ollama_model"],
        "model_name": h["model_name"],
    }
