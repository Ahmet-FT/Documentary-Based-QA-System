# Source Citation QA System

Türkçe ve çok dilli dökümanlar için **semantik arama** ve **kaynak göstermeli soru-cevap** sistemi.  
LlamaIndex · ChromaDB · HuggingFace Embeddings · FastAPI · Ollama üzerine inşa edilmiştir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [API Endpoint'leri](#api-endpointleri)
- [Modüller](#modüller)
- [Desteklenen Dosya Formatları](#desteklenen-dosya-formatları)
- [Chunking Stratejileri](#chunking-stratejileri)
- [Yapılandırma (.env)](#yapılandırma-env)
- [Bağımlılıklar](#bağımlılıklar)

---

## Genel Bakış

Bu sistem bir dokümanı alıp şu adımlardan geçirir:

```
Doküman → Yükle → Temizle → Chunk → Embedding → ChromaDB
                                                      ↓
                                  Sorgu → Retriever → LLM → Cevap + Kaynaklar
```

Kullanıcı herhangi bir dokümanı sisteme yükler; sistem metni parçalara ayırır, her parça için 1024 boyutlu anlam vektörü üretir ve ChromaDB'ye kaydeder. Sorgu anında kullanıcının sorusu da vektöre dönüştürülür, en yakın parçalar bulunur ve Ollama üzerinden çalışan bir LLM kaynak göstererek cevap üretir.

---

## Mimari

```
┌─────────────────────────────────────────────────────┐
│                  IngestionPipeline                  │
│                                                     │
│  DocumentLoader → Chunker → EmbeddingManager        │
│                                  ↓                  │
│                            VectorStore (ChromaDB)   │
│                                  ↓                  │
│                            Retriever → LLMManager   │
│                                  ↓                  │
│                             QAEngine                │
└─────────────────────────────────────────────────────┘
                        ↑
              FastAPI REST API (api.py)
              Statik HTML Arayüzü (static/index.html)
```

| Katman | Teknoloji | Görev |
|---|---|---|
| Doküman Yükleme | PyMuPDF, python-docx | PDF/DOCX/TXT okuma, metadata ekleme |
| Metin Temizleme | Özel `text_cleaner` | Gürültü giderme, normalize etme |
| Parçalama | LlamaIndex SentenceSplitter | Anlamlı chunk'lara bölme |
| Embedding | `multilingual-e5-large` | 1024-boyutlu çok dilli vektör |
| Vektör DB | ChromaDB (persist) | Cosine benzerlik araması |
| Retrieval | `Retriever` modülü | Top-K, skor filtresi, metadata |
| LLM | Ollama (llama3.1:8b) | Kaynak göstermeli cevap üretimi |
| API | FastAPI + Uvicorn | REST API, statik dosya sunumu |

---

## Proje Yapısı

```
Source_citation_QA_System/
│
├── app/                        # Ana Python paketi
│   ├── api.py                  # FastAPI uygulama ve tüm endpoint'ler
│   ├── config.py               # .env tabanlı merkezi yapılandırma
│   ├── ingestion_pipeline.py   # Uçtan uca pipeline (Yükle→Chunk→Embed→Kaydet)
│   │
│   ├── loaders/                # Doküman yükleyiciler
│   │   ├── document_loader.py  # Birleşik giriş noktası (PDF/DOCX/TXT/MD)
│   │   ├── pdf_loader.py       # PyMuPDF tabanlı
│   │   ├── docx_loader.py      # python-docx tabanlı
│   │   ├── txt_loader.py       # Düz metin
│   │   └── text_cleaner.py     # Metin normalize ve temizleme
│   │
│   ├── chunkers/               # Parçalama stratejileri
│   │   ├── chunker.py          # Birleşik arayüz (ChunkMode enum)
│   │   ├── fixed_size_chunker.py
│   │   ├── overlap_chunker.py
│   │   └── paragraph_chunker.py
│   │
│   ├── embeddings/             # Embedding yönetimi
│   │   └── embedding_manager.py  # multilingual-e5-large, lazy loading
│   │
│   ├── vectorstore/            # Vektör veritabanı
│   │   └── vector_store.py     # ChromaDB wrapper (upsert, query, stats)
│   │
│   ├── retriever/              # Retrieval katmanı
│   │   └── retriever.py        # Top-K arama, skor filtresi, raporlama
│   │
│   ├── llm/                    # LLM yönetimi
│   │   ├── llm_manager.py      # Ollama HTTP istemcisi
│   │   └── prompt_templates.py # Sistem ve kullanıcı prompt şablonları
│   │
│   ├── qa_engine/              # Soru-cevap motoru
│   │   └── qa_engine.py        # Retriever + LLM orchestration
│   │
│   └── static/
│       └── index.html          # Web arayüzü (tarayıcıdan doğrudan erişilir)
│
├── chroma_db/                  # ChromaDB kalıcı depolama (otomatik oluşur)
├── .env                        # Yerel yapılandırma (git'e eklenmez)
├── .env.example                # Yapılandırma şablonu
└── README.md
```

---

## Kurulum

### Gereksinimler

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) (önerilen) veya pip
- [Ollama](https://ollama.com) — LLM için

### 1. Sanal ortam oluştur ve bağımlılıkları yükle

```powershell
# uv ile (önerilen)
cd Source_citation_QA_System/app
uv sync

# veya pip ile
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. .env dosyasını yapılandır

```powershell
copy .env.example .env
# .env dosyasını düzenle (Ollama model adı, port vb.)
```

### 3. Ollama'yı başlat ve modeli indir

```powershell
ollama serve
ollama pull llama3.1:8b
```

> **Not:** `multilingual-e5-large` embedding modeli (~2.2 GB) ilk çalıştırmada HuggingFace'den otomatik indirilir.

---

## Hızlı Başlangıç

### Sunucuyu başlat

```powershell
# Proje kök dizininden (Source_citation_QA_System/):
app\.venv\Scripts\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Tarayıcıda `http://localhost:8000` adresine git — web arayüzü otomatik açılır.

### Alternatif: doğrudan uvicorn

```powershell
cd app
uvicorn api:app --reload
```

---

## API Endpoint'leri

Tüm endpoint'ler `http://localhost:8000` üzerinden erişilebilir.  
İnteraktif API dokümantasyonu için: `http://localhost:8000/docs`

| Metot | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/` | Web arayüzü (index.html) |
| `GET` | `/api/health` | Ollama ve VectorStore durumu |
| `GET` | `/api/stats` | Sistem istatistikleri |
| `GET` | `/api/files` | İndekslenen dosyaların listesi |
| `POST` | `/api/upload` | Dosya yükle ve indeksle (multipart/form-data) |
| `POST` | `/api/upload-text` | Yapıştırılan metni indeksle (JSON) |
| `POST` | `/api/ask` | Soru sor, kaynaklı cevap al (JSON) |
| `POST` | `/api/files/delete` | Belirli bir dokümanı sil (JSON) |
| `POST` | `/api/reset` | Tüm veritabanını sıfırla |

### Örnek: Soru Sorma

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Yapay zekanın etik sorunları nelerdir?", "top_k": 5}'
```

```json
{
  "answer": "...",
  "sources": ["rapor.pdf", "makale.txt"],
  "retrieval_details": [
    {
      "rank": 1,
      "score": 0.87,
      "text": "...",
      "source": "rapor.pdf",
      "page": "3"
    }
  ]
}
```

### Örnek: Dosya Yükleme

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "files=@rapor.pdf" \
  -F "chunk_mode=recursive" \
  -F "chunk_size=512" \
  -F "chunk_overlap=128"
```

---

## Modüller

### `IngestionPipeline`

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `chunk_mode` | `"recursive"` | Parçalama stratejisi |
| `chunk_size` | `512` | Token / karakter boyutu |
| `chunk_overlap` | `128` | Ardışık chunk örtüşmesi |
| `embed_manager` | `None` (otomatik) | EmbeddingManager örneği |
| `vector_store` | `None` (otomatik) | VectorStore örneği |

**Metodlar:** `ingest(file_path)` · `ingest_many(file_paths)` · `search(query, n)` · `retriever` (property)

---

### `QAEngine`

Retriever ve LLMManager'ı birleştirerek kaynak göstermeli cevap üretir.

```python
from app.qa_engine import QAEngine
from app.retriever import Retriever
from app.llm import LLMManager

qa = QAEngine(retriever=Retriever(...), llm_manager=LLMManager())
result = qa.ask("Sorum nedir?", top_k=5, min_score=0.0)

print(result.answer)
print(result.sources)
```

---

### `Retriever`

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `default_top_k` | `5` | Döndürülecek sonuç sayısı |
| `min_score` | `0.0` | Minimum benzerlik eşiği |
| `vector_store` | `None` (otomatik) | VectorStore örneği |

**Metodlar:** `retrieve(query, top_k, min_score, source_filter)` · `retrieve_with_report(...)` · `stats()`

**`RetrievalResult` alanları:**

| Alan | Tip | Açıklama |
|---|---|---|
| `rank` | int | Sıralama (1'den başlar) |
| `score` | float | Cosine benzerlik skoru (0–1) |
| `text` | str | Chunk metni |
| `source` | str | Kaynak dosya adı |
| `page` | any | Sayfa numarası |
| `chunk_index` | int | Chunk sıra numarası |
| `metadata` | dict | Tüm metadata alanları |
| `doc_id` | str | ChromaDB ID |

---

### `VectorStore`

ChromaDB üzerine persist edilebilir vektör deposu. Cosine benzerlik kullanır.

```python
from app.vectorstore import VectorStore
from app.embeddings import EmbeddingManager

vs = VectorStore(
    embed_manager=EmbeddingManager(),
    collection_name="source_citation_qa",  # varsayılan
    persist_dir="./chroma_db",            # varsayılan
    reset=False,                           # True → koleksiyonu sıfırla
)

vs.add_chunks(chunks)                       # chunk listesi ekle
results = vs.query("soru", n=5)            # QueryResult listesi döner
print(vs.stats())                           # istatistikler
vs.clear()                                  # koleksiyonu sıfırla
```

---

### `EmbeddingManager`

**Model:** `intfloat/multilingual-e5-large` · **Boyut:** 1024 · **Dil:** 100+

```python
from app.embeddings import EmbeddingManager

em = EmbeddingManager(device="cpu")       # veya "cuda"
vec  = em.embed_query("Sorum nedir?")     # [float] × 1024
vecs = em.embed_passages(["metin1", ...]) # [[float]] × N
```

E5 modeli `query:` ve `passage:` prefix'lerini otomatik ekler.

---

## Desteklenen Dosya Formatları

| Format | Kütüphane | Sayfa Desteği |
|---|---|---|
| `.pdf` | PyMuPDF | ✅ Sayfa sayfa |
| `.docx` | python-docx | Tek belge |
| `.txt` | built-in | Tek belge |
| `.md` | built-in | Tek belge |

---

## Chunking Stratejileri

| Mod | Açıklama | Önerilen Kullanım |
|---|---|---|
| `recursive` ⭐ | Paragraf → cümle → kelime → karakter fallback zinciri | **Genel kullanım (varsayılan)** |
| `sentence` | LlamaIndex SentenceSplitter, cümle sınırı korumalı | Uzun akademik metinler |
| `paragraph` | Paragraf/bölüm semantik birimi | Yapılandırılmış raporlar |
| `overlap` | Kayan pencere + overlap | Kısa/sık referans metinler |
| `fixed` | Sabit karakter penceresi | Debug / test |

---

## Yapılandırma (.env)

`.env.example` dosyasını kopyalayarak `.env` oluşturun:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Kullanılacak Ollama modeli |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama sunucu adresi |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM yanıt sıcaklığı |
| `OLLAMA_TIMEOUT` | `120` | İstek zaman aşımı (saniye) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | HuggingFace embedding modeli |
| `EMBEDDING_DEVICE` | (otomatik) | `cpu` veya `cuda` |
| `CHROMA_COLLECTION` | `source_citation_qa` | ChromaDB koleksiyon adı |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB depolama dizini |
| `SERVER_HOST` | `0.0.0.0` | Sunucu bind adresi |
| `SERVER_PORT` | `8000` | Sunucu portu |
| `MAX_FILE_SIZE_MB` | `50` | Yüklenebilecek maksimum dosya boyutu |
| `MAX_TEXT_LENGTH` | `500000` | Yapıştırılan metin için karakter limiti |
| `MAX_QUERY_LENGTH` | `2000` | Sorgu için karakter limiti |

---

## Bağımlılıklar

```
Python       >= 3.12
fastapi      >= 0.115.0   # REST API framework
uvicorn      >= 0.34.0    # ASGI sunucu
llama-index  >= 0.14.23   # RAG orkestrasyon
chromadb     >= 1.5.9     # Vektör veritabanı
transformers >= 4.40.0    # HuggingFace model yükleme
torch        >= 2.0.0     # Model çalıştırma (CPU/GPU)
pymupdf      >= 1.28.0    # PDF okuma
python-docx  >= 1.2.0     # DOCX okuma
python-dotenv >= 1.2.2    # .env desteği
```
