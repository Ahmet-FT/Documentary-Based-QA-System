# Source Citation QA System

Türkçe ve çok dilli dökümanlar için **semantik arama** ve **kaynak göstermeli soru-cevap** sistemi.  
LlamaIndex · ChromaDB · HuggingFace Embeddings · Streamlit üzerine inşa edilmiştir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Modüller](#modüller)
- [Streamlit Arayüzü](#streamlit-arayüzü)
- [Desteklenen Dosya Formatları](#desteklenen-dosya-formatları)
- [Chunking Stratejileri](#chunking-stratejileri)
- [Bağımlılıklar](#bağımlılıklar)

---

## Genel Bakış

Bu sistem bir dokümanı alıp şu adımlardan geçirir:

```
Doküman → Yükle → Temizle → Chunk → Embedding → ChromaDB
                                                      ↓
                                  Sorgu → Retriever → Top-K Sonuç
```

Kullanıcı herhangi bir dokümanı sisteme yükler; sistem metni parçalara ayırır, her parça için 1024 boyutlu anlam vektörü üretir ve ChromaDB'ye kaydeder. Sorgu anında kullanıcının sorusu da vektöre dönüştürülür ve en yakın parçalar benzerlik skoru ile birlikte döndürülür.

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
│                            Retriever                │
└─────────────────────────────────────────────────────┘
         ↑                                   ↑
   Pages/indexing_test.py (Streamlit UI)    API / Script
```

| Katman | Teknoloji | Görev |
|---|---|---|
| Doküman Yükleme | PyMuPDF, python-docx | PDF/DOCX/TXT okuma, metadata ekleme |
| Metin Temizleme | Özel `text_cleaner` | Gürültü giderme, normalize etme |
| Parçalama | LlamaIndex SentenceSplitter | Anlamlı chunk'lara bölme |
| Embedding | `multilingual-e5-large` | 1024-boyutlu çok dilli vektör |
| Vektör DB | ChromaDB (persist) | Cosine benzerlik araması |
| Retrieval | `Retriever` modülü | Top-K, skor filtresi, metadata |
| Arayüz | Streamlit | Web tabanlı test ve demo |

---

## Proje Yapısı

```
Source_citation_QA_System/
│
├── app/                        # Ana Python paketi
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
│   └── retriever/              # Retrieval katmanı
│       └── retriever.py        # Top-K arama, skor filtresi, raporlama
│
├── Pages/                      # Streamlit sayfaları
│   └── indexing_test.py        # Doküman yükle & sorgu test arayüzü
│
├── chroma_db/                  # ChromaDB kalıcı depolama (otomatik oluşur)
├── Documents/                  # Örnek dokümanlar
│
├── inspect_chunks.py           # Chunk inceleme aracı
├── test_chunking.py            # Chunking testleri
├── test_embedding.py           # Embedding testleri
├── test_vectorstore.py         # VectorStore testleri
└── README.md
```

---

## Kurulum

### Gereksinimler

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) (önerilen) veya pip

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

### 2. Bağımlılık listesi (`pyproject.toml`)

```
chromadb >= 1.5.9
llama-index >= 0.14.23
llama-index-embeddings-huggingface >= 0.4.0
pymupdf >= 1.28.0
python-docx >= 1.2.0
streamlit >= 1.60.0
torch >= 2.0.0
transformers >= 4.40.0
sentence-transformers >= 3.0.0
```

> **Not:** `multilingual-e5-large` modeli (~2.2 GB) ilk çalıştırmada HuggingFace'den otomatik indirilir.

---

## Hızlı Başlangıç

### Streamlit Arayüzü (Önerilen)

```powershell
# Proje kök dizininden:
app\.venv\Scripts\python.exe -m streamlit run Pages/indexing_test.py
```

Tarayıcıda `http://localhost:8501` adresine git.

---

### Python API ile Kullanım

#### Tek dosya indeksleme

```python
from app.ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline(
    chunk_mode="recursive",  # önerilen strateji
    chunk_size=512,
    chunk_overlap=128,
)

stats = pipeline.ingest("Documents/rapor.pdf")
print(stats)
# {'file': '...', 'pages': 12, 'chunks': 87, 'stored': 87, 'elapsed_sec': 4.3}
```

#### Birden fazla dosya

```python
stats_list = pipeline.ingest_many([
    "Documents/rapor.pdf",
    "Documents/sunum.docx",
    "Documents/notlar.txt",
])
```

#### Sorgulama (Retriever)

```python
from app.retriever import Retriever

retriever = Retriever(
    vector_store=pipeline.vector_store,
    default_top_k=5,
    min_score=0.0,      # 0.0 = filtre yok; 0.5+ = yalnızca yüksek benzerlik
)

results = retriever.retrieve("Yapay zekanın etik sorunları nelerdir?", top_k=5)

for r in results:
    print(f"#{r.rank} | Skor: {r.score:.4f} | Kaynak: {r.source} | Sayfa: {r.page}")
    print(f"  {r.text[:200]}...\n")
```

#### Konsolda raporlu arama

```python
results = retriever.retrieve_with_report(
    query="Makine öğrenmesi nedir?",
    top_k=3,
    min_score=0.4,
)
```

#### Pipeline üzerinden kısayol

```python
# pipeline.retriever → aynı VectorStore'a bağlı Retriever (lazy)
results = pipeline.retriever.retrieve("Sorum nedir?", top_k=5)
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

## Streamlit Arayüzü

`Pages/indexing_test.py` aşağıdaki özellikleri sunar:

| Bölüm | Özellik |
|---|---|
| **Sol Panel — Yükleme** | PDF / DOCX / TXT yükle veya metin yapıştır, tek tıkla indeksle |
| **Sağ Panel — Sorgulama** | Serbest metin sorgusu, top-k sonuç görüntüleme |
| **Sidebar** | Chunk stratejisi, boyut, overlap, top-k, min-score ayarları |
| **Sonuç Kartları** | Skor renk kodu (yeşil/sarı/kırmızı), metadata pill'leri, chunk metni |
| **JSON Export** | Ham sonuçları JSON olarak görüntüle |
| **Koleksiyon Yönetimi** | Veritabanını tek tıkla sıfırla |

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

## Bağımlılıklar

```
Python       >= 3.12
llama-index  >= 0.14.23    # RAG orkestrasyon
chromadb     >= 1.5.9      # Vektör veritabanı
transformers >= 4.40.0     # HuggingFace model yükleme
torch        >= 2.0.0      # Model çalıştırma (CPU/GPU)
pymupdf      >= 1.28.0     # PDF okuma
python-docx  >= 1.2.0      # DOCX okuma
streamlit    >= 1.60.0     # Web arayüzü
```
