# RAG Retrieval Engine | محرك الاسترجاع
# مسؤول عن embeddings وvector store وsimilarity search واسترجاع أكثر المقاطع ارتباطًا بسؤال الطالب.
"""Shared RAG storage: chunking, embeddings, and similarity search.

The vector store is a flat JSONL file (store/vectors.jsonl) plus a manifest
(store/manifest.json) that tracks which source files have already been
ingested (by content hash). This lets rag/ingest.py add new chapters later
by only embedding what's new, without touching existing chapters.

Embeddings run fully locally via sentence-transformers — no external API,
no account, no quota. Only /api/chat (the final answer) still calls Gemini.
"""
import json
import math
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(BASE_DIR, "store")
VECTORS_PATH = os.path.join(STORE_DIR, "vectors.jsonl")
MANIFEST_PATH = os.path.join(STORE_DIR, "manifest.json")

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

_embedder = None
_vectors_cache = None  # in-memory cache of vectors.jsonl; populated once, kept in sync on writes


# Loads the embedding model once (lazily, on first use) and reuses the same
# instance for every later call, instead of reloading it on every request.
# تحمّل هذه الدالة نموذج الـ Embedding مرة واحدة فقط (عند أول استخدام)
# وتُعيد استخدام نفس النسخة لكل استدعاء لاحق، بدل إعادة تحميله مع كل طلب.
def _get_embedder():
    global _embedder
    if _embedder is None:
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
        print(f"[rag/store] embedding model '{EMBED_MODEL}' loaded in {time.time() - t0:.2f}s")
    return _embedder


# Forces both the embedding model and the vector store into memory right
# away, so a student's first question doesn't pay the loading cost.
# proxy_server.py calls this once when the backend starts up.
# تُجبر هذه الدالة تحميل نموذج الـ Embedding والـ vector store في الذاكرة
# فورًا، حتى لا يدفع أول سؤال من الطالب تكلفة التحميل. يستدعيها proxy_server.py
# مرة واحدة فقط عند تشغيل الباك إند.
def warmup():
    """Force the embedding model and the vector store to load into memory now,
    instead of on the first incoming request. Call once at server startup."""
    t0 = time.time()
    _get_embedder()
    n = len(load_vectors())
    print(f"[rag/store] warmup done in {time.time() - t0:.2f}s ({n} chunk(s) in memory)")


# Splits the extracted course text into overlapping, paragraph-aware chunks
# small enough to embed and retrieve individually. The overlap keeps context
# from being lost right at a chunk boundary. Used by rag/ingest.py.
# تقسّم هذه الدالة نص المقرر المُستخرج إلى مقاطع (chunks) متداخلة جزئيًا تراعي
# حدود الفقرات، بحجم مناسب لعمل embedding واسترجاع كل مقطع على حدة. التداخل
# يمنع فقدان السياق عند حدود المقطع. تُستخدم من rag/ingest.py.
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}" if buf else para
            continue
        if buf:
            chunks.append(buf)
        if len(para) > chunk_size:
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append(para[i:i + chunk_size])
            buf = ""
        else:
            buf = para
    if buf:
        chunks.append(buf)

    overlapped = []
    for i, c in enumerate(chunks):
        if i > 0 and overlap > 0:
            c = chunks[i - 1][-overlap:] + "\n" + c
        overlapped.append(c.strip())
    return [c for c in overlapped if c]


# Converts a piece of text (a course chunk during ingestion, or the
# student's question during retrieval) into a numeric embedding vector using
# the local sentence-transformers model — this is the core "Embeddings" step
# of the RAG pipeline, and it never leaves the machine.
# تحوّل هذه الدالة نصًا (مقطع من المقرر أثناء الفهرسة، أو سؤال الطالب أثناء
# الاسترجاع) إلى متجه رقمي (embedding) باستخدام نموذج sentence-transformers
# المحلي — وهذه هي خطوة "Embeddings" الأساسية في مسار RAG، ولا تغادر الجهاز أبدًا.
def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    """Embed text locally with sentence-transformers (task_type kept for call-site
    compatibility with the previous Google-embeddings version; unused locally)."""
    embedder = _get_embedder()
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


# Measures how similar two embedding vectors are (1 = identical direction,
# 0 = unrelated) — this is the similarity metric used to rank stored chunks
# against the student's question.
# تقيس هذه الدالة مدى تشابه متجهين (1 = نفس الاتجاه تمامًا، 0 = لا علاقة) —
# وهي مقياس التشابه المستخدم لترتيب المقاطع المخزّنة مقابل سؤال الطالب.
def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Reads every stored chunk (text + embedding + source/page metadata) from
# the vectors.jsonl file on disk. This is the raw disk-loading step; callers
# should normally use load_vectors() instead so it only runs once.
# تقرأ هذه الدالة كل مقطع مخزَّن (النص + الـembedding + بيانات المصدر والصفحة)
# من ملف vectors.jsonl على القرص. هذه هي خطوة القراءة الخام من القرص؛ يجب أن
# تستخدم بقية الكود load_vectors() بدلها حتى لا تُعاد القراءة إلا مرة واحدة.
def _read_vectors_from_disk():
    if not os.path.exists(VECTORS_PATH):
        return []
    records = []
    with open(VECTORS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# This is the project's in-memory vector store: it returns the cached list
# of chunk embeddings, reading vectors.jsonl from disk only the first time
# it's called in this process — every later request reuses the same
# in-memory copy instead of re-reading the file.
# هذه هي الـvector store الفعلية للمشروع في الذاكرة: تُعيد قائمة الـembeddings
# المخزَّنة مؤقتًا، وتقرأ ملف vectors.jsonl من القرص أول مرة فقط في هذه العملية —
# كل طلب لاحق يستخدم نفس النسخة في الذاكرة بدل إعادة قراءة الملف.
def load_vectors():
    """Returns the in-memory vector cache, reading it from disk only once
    (per process) instead of on every call/request."""
    global _vectors_cache
    if _vectors_cache is None:
        t0 = time.time()
        _vectors_cache = _read_vectors_from_disk()
        print(f"[rag/store] vector store loaded from disk in {time.time() - t0:.2f}s ({len(_vectors_cache)} chunk(s))")
    return _vectors_cache


# Appends newly-embedded chunks to vectors.jsonl (used by rag/ingest.py
# right after ingesting a PDF page) and keeps the in-memory cache in sync so
# a running server process never sees stale data during a single run.
# تُضيف هذه الدالة المقاطع المُشفَّرة (embedded) حديثًا إلى ملف vectors.jsonl
# (يستخدمها rag/ingest.py مباشرة بعد فهرسة صفحة PDF)، وتُبقي النسخة في الذاكرة
# متزامنة حتى لا تكون بيانات عملية السيرفر قديمة أثناء نفس التشغيل.
def append_vectors(records):
    if not records:
        return
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(VECTORS_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    load_vectors().extend(records)  # keep the in-memory cache consistent with disk


# Removes every chunk that came from a given source PDF, so rag/ingest.py
# can safely re-index a file that changed without leaving duplicate/stale
# chunks behind.
# تحذف هذه الدالة كل مقطع مصدره ملف PDF معيّن، حتى يستطيع rag/ingest.py إعادة
# فهرسة ملف تغيّر محتواه بأمان دون ترك مقاطع مكرّرة أو قديمة.
def remove_source(source_filename):
    """Drop all stored chunks for a given source file (used when re-ingesting a changed file)."""
    global _vectors_cache
    records = [r for r in load_vectors() if r["source"] != source_filename]
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(VECTORS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _vectors_cache = records


# Reads manifest.json, which tracks which source PDFs have already been
# ingested (by content hash), so rag/ingest.py knows what's new vs. unchanged.
# تقرأ هذه الدالة ملف manifest.json الذي يتتبّع أي ملفات PDF تمت فهرستها
# مسبقًا (عبر بصمة المحتوى/hash)، حتى يعرف rag/ingest.py ما هو جديد وما لم يتغيّر.
def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Writes the updated manifest back to disk after a file finishes ingesting.
# تكتب هذه الدالة الـmanifest المُحدَّث على القرص بعد انتهاء فهرسة ملف.
def save_manifest(manifest):
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# This is the main retrieval function of the RAG pipeline: it embeds the
# student's question, compares it against every stored chunk embedding with
# cosine similarity, and returns the top_k most relevant chunks. It is called
# from proxy_server.py's /api/rag/query endpoint.
# هذه هي دالة الاسترجاع الرئيسية في مسار RAG: تعمل embedding لسؤال الطالب،
# تقارنه بكل embedding مخزَّن باستخدام cosine similarity، وتُعيد أكثر top_k
# مقاطع ارتباطًا. يستدعيها proxy_server.py من نقطة /api/rag/query.
def search(query, top_k=4):
    """Embed the query and return the top_k most relevant stored chunks."""
    records = load_vectors()  # in-memory after the first call; no disk re-read
    if not records:
        return []

    t0 = time.time()
    query_vec = embed_text(query, task_type="RETRIEVAL_QUERY")
    embed_ms = (time.time() - t0) * 1000

    t1 = time.time()
    scored = [
        {
            "text": r["text"],
            "source": r["source"],
            "page": r.get("page"),
            "chunk_index": r["chunk_index"],
            "score": _cosine(query_vec, r["embedding"]),
        }
        for r in records
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    search_ms = (time.time() - t1) * 1000

    print(f"[rag/store] query embedding: {embed_ms:.1f}ms | vector search over {len(records)} chunk(s): {search_ms:.1f}ms")
    return scored[:top_k]
