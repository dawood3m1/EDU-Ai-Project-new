# RAG Ingestion Pipeline | خط إدخال وفهرسة المحتوى
# يعالج ملفات المقرر، يقسم المحتوى إلى chunks ويجهزه للفهرسة.
"""Incrementally ingest PDF files from knowledge_base/ into the RAG vector store.

Run any time you add a new chapter:
    python rag/ingest.py

OCR and embeddings both run fully locally (EasyOCR + sentence-transformers)
— no API key, account, or quota needed for ingestion. Only new or changed
files are processed (tracked in store/manifest.json by content hash) —
existing chapters are left untouched, so you can add chapter2.pdf,
chapter3.pdf, etc. later without rebuilding the whole index.
"""
import hashlib
import os
import sys

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ocr as rag_ocr  # noqa: E402
import store as rag_store  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
MIN_NATIVE_TEXT_CHARS = 40  # below this, treat the page as image-only and OCR it


# Computes a content hash of a PDF file, used to detect whether a file in
# knowledge_base/ is new, unchanged, or has been edited since the last run.
# تحسب هذه الدالة بصمة (hash) لمحتوى ملف PDF، تُستخدم لمعرفة هل الملف في
# knowledge_base/ جديد، أو بلا تغيير، أو عُدِّل منذ آخر تشغيل.
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# Gets the text of one PDF page: uses the page's native/embedded text when
# available (fast, free), and only falls back to OCR (rag/ocr.py) for pages
# that have no real text layer, such as scanned pages.
# تحصل هذه الدالة على نص صفحة PDF واحدة: تستخدم النص الأصلي المُضمَّن في
# الصفحة إن وُجد (سريع ومجاني)، ولا تلجأ لـOCR (rag/ocr.py) إلا للصفحات التي
# لا تحتوي طبقة نص حقيقية، مثل الصفحات الممسوحة ضوئيًا.
def extract_page_text(doc, page_index):
    page = doc.load_page(page_index)
    text = (page.get_text() or "").strip()
    if len(text) >= MIN_NATIVE_TEXT_CHARS:
        return text
    png_bytes = rag_ocr.render_page_png(doc, page_index)
    return rag_ocr.transcribe_page_image(png_bytes)


# The core ingestion routine for a single PDF: extracts each page's text
# (OCR when needed), splits it into chunks, embeds every chunk locally, and
# saves the results into the vector store — this is the PDF → chunks →
# embeddings → vector store part of the RAG pipeline. Skips files that
# haven't changed since the last run (via the manifest hash).
# روتين الفهرسة الأساسي لملف PDF واحد: يستخرج نص كل صفحة (مع OCR عند الحاجة)،
# يقسّمه إلى مقاطع (chunks)، يعمل embedding لكل مقطع محليًا، ويحفظ النتائج في
# الـvector store — وهذا هو جزء PDF ← chunks ← embeddings ← vector store من
# مسار RAG. يتخطّى الملفات التي لم تتغيّر منذ آخر تشغيل (عبر بصمة الـmanifest).
def ingest_file(filename, manifest):
    path = os.path.join(KB_DIR, filename)
    digest = file_hash(path)
    if manifest.get(filename, {}).get("hash") == digest:
        print(f"skip (unchanged): {filename}")
        return manifest

    # clear any chunks from a previous partial/interrupted run of this file
    # (or a prior version of it) before starting, so a resumed run never duplicates
    print(f"clearing any existing chunks for: {filename}")
    rag_store.remove_source(filename)

    print(f"ingesting: {filename}")
    doc = fitz.open(path)
    total_chunks = 0
    for page_index in range(doc.page_count):
        page_num = page_index + 1
        try:
            text = extract_page_text(doc, page_index)
        except Exception as e:
            print(f"  WARNING: page {page_num}/{doc.page_count} failed after retries, skipping it: {e}")
            continue
        if not text.strip():
            continue
        chunks = rag_store.chunk_text(text)
        page_records = []
        for i, chunk in enumerate(chunks):
            print(f"  page {page_num}/{doc.page_count} chunk {i + 1}/{len(chunks)}")
            try:
                embedding = rag_store.embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
            except Exception as e:
                print(f"  WARNING: embedding failed for page {page_num} chunk {i + 1}, skipping it: {e}")
                continue
            page_records.append({
                "source": filename,
                "page": page_num,
                "chunk_index": i,
                "text": chunk,
                "embedding": embedding,
            })
        # flush after every page so progress survives a crash/interruption instead
        # of being held in memory until the whole (possibly 90-page) file finishes
        rag_store.append_vectors(page_records)
        total_chunks += len(page_records)

    manifest[filename] = {"hash": digest, "chunks": total_chunks}
    rag_store.save_manifest(manifest)
    print(f"done: {filename} ({total_chunks} chunks)")
    return manifest


# Entry point: scans knowledge_base/ for PDF files and ingests each one that
# is new or changed. This is what makes the knowledge base incremental — add
# chapter2.pdf, chapter3.pdf, etc. later and re-run this without rebuilding
# the whole index.
# نقطة الدخول: تفحص مجلد knowledge_base/ بحثًا عن ملفات PDF وتُفهرس كل ملف
# جديد أو مُعدَّل. هذا ما يجعل قاعدة المعرفة تراكمية — أضف chapter2.pdf أو
# chapter3.pdf لاحقًا وأعد تشغيل هذا الملف دون إعادة بناء الفهرس بالكامل.
def main():
    if not os.path.isdir(KB_DIR):
        print(f"no knowledge_base directory at {KB_DIR}")
        return
    manifest = rag_store.load_manifest()
    pdfs = sorted(f for f in os.listdir(KB_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        print("no PDF files found in knowledge_base/")
        return
    for filename in pdfs:
        manifest = ingest_file(filename, manifest)


if __name__ == "__main__":
    main()
