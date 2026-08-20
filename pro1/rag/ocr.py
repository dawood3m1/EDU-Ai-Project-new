# Document Text Extraction | استخراج نص المستندات
# يستخرج النص من صفحات المستندات عند الحاجة قبل الفهرسة.
"""Page-image transcription for PDFs that have no embedded text layer
(e.g. scanned/print-to-PDF textbook chapters).

Runs fully locally with EasyOCR — no external API, no account, no quota.
EasyOCR downloads its detection/recognition model weights once (via pip's
cache) the first time it runs, then works completely offline afterward.
"""
import os

OCR_LANGS = os.environ.get("RAG_OCR_LANGS", "en").split(",")

_reader = None


# Loads the local EasyOCR model once and reuses it for every page, instead
# of reloading the OCR model for every single page image.
# تحمّل هذه الدالة نموذج EasyOCR المحلي مرة واحدة فقط وتُعيد استخدامه لكل
# صفحة، بدل إعادة تحميل نموذج الـOCR مع كل صورة صفحة.
def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(OCR_LANGS, gpu=False)
    return _reader


# Renders one PDF page to a PNG image, so pages with no embedded text layer
# (scanned pages) can be handed to the OCR reader below.
# تُحوّل هذه الدالة صفحة PDF واحدة إلى صورة PNG، حتى يمكن تمرير الصفحات
# الممسوحة ضوئيًا (بدون طبقة نص) إلى قارئ الـOCR أدناه.
def render_page_png(doc, page_index, dpi=150):
    page = doc.load_page(page_index)
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


# Runs OCR on a rendered page image and returns the extracted plain text.
# This is the fallback text-extraction path used by rag/ingest.py only when
# a PDF page has no native/embedded text to read directly.
# تُشغّل هذه الدالة الـOCR على صورة الصفحة المُصيَّرة وتُعيد النص المُستخرج.
# هذا هو مسار استخراج النص الاحتياطي الذي يستخدمه rag/ingest.py فقط عندما لا
# تحتوي صفحة الـPDF على نص أصلي يمكن قراءته مباشرة.
def transcribe_page_image(png_bytes):
    reader = _get_reader()
    lines = reader.readtext(png_bytes, detail=0, paragraph=True)
    return "\n".join(lines).strip()
