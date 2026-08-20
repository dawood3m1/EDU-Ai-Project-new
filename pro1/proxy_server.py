# Backend API | الواجهة الخلفية
# يربط واجهة الطالب مع نظام RAG والـLLM ويعيد الإجابة النهائية.
"""
EDU AI - Secure Proxy Server (Anthropic / OpenAI / Google Gemini)
================================================================
Flask proxy that hides your API key from the browser, fixes CORS, adds
upstream retry + backoff, and passes anti-repetition penalties where the
provider supports them.

Setup:
    pip install flask flask-cors requests

Providers (set the matching key; model prefix decides routing):
    - Google Gemini  (default):  $env:GOOGLE_API_KEY="..."   models like gemini-*
    - Anthropic:                  $env:ANTHROPIC_API_KEY="..."  models like claude-*
    - OpenAI-compatible:          $env:OPENAI_API_KEY="..."     models like gpt-*, o1, o3, deepseek-*

Run (PowerShell):
    python proxy_server.py

The frontend (pro1.html) sends POST requests to:
    http://localhost:5000/api/chat
"""

import json
import os
import sys
import time

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag"))
import store as rag_store  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)  # allow the browser frontend (file:// or localhost) to call this proxy

# Load the embedding model and the vector store into memory once, at startup,
# instead of paying that cost on the first student request.
# تحميل نموذج الـ Embedding والـ vector store مرة واحدة عند تشغيل السيرفر،
# بدل تحميلهما عند أول سؤال من الطالب.
rag_store.warmup()

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_VERSION = "2023-06-01"

# Never log the key values themselves — only whether each is present, so a
# missing-key auth failure (401/403 from the provider) is easy to tell apart
# from an invalid-key one just by reading this line at startup.
# لا نطبع قيمة المفتاح أبدًا، فقط هل هو موجود أم لا، حتى نفرّق بسهولة بين
# خطأ "المفتاح غير موجود" وخطأ "المفتاح غير صالح" من مجرد قراءة الـ log.
print(
    "[proxy] API keys detected: "
    f"GOOGLE_API_KEY={'set' if GOOGLE_API_KEY else 'MISSING'}, "
    f"ANTHROPIC_API_KEY={'set' if ANTHROPIC_API_KEY else 'not set'}, "
    f"OPENAI_API_KEY={'set' if OPENAI_API_KEY else 'not set'}"
)
if not GOOGLE_API_KEY:
    print(
        "[proxy] WARNING: GOOGLE_API_KEY is empty in this process's environment — "
        "every gemini-* request will fail with 401/403. Export it in THIS terminal "
        "before running the server, e.g.: export GOOGLE_API_KEY=\"...\" && python3 proxy_server.py"
    )

DEFAULT_MODEL = "gemini-flash-latest"
MAX_TOKENS = 1000
MAX_ATTEMPTS = 3
BASE_DELAY = 1.2  # seconds, multiplied by attempt number (1.2s, 2.4s, ...)


# Detects which LLM provider a model name belongs to, so the request can be
# translated into that provider's own API format.
# تحدّد هذه الدوال أي مزوّد (provider) ينتمي له اسم الموديل، حتى يُبنى الطلب
# بالصيغة الخاصة بذلك المزوّد.
def is_gemini_model(model):
    return (model or "").lower().startswith("gemini")


def is_openai_model(model):
    """OpenAI-compatible models: gpt-*, o1, o3, deepseek-*, etc."""
    model = (model or "").lower()
    return model.startswith(("gpt-", "o1", "o3", "o4", "deepseek-"))


# Converts the frontend's generic chat request (model/system/messages) into
# the exact JSON shape each provider's API expects (Gemini, OpenAI, or
# Anthropic), so the rest of the backend can stay provider-agnostic.
# تحوّل هذه الدالة طلب الدردشة العام القادم من الواجهة الأمامية (model/system/messages)
# إلى الشكل JSON الذي يتوقعه كل مزوّد تحديدًا (Gemini أو OpenAI أو Anthropic)،
# حتى تبقى بقية الباك إند غير مرتبطة بمزوّد معيّن.
def build_payload(body):
    model = body.get("model", DEFAULT_MODEL)
    if is_gemini_model(model):
        contents = []
        for msg in body.get("messages", []):
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]
        return model, {
            "systemInstruction": {"parts": [{"text": body.get("system", "")}]},
            "contents": contents,
            "generationConfig": {"temperature": float(body.get("temperature", 0.7))},
        }, "gemini"
    if is_openai_model(model):
        messages = [{"role": "system", "content": body.get("system", "")}]
        messages += body.get("messages", [])
        return model, {
            "model": model,
            "messages": messages,
            "max_tokens": int(body.get("max_tokens", MAX_TOKENS)),
            "temperature": float(body.get("temperature", 0.7)),
            "frequency_penalty": float(body.get("frequency_penalty", 0.7)),
            "presence_penalty": float(body.get("presence_penalty", 0.7)),
        }, "openai"
    # Anthropic path: build an explicit payload so unsupported params never leak upstream
    return model, {
        "model": model,
        "max_tokens": int(body.get("max_tokens", MAX_TOKENS)),
        "system": body.get("system", ""),
        "messages": body.get("messages", []),
        "temperature": float(body.get("temperature", 0.7)),
    }, "anthropic"


# Sends the actual HTTPS request to the chosen LLM provider (this is the
# only place the real API keys are attached to a request) and returns the
# raw HTTP response, keeping the API key hidden from the browser.
# ترسل هذه الدالة الطلب الفعلي عبر HTTPS إلى مزوّد الـLLM المختار (وهي المكان
# الوحيد الذي تُرفق فيه مفاتيح الـAPI الحقيقية بالطلب)، وتعيد الاستجابة الخام،
# بحيث يبقى المفتاح مخفيًا تمامًا عن المتصفح.
def do_request(provider, model, payload):
    if provider == "gemini":
        return requests.post(
            GEMINI_BASE.format(model=model),
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GOOGLE_API_KEY,
            },
            timeout=30,
        )
    if provider == "openai":
        return requests.post(
            OPENAI_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            timeout=30,
        )
    return requests.post(
        ANTHROPIC_URL,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        timeout=30,
    )


# Pulls the plain generated text out of each provider's differently-shaped
# JSON response, so the rest of the backend always works with a single string
# regardless of which LLM answered.
# تستخرج هذه الدالة النص المُولَّد فقط من استجابة كل مزوّد (شكل JSON مختلف
# لكل مزوّد)، حتى تتعامل بقية الباك إند مع نص واحد بسيط بغض النظر عن أي LLM أجاب.
def extract_text(provider, data):
    if provider == "gemini":
        text = ""
        for cand in data.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                text += part.get("text", "")
        return text
    if provider == "openai":
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )


# Serves the student/faculty frontend itself, so the page and the API share
# the same origin (http://127.0.0.1:5000) instead of being opened as a
# file:// URL — this is the entry point of frontend↔backend communication.
# تُقدّم هذه الدالة واجهة الطالب/الدكتور نفسها، بحيث تكون الصفحة والـAPI على
# نفس الـorigin (http://127.0.0.1:5000) بدل فتح الملف مباشرة كـ file://،
# وهذه هي نقطة البداية للتواصل بين الواجهة الأمامية والباك إند.
@app.route("/", methods=["GET"])
def index():
    """Serve the frontend so the page and the API share the same origin
    (http://127.0.0.1:5000) instead of being opened as a file:// URL."""
    return send_from_directory(APP_DIR, "pro1.html")


# Detects when the LLM's raw reply is itself a JSON object (the tutor/review
# contract used for phase/step tracking) and parses it, so /api/chat can
# return one natural JSON object instead of a JSON string nested inside JSON.
# تكتشف هذه الدالة عندما يكون رد الـLLM الخام كائن JSON بحد ذاته (عقد
# tutor/review المستخدم لتتبّع المرحلة والخطوة) وتحلّله، حتى يُرجع /api/chat
# كائن JSON طبيعيًا واحدًا بدل نص JSON متداخل داخل JSON آخر.
def _try_flatten_json_message(text):
    """If the model's raw text is itself a JSON object (the tutor/review JSON
    contract used internally for phase/step tracking), parse it and return the
    dict so the HTTP response is one natural JSON object instead of a JSON
    string double-encoded inside another JSON response. Returns None for
    plain-prose replies (e.g. the academic scenario), which are left as-is."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        return None
    try:
        parsed = json.loads(cleaned)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
        return parsed
    return None


# Main LLM integration endpoint: receives the question/system-prompt (which
# already includes any retrieved RAG chunks, attached by the frontend) from
# pro1.html, calls the configured LLM provider with retries on transient
# errors, and returns the final generated answer to the frontend.
# نقطة تكامل الـLLM الرئيسية: تستقبل هذه الدالة السؤال والـsystem prompt
# (والذي يتضمن مسبقًا أي مقاطع RAG مسترجعة، أرفقتها الواجهة الأمامية) من
# pro1.html، وتستدعي مزوّد الـLLM المُعدّ مع إعادة محاولة عند الأخطاء المؤقتة،
# ثم تُعيد الإجابة النهائية المُولَّدة إلى الواجهة الأمامية.
@app.route("/api/chat", methods=["POST"])
def chat():
    t_start = time.time()
    body = request.get_json(silent=True) or {}
    model, payload, provider = build_payload(body)
    print(f"[chat] request received: provider={provider} model={model} messages={len(body.get('messages') or [])}")

    last_status = None
    last_detail = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(BASE_DELAY * (attempt - 1))
        t_llm = time.time()
        try:
            resp = do_request(provider, model, payload)
        except requests.RequestException as e:
            last_status = 502
            last_detail = str(e)
            print(f"[chat] attempt {attempt}: request exception after {time.time() - t_llm:.2f}s: {e}")
            continue

        llm_ms = (time.time() - t_llm) * 1000
        print(f"[chat] attempt {attempt}: LLM call took {llm_ms:.0f}ms, upstream status={resp.status_code}")
        if resp.status_code == 200:
            text = extract_text(provider, resp.json())
            flattened = _try_flatten_json_message(text)
            print(f"[chat] total request time: {(time.time() - t_start) * 1000:.0f}ms")
            if flattened is not None:
                return jsonify(flattened)
            return jsonify({"text": text})

        # retry only on transient server errors / rate limits
        if resp.status_code in (429, 500, 502, 503, 504):
            last_status = resp.status_code
            last_detail = resp.text[:300]
            continue

        # non-retryable client error: report as-is
        print(f"[chat] non-retryable error {resp.status_code}: {resp.text[:300]}")
        print(f"[chat] total request time: {(time.time() - t_start) * 1000:.0f}ms")
        return jsonify({"error": "provider_error", "status": resp.status_code, "detail": resp.text[:500]}), resp.status_code

    print(f"[chat] giving up after {MAX_ATTEMPTS} attempts: last_status={last_status} last_detail={last_detail[:200]}")
    print(f"[chat] total request time: {(time.time() - t_start) * 1000:.0f}ms")
    return jsonify({"error": "upstream_error", "status": last_status, "detail": last_detail}), 502


# RAG retrieval endpoint: receives the student's question, delegates to the
# vector store (rag/store.py) to embed the question and find the most
# relevant course-content chunks, and returns only those chunks — it never
# calls the LLM itself, keeping retrieval and generation as separate steps.
# نقطة استرجاع RAG: تستقبل سؤال الطالب، وتُفوّض إلى الـvector store
# (rag/store.py) لعمل embedding للسؤال وإيجاد أكثر مقاطع المحتوى الدراسي
# ارتباطًا به، وتُعيد هذه المقاطع فقط — لا تستدعي الـLLM إطلاقًا، بحيث يبقى
# الاسترجاع والتوليد خطوتين منفصلتين.
@app.route("/api/rag/query", methods=["POST"])
def rag_query():
    """Retrieval only — returns the most relevant knowledge-base chunks for a
    question. Does not call the chat model; the frontend attaches the
    returned chunks to its own prompt before calling /api/chat as usual."""
    t_start = time.time()
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    print(f"[rag] request received: question={question!r}")
    if not question:
        return jsonify({"chunks": []})
    top_k = int(body.get("top_k", 4))
    try:
        chunks = rag_store.search(question, top_k=top_k)  # logs its own embedding/search timing
    except Exception as e:
        print(f"[rag] search failed: {e}")
        return jsonify({"error": "rag_error", "detail": str(e)}), 502
    print(f"[rag] returned {len(chunks)} chunk(s)" + (f", top score={chunks[0]['score']:.3f}" if chunks else " (store is empty or nothing matched)"))
    print(f"[rag] total request time: {(time.time() - t_start) * 1000:.0f}ms")
    return jsonify({"chunks": chunks})


# Starts the Flask development server that hosts both the frontend and the
# API endpoints above.
# يشغّل خادم التطوير الخاص بـ Flask الذي يستضيف الواجهة الأمامية ونقاط الـAPI أعلاه.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
