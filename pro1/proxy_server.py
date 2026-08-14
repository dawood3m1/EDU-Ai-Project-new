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

import os
import time

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the browser frontend (file:// or localhost) to call this proxy

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_MODEL = "gemini-flash-latest"
MAX_TOKENS = 1000
MAX_ATTEMPTS = 3
BASE_DELAY = 1.2  # seconds, multiplied by attempt number (1.2s, 2.4s, ...)


def is_gemini_model(model):
    return (model or "").lower().startswith("gemini")


def is_openai_model(model):
    """OpenAI-compatible models: gpt-*, o1, o3, deepseek-*, etc."""
    model = (model or "").lower()
    return model.startswith(("gpt-", "o1", "o3", "o4", "deepseek-"))


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


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    model, payload, provider = build_payload(body)

    last_status = None
    last_detail = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(BASE_DELAY * (attempt - 1))
        try:
            resp = do_request(provider, model, payload)
        except requests.RequestException as e:
            last_status = 502
            last_detail = str(e)
            continue

        if resp.status_code == 200:
            text = extract_text(provider, resp.json())
            return jsonify({"text": text})

        # retry only on transient server errors / rate limits
        if resp.status_code in (429, 500, 502, 503, 504):
            last_status = resp.status_code
            last_detail = resp.text[:300]
            continue

        # non-retryable client error: report as-is
        return jsonify({"error": "provider_error", "status": resp.status_code, "detail": resp.text[:500]}), resp.status_code

    return jsonify({"error": "upstream_error", "status": last_status, "detail": last_detail}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
