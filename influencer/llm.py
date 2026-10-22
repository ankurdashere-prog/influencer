from __future__ import annotations

import json
import time

import requests

from .config import env


class LLMError(RuntimeError):
    pass


def gemini_json(prompt: str) -> dict:
    """Gemini free-tier se JSON jawab leta hai. Key header mein jaati hai (URL mein nahi)."""
    key = env("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY set nahi hai")
    model = env("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.95},
    }
    last = ""
    for attempt in range(4):
        r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=90)
        if r.status_code in (429, 500, 502, 503):
            last = f"{r.status_code} {r.text[:200]}"
            time.sleep(2 ** attempt * 2)
            continue
        if not r.ok:
            raise LLMError(f"Gemini {r.status_code}: {r.text[:300]}")
        try:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            last = f"galat response: {e}"
            time.sleep(2)
    raise LLMError(f"Gemini fail hua: {last}")
