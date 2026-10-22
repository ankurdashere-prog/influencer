from __future__ import annotations

import json
import time

import requests

from .config import env


class LLMError(RuntimeError):
    pass


def _coerce(data):
    """Gemini kabhi JSON ko string mein lapet deta hai (double-encoded) ya list mein. Dict nikalo."""
    for _ in range(4):
        if isinstance(data, str):
            txt = data.strip()
            if txt.startswith("```"):
                txt = txt.strip("`").removeprefix("json").strip()
            try:
                data = json.loads(txt)
                continue
            except json.JSONDecodeError:
                break
        if isinstance(data, list):
            data = next((d for d in data if isinstance(d, (dict, str))), None)
            continue
        break
    if not isinstance(data, dict):
        raise LLMError(f"Gemini ne dict nahi diya ({type(data).__name__})")
    return data


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
            return _coerce(json.loads(text))
        except (KeyError, IndexError, json.JSONDecodeError, LLMError) as e:
            last = f"galat response: {e}"
            time.sleep(2)
    raise LLMError(f"Gemini fail hua: {last}")
