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


def model_candidates() -> list[str]:
    """GEMINI_MODEL pehle, phir naye default. Google purane model band kar deta hai (404)."""
    out = []
    for m in (env("GEMINI_MODEL"), "gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"):
        if m and m not in out:
            out.append(m)
    return out


def _is_gone(r) -> bool:
    return r.status_code == 404 and "model" in r.text.lower()


def gemini_json(prompt: str) -> dict:
    """Gemini free-tier se JSON jawab leta hai. Key header mein jaati hai (URL mein nahi)."""
    key = env("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY set nahi hai")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.95},
    }
    last = ""
    models = model_candidates()
    mi = 0
    for attempt in range(4 + len(models)):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{models[mi]}:generateContent"
        r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=90)
        if _is_gone(r) and mi + 1 < len(models):
            last = f"{models[mi]} band hai, agla try kar raha hu"
            mi += 1
            continue
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
