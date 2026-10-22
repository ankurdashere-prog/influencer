"""Bolta hua (lip-sync) influencer: face image + voice -> talking video, hosted API (fal.ai) se.

Render free par GPU nahi hai, isliye model kisi aur ke server par chalta hai (paid, per-reel).
Kuch bhi fail ho to reel slideshow mode mein ban jati hai (kabhi khaali haath nahi).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path

import requests

from .config import env
from .tts import Word
from .video import run_ff

QUEUE = "https://queue.fal.run"
DEFAULT_MODEL = "fal-ai/kling-video/v1/standard/ai-avatar"  # FAL_MODEL se badal sakte ho


class TalkingError(RuntimeError):
    pass


def enabled() -> bool:
    return env("TALKING_PROVIDER", "none").lower() == "fal"


def pick_cut(words: list[Word], total: float, want: float) -> float:
    """Audio ko kitni der tak talking-head banana hai: shabd ki seema par, 3s se kam nahi."""
    want = min(max(want, 3.0), total)
    ends = [w.end for w in words if w.end <= want + 0.05]
    cut = max(ends) if ends else want
    return round(min(max(cut, min(3.0, total)), total), 2)


def slice_audio(audio: Path, out: Path, seconds: float) -> Path:
    run_ff(["-i", str(audio), "-t", f"{seconds:.2f}", "-c:a", "libmp3lame", "-b:a", "128k", str(out)])
    return out


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _find_video_url(obj) -> str | None:
    if isinstance(obj, dict):
        v = obj.get("video")
        if isinstance(v, dict) and isinstance(v.get("url"), str):
            return v["url"]
        if isinstance(v, str) and v.startswith("http"):
            return v
        for val in obj.values():
            r = _find_video_url(val)
            if r:
                return r
    elif isinstance(obj, list):
        for val in obj:
            r = _find_video_url(val)
            if r:
                return r
    return None


def generate(image: Path, audio: Path, out: Path, prompt: str = "", timeout: int = 900,
             poll: float = 5.0, session=requests) -> Path:
    key = env("FAL_KEY")
    if not key:
        raise TalkingError("FAL_KEY set nahi hai (Render Environment mein daalo).")
    model = env("FAL_MODEL", DEFAULT_MODEL).strip("/")
    hdr = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    body = {"image_url": _data_uri(image), "audio_url": _data_uri(audio)}
    if prompt:
        body["prompt"] = prompt
    extra = env("FAL_EXTRA_JSON")
    if extra:
        try:
            body.update(json.loads(extra))
        except json.JSONDecodeError as e:
            raise TalkingError(f"FAL_EXTRA_JSON sahi JSON nahi hai: {e}") from e

    r = session.post(f"{QUEUE}/{model}", headers=hdr, json=body, timeout=120)
    if r.status_code >= 400:
        raise TalkingError(f"fal submit fail {r.status_code}: {r.text[:300]}")
    job = r.json()
    status_url, result_url = job.get("status_url"), job.get("response_url")
    if not status_url or not result_url:
        raise TalkingError(f"fal ka jawab samajh nahi aaya: {str(job)[:200]}")

    end = time.time() + timeout
    while True:
        s = session.get(status_url, headers=hdr, timeout=60)
        if s.status_code >= 400:
            raise TalkingError(f"fal status fail {s.status_code}: {s.text[:200]}")
        st = (s.json().get("status") or "").upper()
        if st == "COMPLETED":
            break
        if st in ("FAILED", "ERROR", "CANCELLED"):
            raise TalkingError(f"fal job {st}: {s.text[:200]}")
        if time.time() > end:
            raise TalkingError("fal job time out (15 min).")
        time.sleep(poll)

    res = session.get(result_url, headers=hdr, timeout=60)
    if res.status_code >= 400:
        raise TalkingError(f"fal result fail {res.status_code}: {res.text[:300]}")
    url = _find_video_url(res.json())
    if not url:
        raise TalkingError(f"Result mein video URL nahi mila: {res.text[:200]}")
    data = session.get(url, timeout=180)
    if data.status_code >= 400 or not data.content:
        raise TalkingError(f"Video download fail {data.status_code}")
    out.write_bytes(data.content)
    return out
