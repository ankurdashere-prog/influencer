"""Reference reel (link) se sirf STYLE seekhna: hook, pacing, structure. Copy nahi.

Instagram aksar server IPs ko block karta hai, isliye download best-effort hai.
Fail ho to saaf error aata hai (video khud upload karne wala raasta baad mein).
"""
from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from .config import env
from .llm import LLMError, _coerce, _is_gone, model_candidates
from .video import media_duration, run_ff

URL_RE = re.compile(r"^https?://(www\.)?(instagram\.com|youtube\.com|youtu\.be)/\S+$", re.I)

ANALYZE_PROMPT = (
    "Watch this short vertical video and describe its STRUCTURE for a content creator who will make an "
    "ORIGINAL video with a similar style. Return ONE JSON object with keys: "
    "topic (what it is about, 1 line), hook (what happens/said in first 3 seconds and why it grabs attention), "
    "structure (list of beats with approx seconds, e.g. 'problem 3-8s'), pacing (cuts per 10s, energy), "
    "tone (speaking style), cta (how it ends), on_screen_text_style (short), "
    "why_it_works (1-2 lines). Do NOT quote the speaker word-for-word."
)


class ReferenceError(RuntimeError):
    pass


def download(url: str, out_dir: Path) -> Path:
    if not URL_RE.match(url.strip()):
        raise ReferenceError("Sirf Instagram/YouTube ka public link do.")
    tmpl = str(out_dir / "ref.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp", "-f", "mp4/best", "--no-playlist", "--max-filesize", "60M",
           "--socket-timeout", "30", "-o", tmpl, url.strip()]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    files = sorted(out_dir.glob("ref.*"))
    if p.returncode != 0 or not files:
        raise ReferenceError("Link se video download nahi ho paya (Instagram block kar sakta hai ya reel private hai). "
                             f"Detail: {(p.stderr or p.stdout)[-200:].strip()}")
    return files[0]


def shrink(src: Path, dst: Path, max_seconds: int = 60) -> Path:
    if media_duration(src) > max_seconds + 5:
        raise ReferenceError(f"Video {max_seconds}s se lamba hai.")
    run_ff(["-i", str(src), "-t", str(max_seconds), "-vf", "scale=360:-2,fps=12", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "32", "-c:a", "aac", "-b:a", "48k", "-ac", "1", str(dst)])
    if dst.stat().st_size > 14 * 1024 * 1024:
        raise ReferenceError("Video compress hone ke baad bhi bahut bada hai.")
    return dst


def analyze_video(path: Path, session=requests) -> dict:
    key = env("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY set nahi hai")
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(path.read_bytes()).decode()}},
        {"text": ANALYZE_PROMPT}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3}}
    for model in model_candidates():
        r = session.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                         headers={"x-goog-api-key": key}, json=body, timeout=180)
        if not _is_gone(r):
            break
    if not r.ok:
        raise LLMError(f"Gemini video analysis {r.status_code}: {r.text[:300]}")
    try:
        import json

        return _coerce(json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"]))
    except Exception as e:  # noqa: BLE001
        raise LLMError(f"Video analysis ka jawab samajh nahi aaya: {e}") from e


def inspiration_from_url(url: str, say=print) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="ref_"))
    try:
        say("Reference reel download ho rahi hai...")
        raw = download(url, tmp)
        small = shrink(raw, tmp / "small.mp4")
        say("Reel ko samajh raha hu (hook, pacing, structure)...")
        notes = analyze_video(small)
        return "; ".join(f"{k}: {v}" for k, v in notes.items())[:1800]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
