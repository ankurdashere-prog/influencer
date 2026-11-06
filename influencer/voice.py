"""Voice clone: ElevenLabs API (Instant Voice Clone) se client ki awaaz banao aur script bulwao.
Sirf us insaan ki likhit permission ke saath. Voice IDs state/voices.json mein rehti hain (Supabase se sync)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from . import persist
from .config import ROOT, env, now_ist

VOICES_FILE = ROOT / "state" / "voices.json"
API = "https://api.elevenlabs.io"
MODEL = "eleven_multilingual_v2"
STS_MODEL = "eleven_multilingual_sts_v2"  # voice changer: timing wahi rehti hai, awaaz clone ki
MAX_SAMPLE_BYTES = 10 * 1024 * 1024
MAX_TEXT = 1500


class VoiceError(RuntimeError):
    pass


def enabled() -> bool:
    return bool(env("ELEVENLABS_API_KEY"))


def _headers() -> dict:
    key = env("ELEVENLABS_API_KEY")
    if not key:
        raise VoiceError("ELEVENLABS_API_KEY set nahi hai (Render > Environment mein daalo).")
    return {"xi-api-key": key}


def _fail(r: requests.Response, what: str) -> VoiceError:
    try:
        d = r.json().get("detail")
        msg = d.get("message") if isinstance(d, dict) else str(d)
    except Exception:  # noqa: BLE001
        msg = r.text[:200]
    return VoiceError(f"ElevenLabs {what} fail ({r.status_code}): {msg}")


_cache: dict = {"t": 0.0, "v": None}


def _local_voices() -> list[dict]:
    if VOICES_FILE.exists():
        try:
            return [v for v in json.loads(VOICES_FILE.read_text(encoding="utf-8")) if isinstance(v, dict)]
        except json.JSONDecodeError:
            return []
    return []


def _remote_voices() -> list[dict]:
    """ElevenLabs account ki apni clone ki hui voices (restart ke baad bhi wahi rehti hain)."""
    r = requests.get(f"{API}/v2/voices", headers=_headers(), timeout=20,
                     params={"category": "cloned", "voice_type": "personal", "page_size": 100})
    if r.status_code != 200:
        raise _fail(r, "voice list")
    out = []
    for v in r.json().get("voices") or []:
        if isinstance(v, dict) and v.get("voice_id"):
            name = str(v.get("name") or v["voice_id"]).replace(" (client voice)", "").strip()[:60]
            out.append({"voice_id": v["voice_id"], "name": name, "provider": "elevenlabs", "source": "elevenlabs"})
    return out


def load_voices(force: bool = False) -> list[dict]:
    """ElevenLabs se list (30 sec cache). Wahan na pahunch paye to local file ki list."""
    if enabled():
        if not force and _cache["v"] is not None and time.time() - _cache["t"] < 30:
            return _cache["v"]
        try:
            _cache["v"], _cache["t"] = _remote_voices(), time.time()
            return _cache["v"]
        except Exception as e:  # noqa: BLE001 - network/key dikkat: local list par chalo
            print(f"[voice] ElevenLabs list nahi mili: {str(e)[:150]}")
    return _local_voices()


def _save(items: list[dict]) -> None:
    VOICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    VOICES_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    persist.push(VOICES_FILE)


def clean_name(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "")).strip()[:60]
    if not n:
        raise VoiceError("Voice ka naam likho (jaise: Dr. Ashish).")
    return n


def clone(name: str, samples: list[tuple[str, bytes, str]], consent: bool) -> dict:
    """samples = [(filename, bytes, mimetype)]. Consent True hona zaroori hai."""
    if consent is not True:
        raise VoiceError("Voice clone ke liye consent confirm karo.")
    name = clean_name(name)
    if not samples:
        raise VoiceError("Kam se kam ek audio sample chahiye.")
    if sum(len(b) for _, b, _ in samples) > MAX_SAMPLE_BYTES * 2 or any(len(b) > MAX_SAMPLE_BYTES for _, b, _ in samples):
        raise VoiceError("Sample bahut bada hai (har file 10MB se kam rakho).")
    r = requests.post(f"{API}/v1/voices/add", headers=_headers(), data={"name": f"{name} (client voice)"},
                      files=[("files", s) for s in samples], timeout=120)
    if r.status_code != 200:
        raise _fail(r, "clone")
    vid = r.json().get("voice_id")
    if not vid:
        raise VoiceError("ElevenLabs ne voice_id nahi diya.")
    items = [v for v in _local_voices() if v.get("voice_id") != vid]
    rec = {"voice_id": vid, "name": name, "provider": "elevenlabs", "consent": True, "created": now_ist().isoformat()}
    items.append(rec)
    _save(items)
    _cache["v"] = None
    return rec


def speak(text: str, voice_id: str, out: Path) -> Path:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        raise VoiceError("Script khaali hai.")
    if len(text) > MAX_TEXT:
        raise VoiceError(f"Script {MAX_TEXT} characters se chhoti rakho (credits bachane ke liye).")
    if voice_id not in {v.get("voice_id") for v in load_voices()}:
        raise VoiceError("Ye voice list mein nahi hai. Pehle clone banao.")
    r = requests.post(f"{API}/v1/text-to-speech/{voice_id}", headers={**_headers(), "Accept": "audio/mpeg"},
                      params={"output_format": "mp3_44100_128"},
                      json={"text": text, "model_id": MODEL}, timeout=120)
    if r.status_code != 200:
        raise _fail(r, "awaaz")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


def delete(voice_id: str) -> None:
    items = _local_voices()
    if voice_id not in {v.get("voice_id") for v in load_voices(force=True)} | {v.get("voice_id") for v in items}:
        raise VoiceError("Voice nahi mili.")
    if enabled():
        r = requests.delete(f"{API}/v1/voices/{voice_id}", headers=_headers(), timeout=60)
        if r.status_code not in (200, 404):
            raise _fail(r, "delete")
    _save([v for v in items if v.get("voice_id") != voice_id])
    _cache["v"] = None


def convert(audio: Path, voice_id: str, out: Path) -> Path:
    """Kisi bhi bolti hui audio (jaise Flow clip ki awaaz) ko clone voice mein badalta hai. Timing wahi rehti hai,
    isliye lip-sync bacha rehta hai. Max 5 minute / 50 MB per request."""
    if voice_id not in {v.get("voice_id") for v in load_voices()}:
        raise VoiceError("Ye voice list mein nahi hai. Pehle clone banao.")
    with open(audio, "rb") as f:
        r = requests.post(f"{API}/v1/speech-to-speech/{voice_id}", headers={**_headers(), "Accept": "audio/mpeg"},
                          params={"output_format": "mp3_44100_128"},
                          data={"model_id": STS_MODEL}, files={"audio": (audio.name, f, "audio/mpeg")}, timeout=300)
    if r.status_code != 200:
        raise _fail(r, "voice badalna")
    out.write_bytes(r.content)
    return out
