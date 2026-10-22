"""Voiceover + word timings. edge-tts (free) ya silent (test)."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from .config import env
from .video import media_duration, run_ff


@dataclass
class Word:
    text: str
    start: float
    end: float


def estimate_words(text: str, duration: float) -> list[Word]:
    """Word timings ka andaza: audio duration ko word ki lambai ke hisaab se baantta hai."""
    tokens = text.split()
    if not tokens:
        return []
    weights = [len(re.sub(r"\W", "", t)) + 1 + (3 if t[-1] in ".?!," else 0) for t in tokens]
    total = sum(weights)
    out, t = [], 0.0
    for tok, wgt in zip(tokens, weights):
        d = duration * wgt / total
        out.append(Word(tok, t, t + d))
        t += d
    return out


async def _edge(text: str, voice: str, rate: str, out: Path) -> list[Word]:
    import edge_tts

    try:
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    except TypeError:  # purane version mein boundary argument nahi hota
        comm = edge_tts.Communicate(text, voice, rate=rate)
    words: list[Word] = []
    with open(out, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                s = chunk["offset"] / 1e7
                words.append(Word(chunk["text"], s, s + chunk["duration"] / 1e7))
    return words


def synthesize(text: str, out: Path, provider: str | None = None) -> tuple[Path, list[Word], float]:
    provider = (provider or env("TTS_PROVIDER", "edge")).lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    if provider == "silent":
        dur = max(6.0, len(text.split()) / 2.6)
        run_ff(["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", f"{dur:.2f}", "-c:a", "libmp3lame", str(out)])
        return out, estimate_words(text, dur), dur
    if provider not in ("edge", "gtts"):
        raise RuntimeError(f"Unknown TTS_PROVIDER: {provider}")
    voice = env("TTS_VOICE", "en-IN-NeerjaNeural")
    words: list[Word] = []
    last = None
    if provider == "edge":
        for _ in range(3):
            try:
                words = asyncio.run(_edge(text, voice, "+4%", out))
                break
            except Exception as e:  # network / service errors
                last = e
        else:
            print(f"[tts] edge-tts nahi chala ({last}). Google TTS fallback use kar raha hu.")
            provider = "gtts"
    if provider == "gtts":
        try:
            from gtts import gTTS

            gTTS(text=text, lang="en", tld="co.in").save(str(out))
        except Exception as e:
            raise RuntimeError(f"Voice nahi ban payi (edge-tts aur gTTS dono fail): {e}") from e
    dur = media_duration(out)
    if not words or words[-1].end < dur * 0.5:
        words = estimate_words(text, dur)
    return out, words, dur
