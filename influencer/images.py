"""Image generation: openai (reference se same face), pollinations (free), placeholder (test)."""
from __future__ import annotations

import base64
import time
import urllib.parse
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from .config import REFERENCE_IMAGE, env


class ImageError(RuntimeError):
    pass


def full_prompt(persona: dict, scene: str) -> str:
    look = " ".join(str(persona.get("look_prompt", "")).split())
    return (
        f"{look}. Scene: {scene}. Vertical 9:16 composition, subject centered with some headroom, "
        "same person as in the reference image, natural phone-camera realism, no text, no watermark."
    )


def _save_bytes(data: bytes, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)


def _openai(prompt: str, out: Path, ref: Path | None) -> None:
    key = env("OPENAI_API_KEY")
    if not key:
        raise ImageError("OPENAI_API_KEY set nahi hai")
    model = env("OPENAI_IMAGE_MODEL", "gpt-image-1")
    quality = env("OPENAI_IMAGE_QUALITY", "low")
    headers = {"Authorization": f"Bearer {key}"}
    last = ""
    for attempt in range(3):
        if ref and ref.exists():
            with open(ref, "rb") as f:
                r = requests.post(
                    "https://api.openai.com/v1/images/edits", headers=headers,
                    files={"image": (ref.name, f, "image/png")},
                    data={"model": model, "prompt": prompt, "size": "1024x1536",
                          "quality": quality, "n": "1"},
                    timeout=240,
                )
        else:
            r = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={**headers, "Content-Type": "application/json"},
                json={"model": model, "prompt": prompt, "size": "1024x1536",
                      "quality": quality, "n": 1},
                timeout=240,
            )
        if r.status_code in (429, 500, 502, 503):
            last = f"{r.status_code} {r.text[:200]}"
            time.sleep(4 * (attempt + 1))
            continue
        if not r.ok:
            raise ImageError(f"OpenAI image {r.status_code}: {r.text[:400]}")
        item = r.json()["data"][0]
        if item.get("b64_json"):
            _save_bytes(base64.b64decode(item["b64_json"]), out)
        elif item.get("url"):
            _save_bytes(requests.get(item["url"], timeout=120).content, out)
        else:
            raise ImageError("OpenAI response mein image nahi mili")
        return
    raise ImageError(f"OpenAI image fail: {last}")


def _pollinations(prompt: str, out: Path, seed: int) -> None:
    url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:900])
    last = ""
    for attempt in range(3):
        r = requests.get(
            url, params={"width": 1024, "height": 1536, "model": "flux", "nologo": "true", "seed": seed},
            timeout=180,
        )
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            _save_bytes(r.content, out)
            return
        last = f"{r.status_code}"
        time.sleep(5 * (attempt + 1))
    raise ImageError(f"Pollinations fail: {last}")


def _placeholder(prompt: str, out: Path, seed: int) -> None:
    w, h = 1024, 1536
    img = Image.new("RGB", (w, h))
    px = img.load()
    c1 = ((seed * 53) % 120 + 60, (seed * 97) % 120 + 40, (seed * 31) % 120 + 90)
    c2 = (230 - c1[0] // 2, 200 - c1[1] // 3, 230 - c1[2] // 3)
    for y in range(h):
        t = y / h
        row = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = row
    d = ImageDraw.Draw(img)
    d.ellipse((w // 2 - 170, 420, w // 2 + 170, 820), fill=(245, 215, 190))
    d.rectangle((w // 2 - 260, 830, w // 2 + 260, 1400), fill=(250, 250, 250))
    d.text((40, 40), f"DEMO scene {seed}\n{prompt[:120]}", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def generate(prompt: str, out: Path, seed: int = 1, ref: Path | None = None,
             provider: str | None = None) -> Path:
    provider = (provider or env("IMAGE_PROVIDER", "openai")).lower()
    if provider == "openai":
        _openai(prompt, out, ref if ref is not None else (REFERENCE_IMAGE if REFERENCE_IMAGE.exists() else None))
    elif provider == "pollinations":
        _pollinations(prompt, out, seed)
    elif provider == "placeholder":
        _placeholder(prompt, out, seed)
    else:
        raise ImageError(f"Unknown IMAGE_PROVIDER: {provider}")
    return out


def to_feed_ratio(src: Path, dst: Path) -> Path:
    """Instagram feed photo ka ratio 4:5 se 1.91:1 ke beech hona chahiye. Center crop 1080x1350 (4:5)."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    target = 4 / 5
    if w / h > target:
        nw = int(h * target)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / target)
        top = max(0, int((h - nh) * 0.35))
        img = img.crop((0, top, w, top + nh))
    img = img.resize((1080, 1350), Image.LANCZOS)
    img.save(dst, "JPEG", quality=92)
    return dst
