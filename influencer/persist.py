"""Hugging Face Space ki disk temporary hoti hai. Isliye character, settings aur history
Supabase Storage mein (state/ prefix) sync hoti hain. Supabase set na ho to ye sab no-op hai."""
from __future__ import annotations

from pathlib import Path

import requests

from .config import ROOT, env

SEP = "__"


def _cfg() -> tuple[str, str, str] | None:
    url, key = env("SUPABASE_URL").rstrip("/"), env("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return url, key, env("SUPABASE_BUCKET", "influencer-media")


def _key(path: Path) -> str:
    return "state/" + path.resolve().relative_to(ROOT).as_posix().replace("/", SEP)


def enabled() -> bool:
    return _cfg() is not None


def push(path: Path) -> None:
    c = _cfg()
    if not c or not path.exists():
        return
    url, key, bucket = c
    try:
        requests.post(
            f"{url}/storage/v1/object/{bucket}/{_key(path)}",
            headers={"Authorization": f"Bearer {key}", "apikey": key,
                     "Content-Type": "application/octet-stream", "x-upsert": "true"},
            data=path.read_bytes(), timeout=60,
        )
    except requests.RequestException as e:
        print(f"[persist] push fail ({path.name}): {e}")


def remove(path: Path) -> None:
    c = _cfg()
    if not c:
        return
    url, key, bucket = c
    try:
        requests.delete(f"{url}/storage/v1/object/{bucket}/{_key(path)}",
                        headers={"Authorization": f"Bearer {key}", "apikey": key}, timeout=60)
    except requests.RequestException:
        pass


def pull() -> int:
    """Startup par saved state wapas laata hai. Kitni files aayi wo return karta hai."""
    c = _cfg()
    if not c:
        return 0
    url, key, bucket = c
    h = {"Authorization": f"Bearer {key}", "apikey": key}
    n = 0
    try:
        r = requests.post(f"{url}/storage/v1/object/list/{bucket}", headers=h,
                          json={"prefix": "state", "limit": 1000, "offset": 0}, timeout=60)
        if not r.ok:
            return 0
        for item in r.json():
            name = item.get("name", "")
            if not name or SEP not in name and "." not in name:
                continue
            rel = name.replace(SEP, "/")
            dest = (ROOT / rel).resolve()
            if ROOT not in dest.parents:  # path traversal se bachav
                continue
            g = requests.get(f"{url}/storage/v1/object/{bucket}/state/{name}", headers=h, timeout=120)
            if g.ok:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(g.content)
                n += 1
    except requests.RequestException as e:
        print(f"[persist] pull fail: {e}")
    return n
