"""Supabase Storage (free tier) par file upload karke public URL deta hai. Graph API ko public URL chahiye."""
from __future__ import annotations

import mimetypes
from pathlib import Path

import requests

from .config import env


class StorageError(RuntimeError):
    pass


def _cfg() -> tuple[str, str, str]:
    url, key, bucket = env("SUPABASE_URL").rstrip("/"), env("SUPABASE_SERVICE_KEY"), env("SUPABASE_BUCKET", "influencer-media")
    if not url or not key:
        raise StorageError("SUPABASE_URL / SUPABASE_SERVICE_KEY set nahi hain")
    return url, key, bucket


def upload_public(path: Path, name: str) -> str:
    url, key, bucket = _cfg()
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        r = requests.post(
            f"{url}/storage/v1/object/{bucket}/{name}",
            headers={"Authorization": f"Bearer {key}", "apikey": key, "Content-Type": ctype, "x-upsert": "true"},
            data=f, timeout=300,
        )
    if not r.ok:
        raise StorageError(f"Supabase upload {r.status_code}: {r.text[:300]}")
    return f"{url}/storage/v1/object/public/{bucket}/{name}"


def delete(name: str) -> None:
    """Free 1 GB bachane ke liye post ke baad file hata deta hai (fail ho to ignore)."""
    try:
        url, key, bucket = _cfg()
        requests.delete(
            f"{url}/storage/v1/object/{bucket}/{name}",
            headers={"Authorization": f"Bearer {key}", "apikey": key}, timeout=60,
        )
    except Exception:
        pass
