"""Instagram publishing via official Meta Graph API (Reels + photo)."""
from __future__ import annotations

import time

import requests

from .config import env


class PublishError(RuntimeError):
    pass


def _base() -> str:
    return f"https://graph.facebook.com/{env('GRAPH_VERSION', 'v23.0')}"


def _creds() -> tuple[str, str]:
    uid, tok = env("IG_USER_ID"), env("IG_ACCESS_TOKEN")
    if not uid or not tok:
        raise PublishError("IG_USER_ID / IG_ACCESS_TOKEN set nahi hain")
    return uid, tok


def _check(r: requests.Response, what: str) -> dict:
    try:
        data = r.json()
    except ValueError:
        data = {}
    if not r.ok or "error" in data:
        err = data.get("error", {})
        raise PublishError(f"{what} fail: {err.get('message') or r.text[:300]} (code {err.get('code')})")
    return data


def _wait_ready(container_id: str, token: str, timeout: int = 420) -> None:
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{_base()}/{container_id}", params={"fields": "status_code,status", "access_token": token}, timeout=60)
        data = _check(r, "status check")
        code = data.get("status_code")
        if code == "FINISHED":
            return
        if code in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram processing {code}: {data.get('status')}")
        time.sleep(10)
    raise PublishError("Instagram processing timeout (7 min)")


def _publish_container(uid: str, token: str, container_id: str) -> str:
    r = requests.post(f"{_base()}/{uid}/media_publish", data={"creation_id": container_id, "access_token": token}, timeout=120)
    return _check(r, "media_publish")["id"]


def publish_reel(video_url: str, caption: str) -> str:
    uid, token = _creds()
    r = requests.post(
        f"{_base()}/{uid}/media",
        data={"media_type": "REELS", "video_url": video_url, "caption": caption,
              "share_to_feed": "true", "access_token": token},
        timeout=120,
    )
    cid = _check(r, "reel container")["id"]
    _wait_ready(cid, token)
    return _publish_container(uid, token, cid)


def publish_photo(image_url: str, caption: str) -> str:
    uid, token = _creds()
    r = requests.post(
        f"{_base()}/{uid}/media",
        data={"image_url": image_url, "caption": caption, "access_token": token},
        timeout=120,
    )
    cid = _check(r, "photo container")["id"]
    _wait_ready(cid, token, timeout=120)
    return _publish_container(uid, token, cid)
