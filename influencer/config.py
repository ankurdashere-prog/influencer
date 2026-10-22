from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

IST = timezone(timedelta(hours=5, minutes=30))
OUTPUT_DIR = ROOT / "output"
STATE_FILE = ROOT / "state" / "history.json"
REFERENCE_IMAGE = ROOT / "character" / "reference.png"


def env(key: str, default: str = "") -> str:
    # GitHub Actions khaali variables ko "" bana deta hai, isliye `or default` zaroori hai
    return (os.getenv(key) or default).strip()


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def persona() -> dict:
    return load_yaml(ROOT / "character" / "persona.yaml")


def settings() -> dict:
    return load_yaml(ROOT / "settings.yaml")


def now_ist() -> datetime:
    return datetime.now(IST)


def load_history() -> list[dict]:
    if STATE_FILE.exists():
        try:
            items = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("items", [])
            return [i for i in items if isinstance(i, dict)]
        except json.JSONDecodeError:
            return []
    return []


def _save_history(items: list[dict]) -> None:
    from . import persist

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"items": items[-400:]}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    persist.push(STATE_FILE)


def add_history(item: dict) -> None:
    items = load_history()
    items.append(item)
    _save_history(items)


def update_history(dirname: str, **fields) -> None:
    items = load_history()
    for it in items:
        if it.get("dir") == dirname:
            it.update(fields)
    _save_history(items)
