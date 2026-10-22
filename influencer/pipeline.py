"""End-to-end: plan -> images -> voice -> captions -> video -> (post)."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import captions, ideas, images, publish, storage, tts
from .config import (OUTPUT_DIR, REFERENCE_IMAGE, ROOT, add_history, env, load_history,
                     load_yaml, now_ist, persona, settings, update_history)
from .video import assemble_reel

# Web UI isko badal deta hai taaki progress browser mein dikhe
progress = print


def _say(msg: str) -> None:
    progress(msg)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "post")[:40]


def load_brand(path: str | Path | None) -> dict | None:
    if path:
        return load_yaml(Path(path))
    return None


def next_brand() -> Path | None:
    """brands/ mein active briefs ko history ke hisaab se baari-baari se chunta hai."""
    files = sorted((ROOT / "brands").glob("*.yaml"))
    active = [f for f in files if load_yaml(f).get("active", True)]
    if not active:
        return None
    used = [h.get("brand") for h in load_history() if h.get("brand")]
    for f in active:
        if f.stem not in used[-len(active):]:
            return f
    return active[0]


def run(kind: str, brand_path: str | Path | None = None, post: bool = False, demo: bool = False,
        topic: str | None = None, source: str = "manual") -> dict:
    cfg, who = settings(), persona()
    brand = load_brand(brand_path) if kind == "ad" else None
    if kind == "ad" and not brand:
        raise ValueError("Ad ke liye brand brief chuno (brands/ mein active brief rakho).")

    stamp = now_ist().strftime("%Y-%m-%d_%H%M%S")
    _say("Idea aur script bana raha hu...")
    plan = ideas.make_plan(kind, who, cfg, load_history(), brand, topic, offline=demo or not env("GEMINI_API_KEY"))
    work = OUTPUT_DIR / f"{stamp}_{kind}_{_slug(plan['topic'])}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    if plan.get("fallback_reason"):
        _say(f"Gemini ka jawab use nahi ho paya ({plan['fallback_reason'][:160]}). Simple template use kiya.")
    _say(f"Topic: {plan['topic']}")

    provider = "placeholder" if demo else None
    ref = REFERENCE_IMAGE if REFERENCE_IMAGE.exists() else None
    if not demo and env("IMAGE_PROVIDER", "openai") == "openai" and not ref:
        _say("Warning: character reference nahi hai, face har baar badal sakta hai (Character tab dekho).")

    imgs: list[Path] = []
    base_seed = random.randint(1, 10**6)
    for i, scene in enumerate(plan["scenes"]):
        p = work / f"scene_{i + 1}.png"
        images.generate(images.full_prompt(who, scene), p, seed=base_seed + i, ref=ref, provider=provider)
        imgs.append(p)
        _say(f"Image {i + 1}/{len(plan['scenes'])} ban gayi")

    (work / "caption.txt").write_text(plan["caption"], encoding="utf-8")

    if kind == "photo":
        final = images.to_feed_ratio(imgs[0], work / "post.jpg")
    else:
        _say("Voiceover bana raha hu...")
        audio, words, dur = tts.synthesize(plan["voiceover"], work / "voice.mp3", "silent" if demo else None)
        ass = captions.build_ass(words, work / "captions.ass", cfg["video"].get("caption", {}),
                                 cfg["video"]["width"], cfg["video"]["height"])
        music_dir = ROOT / "assets" / "music"
        tracks = [p for p in music_dir.glob("*") if p.suffix.lower() in (".mp3", ".m4a", ".wav")] if music_dir.exists() else []
        music = random.choice(tracks) if tracks else None
        _say("Video render ho rahi hai (1-3 minute)...")
        final = assemble_reel(imgs, audio, ass, work / "reel.mp4", work, dur, cfg["video"], music)
    _say("Video/photo tayyar.")

    result = {"dir": work.name, "kind": kind, "topic": plan["topic"], "file": final.name,
              "created": now_ist().isoformat(timespec="seconds"), "brand": Path(brand_path).stem if brand_path else None,
              "source": source, "demo": demo, "posted": False, "ig_media_id": None}
    (work / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    add_history({"date": result["created"], "dir": work.name, "kind": kind, "topic": plan["topic"],
                 "brand": result["brand"], "source": source, "posted": False, "ig_media_id": None})
    if post:
        result["ig_media_id"] = post_output(work.name)
        result["posted"] = True
    else:
        _say("Post nahi kiya. Library mein review karke 'Post' dabao.")
    return result


def post_output(dirname: str) -> str:
    """Bani hui post ko Instagram par publish karta hai."""
    work = (OUTPUT_DIR / dirname).resolve()
    if OUTPUT_DIR.resolve() not in work.parents or not (work / "result.json").exists():
        raise ValueError("Ye post nahi mili.")
    res = json.loads((work / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((work / "plan.json").read_text(encoding="utf-8"))
    if res.get("demo"):
        raise ValueError("Ye demo (placeholder) post hai, isse Instagram par post nahi kar sakte.")
    if res.get("posted"):
        raise ValueError("Ye post pehle hi ho chuki hai.")
    file = work / res["file"]
    name = f"{dirname}{file.suffix}"
    _say("Public URL par upload kar raha hu...")
    url = storage.upload_public(file, name)
    publisher = publish.publish_photo if res["kind"] == "photo" else publish.publish_reel
    try:
        _say("Instagram par publish ho raha hai (video ho to 1-5 minute)...")
        ig_id = publisher(url, plan["caption"])
    finally:
        storage.delete(name)
    res.update(posted=True, ig_media_id=ig_id, posted_at=now_ist().isoformat(timespec="seconds"))
    (work / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    update_history(dirname, posted=True, ig_media_id=ig_id)
    _say(f"Post ho gayi. Instagram media id: {ig_id}")
    return ig_id


def run_auto(post: bool = False, demo: bool = False, source: str = "manual") -> dict:
    cfg = settings()
    day = now_ist().strftime("%a").lower()[:3]
    kind = cfg.get("weekly_plan", {}).get(day, "reel")
    _say(f"Aaj ({day}) ka plan: {kind}")
    brand = None
    if kind == "ad":
        b = next_brand()
        if not b:
            _say("brands/ mein active brief nahi hai, reel bana raha hu.")
            kind = "reel"
        else:
            brand = str(b)
    # ads hamesha client approval ke liye rukte hain
    return run(kind, brand, post=post and kind != "ad", demo=demo, source=source)
