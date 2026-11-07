"""Story Studio: library (kirdaar / jagah / product ki photo, video) + brief se scene-wise plan.

Flow: user library mein photo daalta hai (tag + likhit permission) -> brief likhta hai -> yahan plan banta hai:
scenes, timing, bolne ki lines, aur har scene ka Flow prompt (locks ke saath) + keyframe image ka prompt.
LLM (Gemini/OpenAI) ho to wo plan likhta hai, na ho to simple template. Jo bhi plan ho, user card dekhkar approve karta hai.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path

from . import persist
from .config import ROOT, env, now_ist
from .llm import gemini_json

LIB_DIR = ROOT / "state" / "library"
LIB_FILE = ROOT / "state" / "library.json"
KINDS = ("character", "location", "product", "clip")
KIND_LABEL = {"character": "Kirdaar (insaan)", "location": "Jagah", "product": "Product / cheez", "clip": "Apni video clip"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".webm"}
MAX_BYTES = 40 * 1024 * 1024
SCENE_SECONDS = 8          # Flow mein ek clip ~8 s
WPS = 2.4                  # bolne ki raftaar (words / second)
MAX_SCENES = 8
MAX_ITEMS = 60


class StoryError(ValueError):
    pass


# ---------------------------------------------------------------- library
def _load() -> list[dict]:
    if LIB_FILE.exists():
        try:
            data = json.loads(LIB_FILE.read_text(encoding="utf-8"))
            return [i for i in data.get("items", []) if isinstance(i, dict)]
        except (OSError, json.JSONDecodeError):
            return []
    return []


def _save(items: list[dict]) -> None:
    LIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    LIB_FILE.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    persist.push(LIB_FILE)


def _clean(s: str, n: int) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def list_items() -> list[dict]:
    return sorted(_load(), key=lambda i: i.get("created", ""), reverse=True)


def get_items(ids: list[str]) -> list[dict]:
    by = {i["id"]: i for i in _load()}
    return [by[x] for x in ids if x in by]


def add_item(kind: str, name: str, desc: str, filename: str, data: bytes, consent: bool) -> dict:
    if kind not in KINDS:
        raise StoryError("Type galat hai (kirdaar / jagah / product / clip).")
    name = _clean(name, 60)
    if not name:
        raise StoryError("Naam likho (jaise 'Dr Ashish' ya 'Clinic reception').")
    if kind == "character" and not consent:
        raise StoryError("Kirdaar (insaan) ke liye likhit permission ka tick zaroori hai.")
    ext = Path(filename or "").suffix.lower()
    if kind == "clip":
        if ext not in VID_EXT:
            raise StoryError("Clip ke liye mp4 / mov chahiye.")
    elif ext not in IMG_EXT:
        raise StoryError("Photo jpg / png / webp honi chahiye.")
    if not data or len(data) > MAX_BYTES:
        raise StoryError("File khaali hai ya 40 MB se badi hai.")
    if len(_load()) >= MAX_ITEMS:
        raise StoryError(f"Library mein {MAX_ITEMS} se zyada item nahi ho sakte. Purane hata do.")
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    iid = uuid.uuid4().hex[:10]
    out = LIB_DIR / f"{iid}{ext if ext in VID_EXT else '.jpg'}"
    if ext in IMG_EXT:
        out.write_bytes(_shrink(data))
    else:
        out.write_bytes(data)
    item = {"id": iid, "kind": kind, "name": name, "desc": _clean(desc, 300), "file": out.name,
            "consent": bool(consent) if kind == "character" else None, "created": now_ist().isoformat(timespec="seconds")}
    items = _load()
    items.append(item)
    _save(items)
    persist.push(out)
    return item


def _shrink(data: bytes) -> bytes:
    """Photo ko JPEG mein badal kar max 1280 px karta hai (RAM aur storage bachane ke liye)."""
    import io
    from PIL import Image, ImageOps
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        im = im.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise StoryError("Ye photo khul nahi rahi, doosri try karo.") from e
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def delete_item(iid: str) -> bool:
    items = _load()
    keep = [i for i in items if i["id"] != iid]
    if len(keep) == len(items):
        return False
    for i in items:
        if i["id"] == iid:
            f = LIB_DIR / i.get("file", "")
            if f.is_file() and LIB_DIR.resolve() in f.resolve().parents:
                f.unlink(missing_ok=True)
                persist.remove(f)
    _save(keep)
    return True


# ---------------------------------------------------------------- plan
def scene_count(total_seconds: float) -> int:
    return min(max(int(float(total_seconds) / SCENE_SECONDS + 0.5), 1), MAX_SCENES)


def _bounds(sec: float) -> tuple[int, int]:
    mid = sec * WPS
    return max(int(mid - 3), 4), int(mid)


def _words(s: str) -> int:
    return len(re.findall(r"\S+", s or ""))


def _llm_prompt(brief: str, n: int, cast: list[dict], places: list[dict], things: list[dict], lang: str) -> str:
    def row(i: dict) -> str:
        return f'- id={i["id"]} | {i["name"]}' + (f' | {i["desc"]}' if i.get("desc") else "")
    lo, hi = _bounds(SCENE_SECONDS)
    return "\n".join([
        "You are a short-video story planner and scriptwriter for an Indian clinic / business reel. The video is clearly "
        "AI-assisted and the real people will review and approve every line before posting.",
        f"Language for spoken lines: {lang}. Write natural everyday speech, like a real doctor or customer in India.",
        f"BRIEF FROM THE USER:\n{brief}",
        f"Make exactly {n} scenes, each ONE {SCENE_SECONDS}-second clip, following the brief's story in order.",
        "CHARACTERS (use only these ids):\n" + ("\n".join(row(i) for i in cast) or "- none"),
        "LOCATIONS (use only these ids):\n" + ("\n".join(row(i) for i in places) or "- none"),
        "PRODUCTS / OBJECTS (optional):\n" + ("\n".join(row(i) for i in things) or "- none"),
        "Return ONE JSON object with keys: title, summary, caption, hashtags, scenes.",
        "Each scene is an object with keys: beat (what happens, 1 line), type (\"talking\" = a character speaks to camera "
        "or to another character; \"broll\" = visuals only, a voiceover plays), characters (list of character ids on screen), "
        "speaker (character id who speaks, or \"\" for none), location (location id or \"\"), camera (simple shot like "
        "\"Medium shot, eye level\"), action (what the people physically do, simple), visual (for broll: what is shown), "
        "line (the spoken words; for talking scenes the speaker says it, for broll it is the voiceover).",
        f"- line: between {lo} and {hi} words (speech is about {WPS} words per second), ONE complete idea that finishes "
        "inside the clip. Never an empty line.",
        "- Scene 1 = a hook that makes the viewer stay. Last scene = a soft invitation (ask a question / book a consultation "
        "/ greeting), not a hard sell.",
        "- Mix scene types so the screen changes at least every 8 seconds; never use the same camera twice in a row.",
        "- Use only ids from the lists above. Do not describe faces or clothes (the reference photos decide that).",
        "- hashtags: 8-12 strings without #. caption: 2-4 short lines.",
        "RULES: general information only. Never invent credentials, patient stories, testimonials, before/after results, "
        "prices, offers or guarantees. No claims like cure, permanent, 100%, painless or risk-free. No medical advice for a "
        "specific person. Nothing political. Brand-safe.",
    ])


def _template_scenes(brief: str, n: int, cast: list[dict], places: list[dict]) -> list[dict]:
    """LLM na ho to: brief ke vakyon ko scenes mein baantta hai. Lines user ko edit karni hongi."""
    parts = [p.strip(" .") for p in re.split(r"[.\n;]|\bphir\b|\bfir\b|\bthen\b|,", brief, flags=re.I) if len(p.strip()) > 3]
    if not parts:
        parts = [brief.strip()[:100] or "Kahani"]
    c0 = cast[0]["id"] if cast else ""
    out = []
    cams = ["Medium shot, eye level", "Close-up, eye level", "Wide shot, slightly low angle", "Medium close-up, eye level"]
    for i in range(n):
        beat = parts[min(i * len(parts) // n, len(parts) - 1)]
        c = cast[i % len(cast)]["id"] if cast else ""
        out.append({"beat": beat, "type": "talking" if cast else "broll", "characters": [c] if c else [],
                    "speaker": c, "location": places[i % len(places)]["id"] if places else "",
                    "camera": cams[i % len(cams)], "action": "Speaks naturally, small hand gesture",
                    "visual": beat, "line": ""})
    if out and c0:
        out[0]["speaker"] = c0
    return out


def _norm_scene(s: dict, i: int, cast_ids: set, place_ids: set, warn: list[str]) -> dict:
    t = "broll" if str(s.get("type", "")).lower().startswith("b") else "talking"
    chars = [c for c in (s.get("characters") or []) if c in cast_ids] if isinstance(s.get("characters"), list) else []
    sp = s.get("speaker") if s.get("speaker") in cast_ids else ""
    if t == "talking":
        if not sp and chars:
            sp = chars[0]
        if sp and sp not in chars:
            chars.insert(0, sp)
        if not sp:
            t = "broll"
            warn.append(f"Scene {i + 1}: bolne wala kirdaar nahi mila, ise B-roll bana diya.")
    loc = s.get("location") if s.get("location") in place_ids else ""
    line = _clean(s.get("line") or s.get("presenter_line") or "", 220)
    lo, hi = _bounds(SCENE_SECONDS)
    w = _words(line)
    if not line:
        warn.append(f"Scene {i + 1}: bolne ki line khaali hai, likh do.")
    elif w > hi + 4:
        warn.append(f"Scene {i + 1}: line {w} shabd ki hai (~{hi} se zyada), {SCENE_SECONDS} s me poori nahi bol payegi.")
    elif w < lo - 3:
        warn.append(f"Scene {i + 1}: line {w} shabd ki hai, chhoti hai. Beech me chup rahega.")
    return {"n": i + 1, "seconds": SCENE_SECONDS, "type": t, "beat": _clean(s.get("beat"), 160),
            "characters": chars, "speaker": sp, "location": loc, "camera": _clean(s.get("camera") or "Medium shot, eye level", 80),
            "action": _clean(s.get("action") or "Speaks naturally", 160), "visual": _clean(s.get("visual"), 200), "line": line,
            "words": w}


def _ref(i: dict) -> str:
    return f'"{i["name"]}"'


def flow_prompt(sc: dict, by: dict, lang: str, voice_name: str = "") -> str:
    """Ek scene ka Flow (Veo) prompt: locks + shot + bolna."""
    chars = [by[c] for c in sc["characters"] if c in by]
    loc = by.get(sc["location"])
    lines = []
    if chars:
        lines.append("IDENTITY LOCK (read fully before creating the video): use ONLY the attached reference images for the people. "
                     "Keep each person's exact face, hairstyle, facial hair, skin tone, body, outfit and age. Do not change anything about them.")
        for k, c in enumerate(chars):
            who = "speaker" if c["id"] == sc["speaker"] else "in frame"
            lines.append(f"- Person {k + 1} ({who}): {_ref(c)}" + (f", {c['desc']}" if c.get("desc") else "") + ", as in its attached reference image.")
    if loc:
        lines.append(f"LOCATION LOCK: {_ref(loc)}" + (f", {loc['desc']}" if loc.get("desc") else "")
                     + ". Same place, same decoration and same light as the attached location image; do not change it.")
    lines.append(f"Format: vertical 9:16, {sc['seconds']} seconds, realistic phone-camera look, no subtitles, no text or logos added on screen.")
    if sc["type"] == "talking" and sc["speaker"] in by:
        sp = by[sc["speaker"]]
        lines.append(f"Seconds 0-{sc['seconds']}: {sc['camera']}. Locked-off camera, no camera movement. {sc['action']}. "
                     f"{_ref(sp)} speaks in {lang}, clear natural voice, lips in sync, speaking continuously until the last second: "
                     f"\"{sc['line']}\"")
    else:
        lines.append(f"Seconds 0-{sc['seconds']}: {sc['camera']}. {sc['visual'] or sc['beat']}. Slow, smooth, natural motion. "
                     "No one speaks and no on-screen text (a voiceover is added later in editing).")
    lines.append("Same faces and same place in every clip. Wholesome and brand-safe.")
    return "\n".join(lines)


def keyframe_prompt(sc: dict, by: dict) -> str:
    """ChatGPT / image tool ke liye: is scene ki pehli still image (Flow start frame)."""
    chars = [by[c] for c in sc["characters"] if c in by]
    loc = by.get(sc["location"])
    lines = ["CREATE ONE SINGLE IMAGE (one photograph, never a collage or grid). Vertical 9:16, 1080 x 1920 px, photorealistic.",
             "The attached photos are IDENTITY / PLACE references only."]
    for c in chars:
        lines.append(f"- {_ref(c)}: keep this person's face, hair, skin tone, outfit and age exactly as in their photo.")
    if loc:
        lines.append(f"- {_ref(loc)}: recreate this same place (same decor, colours, light)" + (f": {loc['desc']}" if loc.get("desc") else "") + ".")
    lines.append(f"SCENE: {sc['beat'] or sc['visual']}. Camera: {sc['camera']}. {sc['action'] if chars else sc['visual']}.")
    lines.append("Natural skin texture, realistic light, sharp eyes. No text, no logo, no watermark, no extra people or fingers.")
    return "\n".join(lines)


def plan_story(brief: str, total_seconds: float, cast_ids: list[str], place_ids: list[str], thing_ids: list[str],
               language: str = "Hinglish (Hindi in Roman script)", offline: bool | None = None) -> dict:
    brief = _clean(brief, 1500)
    if len(brief) < 15:
        raise StoryError("Brief thoda detail me likho (kaun, kahan, kya hota hai).")
    cast, places, things = get_items(cast_ids), get_items(place_ids), get_items(thing_ids)
    cast = [c for c in cast if c["kind"] == "character"]
    places = [p for p in places if p["kind"] == "location"]
    things = [t for t in things if t["kind"] in ("product", "clip")]
    if not cast:
        raise StoryError("Kam se kam ek kirdaar (insaan) library se chuno.")
    bad = [c["name"] for c in cast if not c.get("consent")]
    if bad:
        raise StoryError("In kirdaaron ki likhit permission nahi hai: " + ", ".join(bad))
    n = scene_count(total_seconds)
    warn: list[str] = []
    source = "template"
    raw = None
    if offline is None:
        offline = not (env("GEMINI_API_KEY") or env("OPENAI_API_KEY"))
    if not offline:
        try:
            raw = gemini_json(_llm_prompt(brief, n, cast, places, things, language))
            source = "ai"
        except Exception as e:  # noqa: BLE001
            warn.append(f"AI se plan nahi bana ({str(e)[:90]}), simple template use hua. Lines khud likho.")
    else:
        warn.append("AI key set nahi hai, isliye simple template bana. Bolne ki lines khud likhni/edit karni hongi.")
    raw_scenes = raw.get("scenes") if isinstance(raw, dict) and isinstance(raw.get("scenes"), list) else None
    if not raw_scenes:
        raw_scenes = _template_scenes(brief, n, cast, places)
        source = "template"
    raw_scenes = [s for s in raw_scenes if isinstance(s, dict)][:n]
    while len(raw_scenes) < n:
        raw_scenes.append(dict(raw_scenes[-1]) if raw_scenes else {})
        warn.append(f"Scene {len(raw_scenes)}: AI ne kam scenes diye, pichla scene copy hua. Edit karo.")
    cast_ids_s, place_ids_s = {c["id"] for c in cast}, {p["id"] for p in places}
    scenes = [_norm_scene(s, i, cast_ids_s, place_ids_s, warn) for i, s in enumerate(raw_scenes)]
    by = {i["id"]: i for i in cast + places + things}
    for sc in scenes:
        sc["flow_prompt"] = flow_prompt(sc, by, language)
        sc["keyframe_prompt"] = keyframe_prompt(sc, by)
    if len({sc["camera"] for sc in scenes}) == 1 and n > 1:
        warn.append("Saare scenes ka camera angle ek jaisa hai. Kuch scenes me alag angle likho.")
    d = raw if isinstance(raw, dict) and source == "ai" else {}
    return {"title": _clean(d.get("title") or brief[:50], 80), "summary": _clean(d.get("summary") or "", 300),
            "caption": str(d.get("caption") or "")[:600], "hashtags": [str(h).lstrip("#") for h in (d.get("hashtags") or [])][:12],
            "total_seconds": n * SCENE_SECONDS, "n_scenes": n, "source": source, "scenes": scenes, "warnings": warn,
            "language": language,
            "bible": {"characters": [{"id": c["id"], "name": c["name"]} for c in cast],
                      "locations": [{"id": p["id"], "name": p["name"]} for p in places]}}


def rebuild_prompts(plan: dict) -> dict:
    """User ne line/camera edit ki ho to prompts dobara banao (library se)."""
    ids = [c["id"] for c in plan.get("bible", {}).get("characters", [])] + [p["id"] for p in plan.get("bible", {}).get("locations", [])]
    by = {i["id"]: i for i in get_items(ids)}
    lang = plan.get("language") or "Hinglish (Hindi in Roman script)"
    lo, hi = _bounds(SCENE_SECONDS)
    keep = [w for w in plan.get("warnings", []) if "line" not in w]
    for sc in plan.get("scenes", []):
        sc["words"] = _words(sc.get("line", ""))
        if not sc["words"]:
            keep.append(f"Scene {sc.get('n')}: bolne ki line khaali hai, likh do.")
        elif sc["words"] > hi + 4:
            keep.append(f"Scene {sc.get('n')}: line {sc['words']} shabd ki hai, {SCENE_SECONDS} s me poori nahi bol payegi.")
        elif sc["words"] < lo - 3:
            keep.append(f"Scene {sc.get('n')}: line {sc['words']} shabd ki hai, chhoti hai.")
        sc["flow_prompt"] = flow_prompt(sc, by, lang)
        sc["keyframe_prompt"] = keyframe_prompt(sc, by)
    plan["warnings"] = keep
    return plan
