"""Clip Studio: scene-wise prompts banao (Flow/Veo jaise tool ke liye), phir unke clips jodkar reel banao.

Flow: script + prompts -> aap tool mein clips banate ho (awaaz aur lip-sync model khud banata hai)
-> yahan upload -> hum jodte hain (captions, music) -> Library -> Post.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path

import yaml

from . import captions
from .config import OUTPUT_DIR, add_history, env, now_ist
from .llm import gemini_json
from .tts import Word, estimate_words
from .video import ffmpeg_exe, media_duration, run_ff

N_SCENES = 5
CLIP_SECONDS = 4


def build_scene_prompt(s: dict) -> str:
    return (
        f"Camera Angle: {s['camera_angle']}\n"
        "Visual Description: Same female as in the reference image (same face, hair and look). "
        f"{s['visual']}\n"
        f"Emotion & Action: {s['action']}\n"
        f"Dialogue (she speaks this line out loud, clear natural voice, lips in sync): \"{s['dialogue']}\"\n"
        "Vertical 9:16, realistic phone-camera look, no text on screen, no subtitles."
    )


def _llm_prompt(topic: str | None, persona: dict, n: int) -> str:
    view = {k: persona.get(k) for k in ("name", "age", "city", "niche", "tone")}
    lines = [
        "You write a short vertical video script for an AI virtual influencer (the audience knows she is AI).",
        f"PERSONA: {yaml.safe_dump(view, allow_unicode=True)}",
    ]
    if topic:
        lines.append(f"TOPIC (use it): {topic}")
    lines += [
        f"Return ONE JSON object with keys: topic, caption, hashtags, scenes. scenes = exactly {n} objects, "
        "each with keys camera_angle, visual, action, dialogue.",
        f"- Each scene is ONE {CLIP_SECONDS}-second clip of the SAME character. Vary the camera angle "
        "(close-up, medium shot, low angle, over-shoulder, handheld).",
        "- visual: setting, everyday modest outfit, lighting. NEVER describe face, hair or body.",
        "- action: one simple emotion + small movement.",
        "- dialogue: Hinglish (Hindi in Roman script), max 10 words, natural speech, scene 1 = scroll-stopping hook, "
        "last scene = call to action or question. No emojis.",
        "- caption: Hinglish, 2-4 short lines, 1-3 emojis, ends with a question. hashtags: 8-12 strings, no #.",
        "RULES: wholesome and brand-safe, adult character, no sexual or revealing styling, no medical or "
        "guaranteed-result claims, nothing political, never pretend to be a real human.",
    ]
    return "\n".join(lines)


def _fallback(topic: str | None, n: int) -> dict:
    t = topic or "Chhoti si khushi"
    base = [
        ("Close-up, eye level, handheld", "Sunlit balcony with plants, simple cotton kurta, soft morning light",
         "Looks at camera with a curious smile", f"Ruko zara, {t[:30]} ke baare mein ek baat suno."),
        ("Medium shot, slightly low angle", "Cozy cafe corner, casual denim jacket, warm lights",
         "Leans forward and gestures with one hand", "Mujhe pehle yeh bahut mushkil lagta tha."),
        ("Close-up, shallow depth of field", "Window seat, plain white top, natural light",
         "Nods and smiles softly", "Phir maine ek chhota sa change kiya."),
        ("Medium shot, over the shoulder", "Street with market lights, light jacket, golden hour",
         "Walks a few steps and looks back", "Aur dekho, ab farak saaf dikhta hai."),
        ("Close-up, eye level", "Bright living room, comfy sweater, soft daylight",
         "Smiles and raises eyebrows", "Tum kya try karoge? Comment mein batao."),
    ]
    scenes = [dict(zip(("camera_angle", "visual", "action", "dialogue"), base[i % len(base)])) for i in range(n)]
    return {"topic": t[:80], "caption": f"{t} 💛\nTumhara kya khayal hai?", "hashtags": ["aiinfluencer", "reels"], "scenes": scenes}


def _clean_scene(s, fb: dict) -> dict:
    s = s if isinstance(s, dict) else {}
    return {k: re.sub(r"\s+", " ", str(s.get(k) or fb[k])).strip() for k in ("camera_angle", "visual", "action", "dialogue")}


def make_scene_script(topic: str | None, persona: dict, n: int = N_SCENES, offline: bool = False) -> dict:
    fb = _fallback(topic, n)
    raw, note = fb, "Template use hua (Gemini/OpenAI ka jawab nahi mila)."
    if not offline:
        try:
            got = gemini_json(_llm_prompt(topic, persona, n))
            if isinstance(got.get("scenes"), list) and got["scenes"]:
                raw, note = got, ""
        except Exception as e:  # noqa: BLE001
            note = f"Template use hua ({str(e)[:120]})."
    scenes = []
    for i in range(n):
        src = raw["scenes"][i] if i < len(raw["scenes"]) else {}
        sc = _clean_scene(src, fb["scenes"][i])
        sc["prompt"] = build_scene_prompt(sc)
        scenes.append(sc)
    tags = [re.sub(r"[^\w]", "", str(t)) for t in (raw.get("hashtags") or [])] + list(persona.get("hashtags", []))
    tags = list(dict.fromkeys(t for t in tags if t))[:28]
    parts = [str(raw.get("caption") or fb["caption"]).strip(), persona.get("disclosure", ""),
             " ".join("#" + t for t in tags)]
    return {"topic": str(raw.get("topic") or fb["topic"])[:80], "caption": "\n\n".join(p for p in parts if p)[:2150],
            "scenes": scenes, "note": note}


def has_audio(path: Path) -> bool:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in p.stderr


def _normalize(src: Path, dst: Path, w: int, h: int, fps: int, idx: int = 0, cfg: dict | None = None) -> float:
    """Clip ko 9:16 mein laata hai. Pro-style: shuru/aakhir ke dead frames trim, ek-ek chhod kar halka punch-in
    zoom (jump-cut feel), awaaz ka volume barabar, aur chhoti fade (click/pop nahi)."""
    cfg = cfg or {}
    full = media_duration(src)
    head = float(cfg.get("clip_trim_head", 0.15))
    tail = float(cfg.get("clip_trim_tail", 0.2))
    if full - head - tail < 1.0:  # bahut chhoti clip ko mat kaato
        head = tail = 0.0
    sp = min(max(float(cfg.get("clip_speed", 1.0)), 0.8), 1.5)
    dur = (full - head - tail) / sp  # output ki lambai (speed ke baad)
    z = float(cfg.get("punch_zoom", 1.08)) if idx % 2 == 1 else 1.0
    zw, zh = int(w * z) // 2 * 2, int(h * z) // 2 * 2
    vf = (f"scale={zw}:{zh}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,setpts=PTS/{sp:.4f},fps={fps},format=yuv420p")
    enc = ["-threads", "1", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
           "-x264-params", "rc-lookahead=8:ref=2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2",
           "-b:a", "160k", "-t", f"{dur:.2f}"]
    fade = max(dur - 0.08, 0.1)
    tempo = f"atempo={sp:.4f}," if abs(sp - 1.0) > 0.001 else ""
    af = tempo + (f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:d=0.04,afade=t=out:st={fade:.2f}:d=0.08"
                  if cfg.get("normalize_audio", True) else "anull")
    if has_audio(src):
        run_ff(["-ss", f"{head:.2f}", "-i", str(src), "-vf", vf, "-af", af, *enc, str(dst)])
    else:
        run_ff(["-ss", f"{head:.2f}", "-i", str(src), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-vf", vf,
                "-map", "0:v", "-map", "1:a", "-shortest", *enc, str(dst)])
    return dur


def merge(clips: list[Path], dialogues: list[str], work: Path, video_cfg: dict, music: Path | None = None,
          say=print) -> Path:
    w, h, fps = int(video_cfg.get("width", 1080)), int(video_cfg.get("height", 1920)), int(video_cfg.get("fps", 30))
    names, words, t0 = [], [], 0.0
    for i, src in enumerate(clips):
        say(f"Clip {i + 1}/{len(clips)} taiyar ho rahi hai...")
        dst = work / f"norm_{i}.mp4"
        dur = _normalize(src, dst, w, h, fps, i, video_cfg)
        names.append(dst.name)
        text = (dialogues[i] if i < len(dialogues) else "").strip()
        if text:
            words += [Word(x.text, x.start + t0, x.end + t0) for x in estimate_words(text, max(dur - 0.3, 0.5))]
        t0 += dur
    (work / "clips.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
    ass = captions.build_ass(words, work / "captions.ass", video_cfg.get("caption", {}), w, h) if words else None
    out = work / "reel.mp4"

    def final(with_caps: bool) -> None:
        args = ["-f", "concat", "-safe", "0", "-i", "clips.txt"]
        vf = "subtitles=captions.ass" if (with_caps and ass) else "null"
        if music:
            args += ["-stream_loop", "-1", "-i", str(music.resolve())]
            vol = float(video_cfg.get("music_volume", 0.12))
            fc = (f"[0:v]{vf}[v];[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first:"
                  "dropout_transition=0,volume=2,alimiter=limit=0.95[a]")
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            fc, maps = f"[0:v]{vf}[v]", ["-map", "[v]", "-map", "0:a"]
        args += ["-filter_complex", fc, *maps, "-t", f"{t0:.2f}", "-r", str(fps), "-threads", "1", "-filter_threads", "1",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-x264-params", "rc-lookahead=8:ref=2",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out.resolve())]
        run_ff(args, cwd=work)

    say("Clips jod raha hu, captions aur music laga raha hu...")
    try:
        final(True)
    except RuntimeError as e:
        if ass and ("subtitles" in str(e).lower() or "filter" in str(e).lower()):
            say("Captions burn nahi ho paye, bina captions ke bana raha hu.")
            final(False)
        else:
            raise
    return out


def save_item(work: Path, topic: str, caption: str, final: Path, source: str = "web") -> dict:
    (work / "plan.json").write_text(json.dumps({"topic": topic, "caption": caption, "kind": "reel"},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    (work / "caption.txt").write_text(caption, encoding="utf-8")
    result = {"dir": work.name, "kind": "reel", "topic": topic, "file": final.name,
              "created": now_ist().isoformat(timespec="seconds"), "brand": None, "source": source,
              "demo": False, "posted": False, "ig_media_id": None, "clip_studio": True}
    (work / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    add_history({"date": result["created"], "dir": work.name, "kind": "reel", "topic": topic, "brand": None,
                 "source": source, "posted": False, "ig_media_id": None})
    return result


def new_work_dir(topic: str) -> Path:
    slug = (re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-") or "clips")[:30]
    work = OUTPUT_DIR / f"{now_ist().strftime('%Y-%m-%d_%H%M%S')}_reel_{slug}"
    work.mkdir(parents=True, exist_ok=True)
    return work


def pick_music(root: Path) -> Path | None:
    d = root / "assets" / "music"
    tracks = [p for p in d.glob("*") if p.suffix.lower() in (".mp3", ".m4a", ".wav")] if d.exists() else []
    return random.choice(tracks) if tracks else None
