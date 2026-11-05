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
        f"Return ONE JSON object with keys: topic, hook_text, caption, hashtags, scenes. scenes = exactly {n} objects, "
        "each with keys camera_angle, visual, action, dialogue.",
        f"- Each scene is ONE {CLIP_SECONDS}-second clip of the SAME character. Vary the camera angle "
        "(close-up, medium shot, low angle, over-shoulder, handheld).",
        "- visual: setting, everyday modest outfit, lighting. NEVER describe face, hair or body.",
        "- action: one simple emotion + small movement.",
        "- dialogue: Hinglish (Hindi in Roman script), max 10 words, natural speech, scene 1 = scroll-stopping hook, "
        "last scene = call to action or question. No emojis.",
        "- hook_text: max 6 words shown as big on-screen text for the first 2 seconds; curiosity or clear benefit, "
        "Hinglish, no emojis.",
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
    return {"topic": t[:80], "hook_text": f"{t[:28]}? YE DEKHO", "caption": f"{t} 💛\nTumhara kya khayal hai?", "hashtags": ["aiinfluencer", "reels"], "scenes": scenes}


def _clean_scene(s, fb: dict) -> dict:
    s = s if isinstance(s, dict) else {}
    return {k: re.sub(r"\s+", " ", str(s.get(k) or fb[k])).strip() for k in ("camera_angle", "visual", "action", "dialogue")}


# ---------------- Avatar mode (client ka asli avatar: 10 sec scene = 4 sec presenter + 6 sec visuals) ----------------
AVATAR_SCENES = 3
PRESENTER_SECONDS = 4
_RISKY = re.compile(r"guarantee|100\s*%|permanent|\bcure\b|chamatkar|risk[- ]?free|no side effect|pain[- ]?free|painless",
                    re.I)


def _avatar_norm(a: dict | None) -> dict | None:
    if not a or not isinstance(a, dict):
        return None

    def clean(k: str, default: str = "", n: int = 200) -> str:
        return re.sub(r"\s+", " ", str(a.get(k) or default)).strip()[:n]

    try:
        n = int(a.get("scenes") or AVATAR_SCENES)
    except (TypeError, ValueError):
        n = AVATAR_SCENES
    return {"name": clean("name", "Avatar", 60), "role": clean("role", "", 80),
            "gender": "female" if str(a.get("gender", "male")).lower().startswith("f") else "male",
            "look": clean("look", "", 300), "language": clean("language", "Hinglish (Hindi in Roman script)", 60),
            "scenes": min(max(n, 1), 6)}


def build_avatar_prompt(s: dict, a: dict) -> str:
    he, his = ("He", "His") if a["gender"] == "male" else ("She", "Her")
    look = f" {a['look'].rstrip('. ')}." if a["look"] else ""
    return (
        "IDENTITY LOCK (read this fully before creating the video): use ONLY the attached avatar for the person. "
        "Keep the exact same face, hairstyle, facial hair, skin tone, body, outfit, voice, background and lighting "
        f"as the avatar in this clip. Do not change anything about the person.{look}\n"
        "Format: vertical 9:16, 10 seconds, realistic phone-camera look, no subtitles, no text or logos added on screen.\n"
        f"Seconds 0-{PRESENTER_SECONDS} (presenter on camera): {s['camera_angle']}. {s['action']}. "
        f"{he} speaks to the camera in {a['language']}, clear natural voice, lips in sync: \"{s['presenter_line']}\"\n"
        f"Seconds {PRESENTER_SECONDS}-10 (cutaway visuals, presenter not on screen): {s['visual']}. "
        f"{his} voiceover continues in the exact same voice and tone: \"{s['voiceover_line']}\"\n"
        "Same face and same voice in every clip. Wholesome and brand-safe."
    )


def _avatar_llm_prompt(topic: str | None, a: dict, n: int) -> str:
    who = a["name"] + (f", {a['role']}" if a["role"] else "")
    lines = [
        f"You write a short vertical video script for {who}. The video is clearly AI-assisted and the real person "
        "will review and approve every line before posting.",
        f"Language: {a['language']}.",
    ]
    if topic:
        lines.append(f"TOPIC (use it): {topic}")
    lines += [
        f"Return ONE JSON object with keys: topic, hook_text, caption, hashtags, scenes. scenes = exactly {n} objects "
        "with keys camera_angle, action, presenter_line, visual, voiceover_line.",
        f"- Each scene is ONE 10-second clip: seconds 0-{PRESENTER_SECONDS} the presenter speaks to camera "
        f"(presenter_line), seconds {PRESENTER_SECONDS}-10 cutaway visuals while the same voice continues "
        "(voiceover_line).",
        "- presenter_line: max 10 words. voiceover_line: max 16 words. Together about 120-145 characters, one complete "
        "idea, sentences must finish inside the scene.",
        "- Scene 1 = scroll-stopping hook. Last scene = invite the viewer to ask a question or book a consultation.",
        "- camera_angle and action: simple (close-up, medium shot; small gesture, nod, smile). Do not describe the face.",
        "- visual: what the cutaway shows (clinic or lifestyle b-roll, hands, objects, text-free scenes). No faces.",
        "- hook_text: max 6 words shown on screen for the first 2 seconds. caption: 2-4 short lines, ends with a "
        "question. hashtags: 8-12 strings, no #.",
        "RULES: general education only. Never invent credentials, patient stories, testimonials, before/after results, "
        "prices, offers or guarantees. No claims like cure, permanent, 100%, painless or risk-free. No medical advice "
        "for a specific person; suggest consulting a qualified doctor. Nothing political. Brand-safe.",
    ]
    return "\n".join(lines)


def _avatar_fallback(topic: str | None, a: dict, n: int) -> dict:
    t = (topic or "ek zaroori jaankari")[:60]
    nm = a["name"]
    base = [
        ("Close-up, eye level", "Smiles and gives a small wave",
         f"Namaste, main {nm}. Aaj {t} ke baare mein suniye.", "Clean, bright clinic reception, soft daylight, hands arranging papers",
         "Sahi jaankari hi pehla kadam hai, aur galat jaankari se bachna zaroori hai."),
        ("Medium shot, slightly low angle", "Gestures with one hand while explaining",
         "Sabse pehle ek baat samajh lijiye.", "Close-up of a notebook and pen on a desk, warm light",
         "Har insaan ka case alag hota hai, isliye general baatein sabke liye ek jaisi nahi hoti."),
        ("Close-up, shallow depth of field", "Nods and speaks calmly",
         "Faisla lene se pehle sawal poochiye.", "Soft-focus shots of a calm waiting area with plants",
         "Apne doctor se poori baat karein, aur jo samajh na aaye wo dobara poochiye."),
        ("Medium shot, eye level", "Leans slightly forward, friendly tone",
         "Ek aur zaroori baat.", "Hands holding a cup of tea near a window, natural light",
         "Jaldbaazi mein nahi, soch samajh kar aur sahi salah ke saath kadam uthaiye."),
        ("Close-up, eye level", "Smiles and raises eyebrows",
         "Aapka sawal kya hai?", "Phone showing a chat window, text-free, soft light",
         "Comment mein apna sawal likhiye, ya consultation ke liye clinic se baat kijiye."),
    ]
    scenes = [dict(zip(("camera_angle", "action", "presenter_line", "visual", "voiceover_line"), base[i % len(base)]))
              for i in range(n)]
    return {"topic": t[:80], "hook_text": f"{t[:28]}? JAANIYE", "caption": f"{t}\nAapka kya sawal hai?",
            "hashtags": ["education", "reels"], "scenes": scenes}


_AV_KEYS = ("camera_angle", "action", "presenter_line", "visual", "voiceover_line")


def _clean_avatar_scene(s, fb: dict) -> dict:
    s = s if isinstance(s, dict) else {}
    out = {k: re.sub(r"\s+", " ", str(s.get(k) or fb[k])).strip() for k in _AV_KEYS}
    if _RISKY.search(out["presenter_line"] + " " + out["voiceover_line"] + " " + out["visual"]):
        return dict(fb)  # risky wording: safe template line use karo
    return out


def _make_avatar_script(topic: str | None, a: dict, offline: bool) -> dict:
    n = a["scenes"]
    fb = _avatar_fallback(topic, a, n)
    raw, note = fb, "Template use hua (Gemini/OpenAI ka jawab nahi mila). Doctor se script approve karwa lijiye."
    if not offline:
        try:
            got = gemini_json(_avatar_llm_prompt(topic, a, n))
            if isinstance(got.get("scenes"), list) and got["scenes"]:
                raw, note = got, "Script doctor/client se approve karwa kar hi post karein."
        except Exception as e:  # noqa: BLE001
            note = f"Template use hua ({str(e)[:120]}). Doctor se script approve karwa lijiye."
    scenes = []
    for i in range(n):
        src = raw["scenes"][i] if i < len(raw["scenes"]) else {}
        sc = _clean_avatar_scene(src, fb["scenes"][i])
        sc["dialogue"] = f"{sc['presenter_line']} {sc['voiceover_line']}".strip()
        sc["prompt"] = build_avatar_prompt(sc, a)
        scenes.append(sc)
    tags = list(dict.fromkeys(t for t in (re.sub(r"[^\w]", "", str(x)) for x in (raw.get("hashtags") or [])) if t))[:20]
    disclosure = f"AI-assisted video, {a['name']} ki approved script par bani."
    cap = [str(raw.get("caption") or fb["caption"]).strip(), disclosure, " ".join("#" + t for t in tags)]
    hook = re.sub(r"\s+", " ", str(raw.get("hook_text") or fb["hook_text"])).strip()[:60]
    if _RISKY.search(hook):
        hook = fb["hook_text"]
    return {"topic": str(raw.get("topic") or fb["topic"])[:80], "caption": "\n\n".join(p for p in cap if p)[:2150],
            "hook_text": hook, "scenes": scenes, "note": note, "mode": "avatar"}


def make_scene_script(topic: str | None, persona: dict, n: int = N_SCENES, offline: bool = False,
                      avatar: dict | None = None) -> dict:
    av = _avatar_norm(avatar)
    if av:  # client ka asli avatar (Avatar mode)
        return _make_avatar_script(topic, av, offline)
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
            "hook_text": re.sub(r"\s+", " ", str(raw.get("hook_text") or fb["hook_text"])).strip()[:60],
            "scenes": scenes, "note": note}


def has_audio(path: Path) -> bool:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in p.stderr


def speech_bounds(path: Path, db: float = -32.0, min_sil: float = 0.12) -> tuple[float, float] | None:
    """Clip mein awaaz kahan se kahan tak hai (second mein). Awaaz na mile ya poori clip khamosh ho to None."""
    if not has_audio(path):
        return None
    total = media_duration(path)
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
                        f"silencedetect=noise={db}dB:d={min_sil}", "-f", "null", "-"], capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", p.stderr)]
    ivs = [(s, ends[i] if i < len(ends) else None) for i, s in enumerate(starts)]
    start, end = 0.0, total
    if ivs and ivs[0][0] <= 0.05:  # clip khamoshi se shuru hoti hai
        if ivs[0][1] is None or ivs[0][1] >= total - 0.05:
            return None  # poori clip khamosh
        start = ivs[0][1]
    if ivs:
        s_last, e_last = ivs[-1]
        if (e_last is None or e_last >= total - 0.05) and s_last > start:
            end = s_last  # aakhir tak khamoshi chali gayi
    return (start, end) if end - start >= 0.3 else None


def _normalize(src: Path, dst: Path, w: int, h: int, fps: int, idx: int = 0, cfg: dict | None = None) -> float:
    """Clip ko 9:16 mein laata hai. Pro-style: awaaz ke shuru/aakhir ki lambi khamoshi (seam ka asli 'gap') kaat kar
    sirf chhota natural pause chhodta hai, ek-ek chhod kar halka punch-in zoom (jump-cut feel), awaaz ka volume
    barabar, aur bahut chhoti fade (click/pop nahi, par dip bhi nahi)."""
    cfg = cfg or {}
    full = media_duration(src)
    head = float(cfg.get("clip_trim_head", 0.15))
    tail = float(cfg.get("clip_trim_tail", 0.2))
    if cfg.get("trim_silence", True):
        b = speech_bounds(src, float(cfg.get("silence_db", -32.0)))
        if b:  # awaaz mili: usi ke hisaab se kaato (thoda pad chhod kar)
            head = min(max(b[0] - float(cfg.get("clip_pad_head", 0.08)), 0.0), 1.2)
            tail = min(max(full - (b[1] + float(cfg.get("clip_pad_tail", 0.18))), 0.0), 1.5)
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
    fade = max(dur - 0.015, 0.05)
    tempo = f"atempo={sp:.4f}," if abs(sp - 1.0) > 0.001 else ""
    af = tempo + (f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:d=0.012,afade=t=out:st={fade:.3f}:d=0.012"
                  if cfg.get("normalize_audio", True) else "anull")
    if has_audio(src):
        run_ff(["-ss", f"{head:.2f}", "-i", str(src), "-vf", vf, "-af", af, *enc, str(dst)])
    else:
        run_ff(["-ss", f"{head:.2f}", "-i", str(src), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-vf", vf,
                "-map", "0:v", "-map", "1:a", "-shortest", *enc, str(dst)])
    return dur


def revoice(src: Path, dst: Path, convert) -> Path:
    """Clip ki awaaz nikaal kar convert(audio_in, audio_out) se badalta hai aur wapas video par chipkata hai.
    Video ek frame bhi nahi chhedta, awaaz ki timing wahi rehti hai (lip-sync bacha rehta hai)."""
    a_in, a_out = dst.with_suffix(".in.mp3"), dst.with_suffix(".out.mp3")
    run_ff(["-i", str(src), "-vn", "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(a_in)])
    convert(a_in, a_out)
    vd = media_duration(src)
    run_ff(["-i", str(src), "-i", str(a_out), "-filter_complex", "[1:a]aresample=44100,apad[a]", "-map", "0:v",
            "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{vd:.3f}", str(dst)])
    a_in.unlink(missing_ok=True)
    a_out.unlink(missing_ok=True)
    return dst


def merge(clips: list[Path], dialogues: list[str], work: Path, video_cfg: dict, music: Path | None = None,
          say=print, hook: str = "", end_text: str = "", voice: Path | None = None, convert=None) -> Path:
    w, h, fps = int(video_cfg.get("width", 1080)), int(video_cfg.get("height", 1920)), int(video_cfg.get("fps", 30))
    names, words, t0 = [], [], 0.0
    for i, src in enumerate(clips):
        say(f"Clip {i + 1}/{len(clips)} taiyar ho rahi hai...")
        if convert and has_audio(src):
            say(f"Clip {i + 1}: awaaz clone voice mein badli ja rahi hai...")
            src = revoice(src, work / f"vc_{i}.mp4", convert)
        dst = work / f"norm_{i}.mp4"
        dur = _normalize(src, dst, w, h, fps, i, video_cfg)
        names.append(dst.name)
        text = (dialogues[i] if i < len(dialogues) else "").strip()
        if text:
            words += [Word(x.text, x.start + t0, x.end + t0) for x in estimate_words(text, max(dur - 0.3, 0.5))]
        t0 += dur
    (work / "clips.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
    extra = captions.retention_events(hook, t0, w, h, float(video_cfg.get("hook_seconds", 2.5)),
                                      bool(video_cfg.get("progress_bar", True)), end_text)
    ass = (captions.build_ass(words, work / "captions.ass", video_cfg.get("caption", {}), w, h, extra)
           if (words or extra) else None)
    out = work / "reel.mp4"

    if voice:
        vd = media_duration(voice)
        if vd > t0 + 0.3:
            say(f"Dhyan: awaaz ({vd:.0f}s) video ({t0:.0f}s) se lambi hai, aakhir ka hissa kat jayega.")

    def final(with_caps: bool) -> None:
        args = ["-f", "concat", "-safe", "0", "-i", "clips.txt"]
        vf = "subtitles=captions.ass" if (with_caps and ass) else "null"
        nxt = 1
        va = "0:a"  # default: clips ki apni awaaz
        pre = ""
        if voice:  # clone wali awaaz clips ki awaaz ki jagah aati hai
            args += ["-i", str(voice.resolve())]
            pre = f"[{nxt}:a]aresample=44100,loudnorm=I=-16:TP=-1.5:LRA=11,apad[va];"
            va, nxt = "[va]", nxt + 1
        if music:
            args += ["-stream_loop", "-1", "-i", str(music.resolve())]
            vol = float(video_cfg.get("music_volume", 0.12))
            src = va if voice else "[0:a]"
            fc = (f"{pre}[0:v]{vf}[v];[{nxt}:a]volume={vol}[m];{src}[m]amix=inputs=2:duration=first:"
                  "dropout_transition=0,volume=2,alimiter=limit=0.95[a]")
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            fc = f"{pre}[0:v]{vf}[v]"
            maps = ["-map", "[v]", "-map", va if voice else "0:a"]
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
