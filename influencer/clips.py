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

from . import captions, editor_agent
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
AVATAR_SECONDS = 8  # Flow mein 8 sec ki clip free hai; 4-10 chun sakte ho
_RISKY = re.compile(r"guarantee|100\s*%|permanent|\bcure\b|chamatkar|risk[- ]?free|no side effect|pain[- ]?free|painless",
                    re.I)


def _cut_words(t: str, n: int) -> str:
    """Shabd beech mein kaate bina n characters tak."""
    t = t.strip()
    if len(t) <= n:
        return t
    cut = t[:n].rsplit(" ", 1)[0]
    return cut or t[:n]


# Camera presets (presenter wale hisse ke liye). Halki moves, taaki chehra aur lip-sync na bigde.
CAMERA_PRESETS = {
    "auto": ("Auto (Flow khud chune)", ""),
    "static": ("Static (camera hilega nahi)", "Locked-off static camera, no camera movement."),
    "push_in": ("Slow push-in (dheere paas)", "Very slow smooth push-in toward the face."),
    "pull_out": ("Slow pull-out (dheere door)", "Very slow smooth pull-back, gradually revealing more of the room."),
    "handheld": ("Handheld (natural halka hilna)", "Handheld phone camera with natural, subtle shake."),
    "low_angle": ("Low angle (hero look)", "Low-angle shot looking slightly up at the presenter, steady."),
    "closeup": ("Close-up (chehra bhara)", "Tight close-up on the face and shoulders, steady camera."),
    "arc": ("Halka arc (side se ghoomta)", "Camera drifts in a very slow, slight arc to the left around the presenter."),
}


def _avatar_norm(a: dict | None) -> dict | None:
    if not a or not isinstance(a, dict):
        return None

    def clean(k: str, default: str = "", n: int = 200) -> str:
        return re.sub(r"\s+", " ", str(a.get(k) or default)).strip()[:n]

    try:
        n = int(a.get("scenes") or AVATAR_SCENES)
    except (TypeError, ValueError):
        n = AVATAR_SCENES
    def num(k: str, default: int) -> int:
        try:
            return int(float(a.get(k)))
        except (TypeError, ValueError):
            return default

    sec = min(max(num("seconds", AVATAR_SECONDS), 4), 10)
    pres = min(max(num("presenter", min(PRESENTER_SECONDS, sec - 2)), 2), sec)  # sec = doctor poori clip bolte hain
    return {"name": clean("name", "Avatar", 60), "role": clean("role", "", 80),
            "gender": "female" if str(a.get("gender", "male")).lower().startswith("f") else "male",
            "look": clean("look", "", 300), "language": clean("language", "Hinglish (Hindi in Roman script)", 60),
            "scenes": min(max(n, 1), 6), "seconds": sec, "presenter": pres,
            "camera": str(a.get("camera") or "auto") if str(a.get("camera") or "auto") in CAMERA_PRESETS else "auto"}


def build_avatar_prompt(s: dict, a: dict) -> str:
    he, his = ("He", "His") if a["gender"] == "male" else ("She", "Her")
    look = f" {a['look'].rstrip('. ')}." if a["look"] else ""
    cam = CAMERA_PRESETS.get(a.get("camera", "auto"), CAMERA_PRESETS["auto"])[1]
    cam = f" {cam}" if cam else ""
    return (
        "IDENTITY LOCK (read this fully before creating the video): use ONLY the attached avatar for the person. "
        "Keep the exact same face, hairstyle, facial hair, skin tone, body, outfit, voice, background and lighting "
        f"as the avatar in this clip. Do not change anything about the person.{look}\n"
        f"Format: vertical 9:16, {a['seconds']} seconds, realistic phone-camera look, no subtitles, no text or logos added on screen.\n"
        + (f"Seconds 0-{a['seconds']} (presenter on camera for the whole clip, never leaves the frame): {s['camera_angle']}.{cam} "
           f"{s['action']}. {he} speaks to the camera in {a['language']}, clear natural voice, lips in sync, speaking "
           f"continuously until the last second: \"{s['presenter_line']}\"\n"
           if a["presenter"] >= a["seconds"] else
           f"Seconds 0-{a['presenter']} (presenter on camera): {s['camera_angle']}.{cam} {s['action']}. "
           f"{he} speaks to the camera in {a['language']}, clear natural voice, lips in sync: \"{s['presenter_line']}\"\n"
           f"Seconds {a['presenter']}-{a['seconds']} (cutaway visuals, presenter not on screen): {s['visual']}. "
           f"{his} voiceover continues in the exact same voice and tone: \"{s['voiceover_line']}\"\n")
        + "Same face and same voice in every clip. Wholesome and brand-safe."
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
        *([f"- Each scene is ONE {a['seconds']}-second clip where the presenter speaks to camera for the WHOLE clip. "
           f"presenter_line: between {_line_bounds(a['seconds'], False)[0]} and {_line_bounds(a['seconds'], False)[1]} words "
           "(speech is about 2.4 words per second), one complete idea that finishes inside the clip. voiceover_line: "
           "empty string. (Cutaway visuals are added later by the editor.)",
           "- BOTH lines must be in natural Hinglish (Hindi in Roman script) like a doctor talking in India. Do not write "
           "English sentences; only unavoidable terms (liposuction, surgery, skin) stay in English. Do not say any name."]
          if a["presenter"] >= a["seconds"] else
          [f"- Each scene is ONE {a['seconds']}-second clip: seconds 0-{a['presenter']} the presenter speaks to camera "
           f"(presenter_line), seconds {a['presenter']}-{a['seconds']} cutaway visuals while the same voice continues "
           "(voiceover_line).",
           f"- presenter_line: between {_line_bounds(a['presenter'], False)[0]} and {_line_bounds(a['presenter'], False)[1]} words, "
           f"so the speaker keeps talking for the WHOLE {a['presenter']} seconds (never shorter, no pauses). voiceover_line: "
           f"between {_line_bounds(a['seconds'] - a['presenter'], True)[0]} and {_line_bounds(a['seconds'] - a['presenter'], True)[1]} words "
           "(speech is about 2.4 words per second). One complete idea, sentences must finish inside the scene.",
           "- BOTH lines must be in natural Hinglish (Hindi in Roman script) like a doctor talking in India. Do not write "
           "English sentences; only unavoidable terms (liposuction, surgery, skin) stay in English. Do not say any name."]),
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
        ("Close-up, eye level", "Smiles slightly, no wave",
         f"Namaste, aaj ek zaroori baat {t[:34]} ke baare mein.", "Clean, bright clinic reception, soft daylight, hands arranging papers",
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
    short = ["Sahi jaankari pehla kadam hai.", "Har case alag hota hai.",
             "Doctor se poori baat karein.", "Soch samajh kar kadam uthaiye.",
             "Apna sawal comment karein."]
    use_short = (a["seconds"] - a["presenter"]) < 5.5
    scenes = []
    for i in range(n):
        row = list(base[i % len(base)])
        if use_short:
            row[4] = short[i % len(short)]
        if a["presenter"] >= a["seconds"]:  # doctor poori clip bolte hain: dono baatein ek line mein
            row[2] = f"{row[2].rstrip('.')}. {short[i % len(short)]}"
            row[4] = ""
        elif a["presenter"] < 3.5 and len(row[2].split()) > 7:
            row[2] = ["Ek zaroori baat suniye.", "Ye baat samajh lijiye.", "Pehle sawal poochiye."][i % 3]
        scenes.append(dict(zip(("camera_angle", "action", "presenter_line", "visual", "voiceover_line"), row)))
    return {"topic": t[:80], "hook_text": f"{_cut_words(t, 28)}? JAANIYE", "caption": f"{t}\nAapka kya sawal hai?",
            "hashtags": ["education", "reels"], "scenes": scenes}


_AV_KEYS = ("camera_angle", "action", "presenter_line", "visual", "voiceover_line")


_HINDI_MARK = set("hai hain ki ke ka ko se mein me aur par ya nahi na ho hota hoti hote kar karna karein kijiye "
                  "kijie liye lijiye zaroori sahi jaankari baat apne aap aapka aapko ek koi kuch jo ye wo isse jisme "
                  "bahut sabse pehle phir bhi hi tak sakta sakte samajh poochiye poochein doctor se".split())
_PAD_PRES = ["Doctor se zaroor poochiye.", "Sahi salah zaroori hai.", "Soch samajh kar faisla lijiye."]
_PAD_VO = ["Sahi salah zaroori hai.", "Doctor se baat kijiye.", "Soch samajh kar kadam uthaiye."]


def _is_hinglish(t: str) -> bool:
    w = re.findall(r"[a-z]+", t.lower())
    return len(w) < 3 or any(x in _HINDI_MARK for x in w)


def _end(t: str) -> str:
    t = t.strip().rstrip(",;:-")
    return t if t[-1:] in ".?!" else t + "."


_DANGLE = {"aur", "ki", "ke", "ka", "ko", "se", "mein", "me", "jo", "ya", "par", "ek", "jisme", "toh", "to", "ki"}


def _fit_line(text: str, lo: int, hi: int, pads: list[str]) -> str:
    """Line ko bolne ke time ke hisaab se lo..hi shabdon ke beech rakho (na bahut chhoti, na bahut lambi)."""
    text = _end(text)
    for pad in pads:
        if len(text.split()) >= lo:
            break
        if len(text.split()) + len(pad.split()) <= hi:  # poora vakya hi jodo, aadha nahi
            text = f"{text} {pad}"
    w = text.split()
    if len(w) > hi:
        cut = " ".join(w[:hi])
        for mark in (".", "?", "!", ","):  # pehle poora vakya / comma tak kaato
            i = cut.rfind(mark)
            if i != -1 and len(cut[:i].split()) >= lo:
                cut = cut[:i]
                break
        words = cut.split()
        while len(words) > 3 and words[-1].lower().strip(".,;") in _DANGLE:  # "... aur." jaisa adhura ant nahi
            words.pop()
        text = _end(" ".join(words))
    return text


def _line_bounds(sec: float, voice: bool) -> tuple[int, int]:
    lo_rate = 1.6 if voice else 2.0
    return max(3, round(sec * lo_rate)), max(4, round(sec * 2.4))


def _clean_avatar_scene(s, fb: dict, a: dict | None = None) -> dict:
    s = s if isinstance(s, dict) else {}
    out = {k: re.sub(r"\s+", " ", str(s.get(k) or fb[k])).strip() for k in _AV_KEYS}
    if _RISKY.search(out["presenter_line"] + " " + out["voiceover_line"] + " " + out["visual"]):
        out = dict(fb)  # risky wording: safe template line use karo
    if a:
        full = a["presenter"] >= a["seconds"]  # doctor poori clip bolte hain, cutaway line nahi
        for k in ("presenter_line",) if full else ("presenter_line", "voiceover_line"):
            if not _is_hinglish(out[k]):  # English/ajeeb bhasha: Hinglish template line
                out[k] = fb[k]
        lo, hi = _line_bounds(a["seconds"] if full else a["presenter"], False)
        out["presenter_line"] = _fit_line(out["presenter_line"], lo, hi, _PAD_PRES)
        if full:
            out["voiceover_line"] = ""
        else:
            lo, hi = _line_bounds(a["seconds"] - a["presenter"], True)
            out["voiceover_line"] = _fit_line(out["voiceover_line"], lo, hi, _PAD_VO)
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
        sc = _clean_avatar_scene(src, fb["scenes"][i], a)
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


def _normalize(src: Path, dst: Path, w: int, h: int, fps: int, idx: int = 0, cfg: dict | None = None,
               info: dict | None = None) -> float:
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
    if info is not None:  # B-roll ka "kab se" asli clip ke second se output ke second mein badalne ke liye
        info.update(head=head, speed=sp)
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


def _envelope(path: Path, rate: int = 4000, hop: int = 40) -> list[float]:
    """Awaaz ka volume har 10 ms par (lip-sync ki timing naapne ke liye)."""
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate),
                          "-f", "s16le", "-"], capture_output=True).stdout
    import array
    a = array.array("h")
    a.frombytes(raw[: len(raw) // 2 * 2])
    out = []
    for i in range(0, len(a) - hop, hop):
        seg = a[i:i + hop]
        out.append((sum(x * x for x in seg) / hop) ** 0.5)
    return out


def audio_offset(ref: Path, other: Path, max_s: float = 0.6) -> tuple[float, float]:
    """other awaaz ref se kitni der peeche (+) ya aage (-) hai, seconds mein, aur match kitna pakka hai (0 se 1).
    Dono awaazon ke volume-shape ko 10 ms ke steps par milakar dekhta hai."""
    a, b = _envelope(ref), _envelope(other)
    n = min(len(a), len(b))
    if n < 50:
        return 0.0, 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    a = [x - ma for x in a]
    b = [x - mb for x in b]
    na = (sum(x * x for x in a) or 1.0) ** 0.5
    nb = (sum(x * x for x in b) or 1.0) ** 0.5
    best, best_lag = -2.0, 0
    k = int(max_s * 100)
    for lag in range(-k, k + 1):
        if lag >= 0:
            c = sum(x * y for x, y in zip(a[: n - lag], b[lag:]))
        else:
            c = sum(x * y for x, y in zip(a[-lag:], b[: n + lag]))
        if c > best:
            best, best_lag = c, lag
    return best_lag / 100.0, best / (na * nb)


def revoice(src: Path, dst: Path, convert, say=None) -> Path:
    """Clip ki awaaz nikaal kar convert(audio_in, audio_out) se badalta hai aur wapas video par chipkata hai.
    Video ek frame bhi nahi chhedta, awaaz ki timing wahi rehti hai (lip-sync bacha rehta hai)."""
    a_in, a_out = dst.with_suffix(".in.mp3"), dst.with_suffix(".out.mp3")
    run_ff(["-i", str(src), "-vn", "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k", str(a_in)])
    convert(a_in, a_out)
    vd = media_duration(src)
    shift = 0.0
    try:  # ElevenLabs ki awaaz asli awaaz se thodi aage-peeche ho to hont ke hisaab se wapas milao
        off, conf = audio_offset(a_in, a_out)
        if conf >= 0.5 and 0.04 <= abs(off) <= 0.6:
            shift = off
    except Exception:  # noqa: BLE001 - alignment fail ho to bina shift ke chalo
        shift = 0.0
    if say and shift:
        say(f"Awaaz ka timing {abs(shift) * 1000:.0f} ms {'peeche' if shift > 0 else 'aage'} tha, theek kar diya.")
    shifted = "[1:a]aresample=44100,apad[a]"
    if shift > 0:   # clone awaaz peeche hai: shuru ka hissa kaat do
        shifted = f"[1:a]aresample=44100,atrim=start={shift:.3f},asetpts=PTS-STARTPTS,apad[a]"
    elif shift < 0:  # clone awaaz aage hai: shuru mein khamoshi jodo
        shifted = f"[1:a]aresample=44100,adelay={int(-shift * 1000)}:all=1,apad[a]"
    run_ff(["-i", str(src), "-i", str(a_out), "-filter_complex", shifted, "-map", "0:v",
            "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{vd:.3f}", str(dst)])
    a_in.unlink(missing_ok=True)
    a_out.unlink(missing_ok=True)
    return dst


BROLL_EFFECTS = {"slow_push": "Slow push (dheere zoom)", "zoom_in": "Zoom in", "zoom_out": "Zoom out",
                 "pan_left": "Pan left", "pan_right": "Pan right", "pan_up": "Pan up", "pan_down": "Pan down",
                 "blur_bg": "Poori photo + blur background", "static": "Bina hilne wali"}
BROLL_TRANSITIONS = {"fade": "Fade", "slide": "Slide in", "cut": "Seedha cut"}
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_VID_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
BROLL_EXT = _IMG_EXT | _VID_EXT


def _zoompan(effect: str, n: int) -> tuple[str, str, str]:
    """Photo ke effect ke zoom/x/y (ffmpeg zoompan). `on` = frame number, n = kul frames."""
    n = max(n, 1)
    cx, cy = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    p = f"on/{n}"
    table = {
        "slow_push": (f"1+0.10*{p}", cx, cy),
        "zoom_in": (f"1+0.20*{p}", cx, cy),
        "zoom_out": (f"1.20-0.20*{p}", cx, cy),
        "pan_left": ("1.25", f"(iw-iw/zoom)*(1-{p})", cy),
        "pan_right": ("1.25", f"(iw-iw/zoom)*{p}", cy),
        "pan_up": ("1.25", cx, f"(ih-ih/zoom)*(1-{p})"),
        "pan_down": ("1.25", cx, f"(ih-ih/zoom)*{p}"),
        "static": ("1", cx, cy),
        "blur_bg": (f"1+0.06*{p}", cx, cy),
    }
    return table.get(effect, table["slow_push"])


def _crop_chain(effect: str, length: float, w: int, h: int, fps: int) -> str:
    """Photo ka zoom/pan: bade canvas (1.3x) se time-based crop, phir w x h. (`t` = second)"""
    L = max(length, 0.1)
    p = f"(t/{L:.3f})"
    # crop window ka size (cw, ch) 1.3x canvas ka hissa; z = zoom (1 = poora canvas ka 1/1.3)
    zoom = {"slow_push": f"(1+0.10*{p})", "zoom_in": f"(1+0.20*{p})", "zoom_out": f"(1.20-0.20*{p})",
            "blur_bg": f"(1+0.06*{p})"}.get(effect, "1.15" if effect.startswith("pan") else "1")
    px = {"pan_left": f"(1-{p})", "pan_right": p}.get(effect, "0.5")
    py = {"pan_up": f"(1-{p})", "pan_down": p}.get(effect, "0.5")
    # pehle zoom ke hisaab se scale (frame-wise), phir fixed w x h ka crop (x/y pan ke liye)
    return (f"fps={fps},scale=w='trunc({w}*{zoom}/2)*2':h='trunc({h}*{zoom}/2)*2':eval=frame:flags=bilinear,"
            f"crop={w}:{h}:x='(iw-{w})*{px}':y='(ih-{h})*{py}':exact=1,setsar=1,format=yuva420p")


def _prep_image(src: Path, out: Path, w: int, h: int, blur_bg: bool) -> Path:
    """Photo ko w x h canvas par ek baar taiyar karta hai (cover-crop, ya poori photo + blur background)."""
    from PIL import Image, ImageFilter, ImageOps
    im0 = Image.open(src)
    try:
        im0.draft("RGB", (w, h))  # bade JPEG ko decode karte waqt hi chhota (RAM bachti hai)
    except Exception:
        pass
    im = ImageOps.exif_transpose(im0).convert("RGB")
    if max(im.size) > max(w, h) * 2:
        im.thumbnail((max(w, h) * 2, max(w, h) * 2), Image.LANCZOS)
    cover = ImageOps.fit(im, (w, h), Image.LANCZOS)
    if blur_bg:
        small = cover.resize((max(w // 8, 2), max(h // 8, 2)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
        canvas = small.resize((w, h), Image.BILINEAR)
        fit = ImageOps.contain(im, (w, h), Image.LANCZOS)
        canvas.paste(fit, ((w - fit.width) // 2, (h - fit.height) // 2))
        cover = canvas
    cover.save(out, quality=92)  # JPEG: ffmpeg mein PNG se ~120 MB kam RAM
    return out


def apply_broll(base: Path, dst: Path, media: Path, start: float, end: float | None, effect: str, transition: str,
                w: int, h: int, fps: int, fade: float = 0.3) -> Path:
    """Clip (base) ke video par `start` second se (`end` tak, warna clip ke aakhir tak) photo/video dikhata hai.
    Clip ki awaaz ko haath nahi lagata, isliye doctor ki awaaz (aur lip-sync) wahi rehti hai."""
    dur = media_duration(base)
    start = min(max(float(start), 0.0), max(dur - 0.5, 0.0))
    stop = dur if end is None or end > dur - 0.15 or end < start + 0.5 else float(end)
    length = max(stop - start, 0.5)
    fade = min(fade, length / 3)
    is_img = media.suffix.lower() in _IMG_EXT
    if is_img:
        # Photo ek baar 1.3x canvas par; fir har frame par sirf crop (zoompan se kai guna tez, kam CPU wale server par bhi)
        prepared = _prep_image(media, dst.with_suffix(".bg.jpg"), int(w * 1.3) // 2 * 2, int(h * 1.3) // 2 * 2,
                               effect == "blur_bg")
        chain = _crop_chain(effect, length, w, h, fps)
        src_args = ["-loop", "1", "-framerate", str(fps), "-t", f"{length:.3f}", "-i", str(prepared)]
    else:
        chain = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps={fps},"
                 "format=yuva420p")
        src_args = ["-stream_loop", "-1", "-t", f"{length:.3f}", "-i", str(media)]
    if transition == "fade":
        chain += f",fade=t=in:st=0:d={fade:.2f}:alpha=1"
        if stop < dur - 0.01:  # beech mein khatam ho to wapas doctor par fade
            chain += f",fade=t=out:st={max(length - fade, 0):.2f}:d={fade:.2f}:alpha=1"
    chain += f",setpts=PTS-STARTPTS+{start:.3f}/TB"
    if transition == "slide":
        ox = f"x='if(lt(t,{start:.3f}+{fade:.2f}),W*(1-(t-{start:.3f})/{fade:.2f}),0)':y=0:"
    else:
        ox = ""
    fc = (f"[1:v]{chain}[b];[0:v][b]overlay={ox}enable='between(t,{start:.3f},{stop:.3f})':eof_action=pass:"
          "repeatlast=0[v]")
    run_ff(["-i", str(base), *src_args, "-filter_complex", fc, "-map", "[v]", "-map", "0:a?", "-c:a", "copy",
            "-threads", "1", "-filter_threads", "1", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
            "-pix_fmt", "yuv420p", "-t", f"{dur:.3f}", str(dst)], timeout=240)
    if is_img:
        dst.with_suffix(".bg.jpg").unlink(missing_ok=True)
    return dst


def merge(clips: list[Path], dialogues: list[str], work: Path, video_cfg: dict, music: Path | None = None,
          say=print, hook: str = "", end_text: str = "", voice: Path | None = None, convert=None,
          broll: dict | None = None, auto_zoom: bool = False) -> Path:
    w, h, fps = int(video_cfg.get("width", 1080)), int(video_cfg.get("height", 1920)), int(video_cfg.get("fps", 30))
    names, words, t0 = [], [], 0.0
    events: list[float] = []
    if convert:
        silent = [c.name for c in clips if not has_audio(c)]
        say("Awaaz: ElevenLabs clone voice (har clip ki awaaz badli jayegi)."
            + (f" Bina awaaz wali clips: {len(silent)} (inki awaaz nahi badlegi)." if silent else ""))
    elif voice:
        say("Awaaz: Voice Lab ki alag audio (clips ki awaaz hatai jayegi).")
    else:
        say("Awaaz: clip ki original (Flow wali) awaaz. Clone voice ke liye dropdown mein 'Badlo' chuno.")
    for i, src in enumerate(clips):
        say(f"Clip {i + 1}/{len(clips)} taiyar ho rahi hai...")
        if convert and has_audio(src):
            say(f"Clip {i + 1}: awaaz clone voice mein badli ja rahi hai...")
            src = revoice(src, work / f"vc_{i}.mp4", convert, say)
        dst = work / f"norm_{i}.mp4"
        info: dict = {}
        dur = _normalize(src, dst, w, h, fps, i, video_cfg, info)
        if i > 0:
            events.append(t0)  # clip ka cut
        if auto_zoom and dur >= editor_agent.MIN_ZOOM_DUR:
            say(f"Clip {i + 1}: doctor ke hisse mein halka zoom in/out laga raha hu...")
            try:
                dst = editor_agent.apply_micro_zoom(dst, work / f"zm_{i}.mp4", w, h, dur)
            except Exception as e:
                say(f"Clip {i + 1}: micro-zoom nahi lag paya ({str(e)[:100]}), bina zoom ke aage badh raha hu.")
                auto_zoom_i = False
            else:
                auto_zoom_i = True
        else:
            auto_zoom_i = False
        skip = None
        bl = (broll or {}).get(i)
        if bl:
            head, sp = info.get("head", 0.0), info.get("speed", 1.0) or 1.0
            b_start = (float(bl["start"]) - head) / sp
            b_end = None if bl.get("end") is None else (float(bl["end"]) - head) / sp
            if b_start >= dur - 0.5:
                say(f"Clip {i + 1}: B-roll ka start ({bl['start']}s) clip ke aakhir ke bahut paas hai, chhod diya.")
            else:
                say(f"Clip {i + 1}: apni photo/video {max(bl['start'], 0):g} second se laga raha hu...")
                out_b = work / f"br_{i}.mp4"
                try:
                    apply_broll(dst, out_b, bl["path"], max(b_start, 0.0), b_end, bl.get("effect", "slow_push"),
                                bl.get("transition", "fade"), w, h, fps)
                    dst = out_b
                    bs = max(b_start, 0.0)
                    be = dur if (b_end is None or b_end > dur - 0.15 or b_end < bs + 0.5) else b_end
                    skip = (bs, be)
                    events += [t0 + bs, t0 + be]
                except Exception as e:  # B-roll fail ho to reel bina B-roll ke aage badhe
                    say(f"Clip {i + 1}: photo/video lagane mein dikkat ({str(e)[:120]}), bina B-roll ke aage badh raha hu.")
        if auto_zoom_i:
            events += editor_agent.zoom_events(t0, dur, skip=skip)
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
    try:  # retention check: hamare lagaye badlav + video mein asli scene-cuts
        ev = events + [c for c in editor_agent.detect_cuts(out) if c < t0]
        rep = editor_agent.retention_report(ev, t0)
        for line in editor_agent.format_report(rep):
            say(line)
        (work / "retention.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        say(f"Retention check nahi ho paya ({str(e)[:80]}).")
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
