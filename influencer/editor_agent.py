"""Editor agent: 10 saal ke short-video editor ki tarah pacing ka plan banata hai, doctor ke lambe hisse mein
halka micro-zoom lagata hai, aur reel banne ke baad "kitni der ek hi frame raha" ki jaanch (retention check) karta hai.

Ye rule-based hai (LLM nahi): har faisle ke saath wajah likhi hoti hai, taaki user card dekhkar approve kar sake.
Rules ek aam short-form thumb rule par hain (hook pehle ~3 s, har ~3 s mein kuch badle, aakhir mein CTA). Ye
guideline hai, platform ka pakka niyam nahi."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .video import ffmpeg_exe, run_ff

ZOOM_AMP = 0.06        # zoom 1.00 se 1.06 tak (halka, chehra/lip-sync nahi bigadta)
ZOOM_PERIOD = 3.0      # har itne second par zoom ki disha badalti hai
MAX_STILL = 4.0        # isse zyada der ek hi frame ho to warning
MIN_ZOOM_DUR = 4.5     # isse chhoti clip par micro-zoom nahi

_EFFECTS = ["slow_push", "zoom_in", "pan_right", "zoom_out", "pan_left"]


def _r(x: float) -> float:
    return round(float(x), 1)


def plan_clip(idx: int, n: int, seconds: float, presenter: float, has_broll: bool, pace: float = ZOOM_PERIOD) -> dict:
    """Ek clip ka plan. Time 'source clip ke second' mein (Flow ki asli clip, bina trim ke)."""
    S = float(seconds)
    last = idx == n - 1
    notes: list[str] = []
    hook = 3.0 if S >= 7 else (2.0 if S >= 5 else 1.5)
    tail = (2.5 if last else 2.0) if S >= 7 else 1.5
    segs: list[dict] = []
    broll = None
    full = presenter >= S - 0.01

    if not full:
        segs.append({"start": 0.0, "end": _r(presenter), "kind": "doctor", "label": "Doctor camera par"})
        segs.append({"start": _r(presenter), "end": _r(S), "kind": "cutaway", "label": "Cutaway visuals (Flow se)"})
        notes.append("Is clip mein cutaway Flow ki taraf se aa chuka hai, isliye alag photo ki zaroorat nahi.")
    else:
        b0, b1 = hook, S - tail
        if b1 - b0 > 3.5:
            b1 = b0 + 3.2
        if b1 - b0 >= 1.5:
            e = _EFFECTS[idx % len(_EFFECTS)]
            broll = {"start": _r(b0), "end": _r(b1), "effect": e, "transition": "fade"}
            segs += [{"start": 0.0, "end": _r(b0), "kind": "doctor", "label": "Hook: doctor seedha camera par"},
                     {"start": _r(b0), "end": _r(b1), "kind": "broll", "label": "Aapki photo/video (B-roll)"},
                     {"start": _r(b1), "end": _r(S), "kind": "doctor",
                      "label": "Doctor wapas" + (" + end CTA" if last else "")}]
            notes.append(f"Pehle {hook:g} s doctor (hook), phir {b1 - b0:.1f} s B-roll, phir doctor"
                         + (" + CTA." if last else "."))
        else:
            segs.append({"start": 0.0, "end": _r(S), "kind": "doctor", "label": "Doctor poori clip"})
            notes.append("Clip chhoti hai, B-roll ki jagah nahi bani; sirf micro-zoom se change aayega.")
    if not has_broll and full and broll:
        notes.append("Is clip par abhi photo/video attach nahi hai. Attach karoge to plan wahi timing bharega.")

    # micro-zoom ke mod (reversal) points: sirf doctor ke hisson mein
    zp: list[float] = []
    if S >= MIN_ZOOM_DUR:
        t = pace
        while t < S - 0.6:
            inside_b = broll and broll["start"] - 0.4 <= t <= broll["end"] + 0.4
            if not inside_b:
                zp.append(_r(t))
            t += pace
    conf = 0.9 if (broll and has_broll) or not full else 0.7
    return {"clip": idx + 1, "seconds": S, "segments": segs, "broll": broll, "zoom_points": zp,
            "confidence": conf, "notes": notes}


def plan_reel(n: int, seconds: float, presenter: float, has_broll: list[bool] | None = None) -> dict:
    hb = list(has_broll or [])
    hb += [False] * (n - len(hb))
    clips = [plan_clip(i, n, seconds, presenter, bool(hb[i])) for i in range(n)]
    total = seconds * n
    changes = 0
    for c in clips:
        changes += len(c["zoom_points"]) + (2 if c["broll"] else 0) + 1
    per10 = round(changes / max(total, 1) * 10, 1)
    low = [c["clip"] for c in clips if c["confidence"] < 0.85]
    summary = (f"{n} clip, kul ~{total:g} s. Har ~3 s par kuch badlega (hook, B-roll, zoom, clip cut); "
               f"lagbhag {per10:g} badlav har 10 s mein.")
    if low:
        summary += f" Clip {', '.join(map(str, low))} par photo/video attach karoge to plan poora bharega."
    return {"clips": clips, "summary": summary, "changes_per_10s": per10}


def zoom_filter(w: int, h: int, amp: float = ZOOM_AMP, period: float = ZOOM_PERIOD) -> str:
    """Triangle wave: 1.00 -> 1+amp (period s) -> 1.00 ... Frame-wise scale, phir fixed w x h crop."""
    z = f"(1+{amp}*(1-abs(mod(t,{2 * period})-{period})/{period}))"
    return (f"scale=w='trunc({w}*{z}/2)*2':h='trunc({h}*{z}/2)*2':eval=frame:flags=bilinear,"
            f"crop={w}:{h}:x='(iw-{w})/2':y='(ih-{h})/2':exact=1,setsar=1")


def apply_micro_zoom(src: Path, dst: Path, w: int, h: int, dur: float, amp: float = ZOOM_AMP,
                     period: float = ZOOM_PERIOD) -> Path:
    """Clip par halka zoom in/out (awaaz jyon ki tyon)."""
    run_ff(["-i", str(src), "-vf", zoom_filter(w, h, amp, period), "-map", "0:v", "-map", "0:a?", "-c:a", "copy",
            "-threads", "1", "-filter_threads", "1", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
            "-pix_fmt", "yuv420p", "-t", f"{dur:.3f}", str(dst)], timeout=240)
    return dst


def zoom_events(t0: float, dur: float, period: float = ZOOM_PERIOD, skip: tuple[float, float] | None = None) -> list[float]:
    """Reel ke time mein zoom ki disha badalne ke pal (B-roll wali window chhodkar)."""
    out, t = [], period
    while t < dur - 0.4:
        if not (skip and skip[0] - 0.3 <= t <= skip[1] + 0.3):
            out.append(t0 + t)
        t += period
    return out


def detect_cuts(path: Path, thr: float = 0.12) -> list[float]:
    """Asli video mein scene-change ke pal (ffmpeg scene detect). Fail ho to khaali list."""
    try:
        p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostats", "-i", str(path), "-an", "-vf",
                            f"scale=180:-2,select='gt(scene,{thr})',showinfo", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=120)
    except Exception:
        return []
    return [float(x) for x in re.findall(r"pts_time:(\d+(?:\.\d+)?)", p.stderr)]


def retention_report(events: list[float], total: float, max_still: float = MAX_STILL) -> dict:
    """events = jin palon par screen par kuch badla. Sabse lambi 'ek hi frame' wali der nikalta hai."""
    pts = sorted({round(e, 1) for e in [0.0, *events] if 0 <= e < total} | {round(total, 1)})
    merged: list[float] = []
    for e in pts:
        if not merged or e - merged[-1] >= 0.3:
            merged.append(e)
    gaps = [(merged[i], merged[i + 1], merged[i + 1] - merged[i]) for i in range(len(merged) - 1)]
    longest = max(gaps, key=lambda g: g[2]) if gaps else (0.0, total, total)
    bad = [g for g in gaps if g[2] > max_still]
    tips = []
    for a, b, d in bad:
        tips.append(f"{a:.0f}-{b:.0f} s ({d:.1f} s) tak ek hi frame hai. Yahan B-roll/photo, text ya zoom lagao.")
    return {"events": merged[:-1], "total": round(total, 1), "longest_still": round(longest[2], 1),
            "longest_at": [round(longest[0], 1), round(longest[1], 1)], "changes": max(len(merged) - 1, 0),
            "per_10s": round(max(len(merged) - 1, 0) / max(total, 1) * 10, 1),
            "verdict": "warn" if bad else "ok", "tips": tips}


def format_report(r: dict, max_still: float = MAX_STILL) -> list[str]:
    head = (f"Retention check: sabse lambi ek-hi-frame der {r['longest_still']:g} s "
            f"({r['longest_at'][0]:g}-{r['longest_at'][1]:g} s), {r['changes']} badlav, ~{r['per_10s']:g} har 10 s mein.")
    if r["verdict"] == "ok":
        return [head, f"Retention check: theek hai (koi hissa {max_still:g} s se zyada ek frame par nahi)."]
    return [head, "Retention check WARNING:", *["  - " + t for t in r["tips"]]]
