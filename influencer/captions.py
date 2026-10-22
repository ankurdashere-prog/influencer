"""Word-by-word highlight captions (ASS subtitle file)."""
from __future__ import annotations

from pathlib import Path

from .tts import Word

_COLORS = {"yellow": "&H0000FFFF&", "green": "&H0000FF66&", "pink": "&H00B469FF&", "white": "&H00FFFFFF&"}


def _t(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _clean(w: str) -> str:
    return w.replace("{", "").replace("}", "").replace("\\", "").replace("\n", " ")


def retention_events(hook: str, total: float, width: int = 1080, height: int = 1920, hook_seconds: float = 2.5,
                     progress_bar: bool = True, end_text: str = "", end_seconds: float = 2.0) -> list[str]:
    """Retention tricks: shuru mein hook text (pehle 2.5s), neeche progress bar, aakhir mein follow text."""
    ev: list[str] = []

    def wrap(t: str) -> str:
        ws = _clean(t.strip()).split()
        return "\\N".join(" ".join(ws[i:i + 3]) for i in range(0, len(ws), 3))

    hook = (hook or "").strip()
    if hook and total > 1:
        ev.append(f"Dialogue: 2,{_t(0)},{_t(min(hook_seconds, total))},Hook,,0,0,0,,"
                  f"{{\\fad(0,300)\\fscx92\\fscy92\\t(0,250,\\fscx100\\fscy100)}}{wrap(hook.upper()[:60])}")
    end_text = (end_text or "").strip()
    if end_text and total > end_seconds + hook_seconds:
        ev.append(f"Dialogue: 2,{_t(total - end_seconds)},{_t(total)},Hook,,0,0,0,,"
                  f"{{\\fad(250,0)}}{wrap(end_text[:60])}")
    if progress_bar and total > 1:
        bh = max(height // 120, 10)
        ev.append(f"Dialogue: 1,{_t(0)},{_t(total)},Cap,,0,0,0,,{{\\an7\\pos(0,{height - bh})\\bord0\\shad0"
                  f"\\1c&H00FFFFFF&\\1a&H50&\\fscx0\\t(0,{int(total * 1000)},\\fscx100)\\p1}}"
                  f"m 0 0 l {width} 0 {width} {bh} 0 {bh}{{\\p0}}")
    return ev


def build_ass(words: list[Word], path: Path, cap: dict, width: int = 1080, height: int = 1920,
              extra: list[str] | None = None) -> Path:
    group_n = int(cap.get("words_per_group", 3))
    hi = _COLORS.get(cap.get("highlight", "yellow"), _COLORS["yellow"])
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Cap,{cap.get('font', 'Arial')},{int(cap.get('size', 84))},&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,{int(cap.get('margin_bottom', 460))},1
Style: Hook,{cap.get('font', 'Arial')},{int(cap.get('hook_size', 92))},&H0000FFFF&,&H00FFFFFF&,&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,8,3,8,50,50,260,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        cur.append(w)
        if len(cur) >= group_n or w.text[-1:] in ".?!":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    lines = []
    for gi, g in enumerate(groups):
        next_start = groups[gi + 1][0].start if gi + 1 < len(groups) else g[-1].end + 0.4
        for i, w in enumerate(g):
            start = w.start
            end = g[i + 1].start if i + 1 < len(g) else min(g[-1].end + 0.25, next_start)
            if end <= start:
                end = start + 0.1
            text = " ".join(
                ("{\\c" + hi + "}" + _clean(x.text) + "{\\c&H00FFFFFF&}") if j == i else _clean(x.text)
                for j, x in enumerate(g)
            )
            lines.append(f"Dialogue: 0,{_t(start)},{_t(end)},Cap,,0,0,0,,{text}")
    lines += extra or []
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path
