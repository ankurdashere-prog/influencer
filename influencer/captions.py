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


def build_ass(words: list[Word], path: Path, cap: dict, width: int = 1080, height: int = 1920) -> Path:
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
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path
