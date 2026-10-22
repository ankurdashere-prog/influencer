"""FFmpeg: images -> zoom/pan clips -> captions + voice + music -> 9:16 reel. CPU only, low RAM."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg nahi mila. `pip install imageio-ffmpeg` ya ffmpeg install karo.") from e


def run_ff(args: list[str], cwd: Path | None = None) -> None:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-900:] or "ffmpeg failed")


def media_duration(path: Path) -> float:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not m:
        raise RuntimeError(f"Duration nahi mili: {path}")
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


def _motion(idx: int, frames: int) -> tuple[str, str, str]:
    f = max(frames, 1)
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    modes = [
        (f"1+0.16*on/{f}", cx, cy),                              # slow zoom in
        (f"1.16-0.16*on/{f}", cx, cy),                           # slow zoom out
        ("1.12", f"(iw-iw/zoom)*on/{f}", cy),                    # pan left -> right
        ("1.12", f"(iw-iw/zoom)*(1-on/{f})", cy),                # pan right -> left
    ]
    return modes[idx % len(modes)]


def make_clip(image: Path, out: Path, dur: float, idx: int, w: int, h: int, fps: int, oversample: float = 1.25) -> None:
    frames = max(int(round(dur * fps)), 1)
    bw, bh = int(w * oversample) // 2 * 2, int(h * oversample) // 2 * 2
    z, x, y = _motion(idx, frames)
    vf = (
        f"scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps},format=yuv420p"
    )
    run_ff(["-i", str(image), "-vf", vf, "-frames:v", str(frames), "-r", str(fps), "-an",
            "-threads", "1", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-x264-params", "rc-lookahead=8:ref=2", "-pix_fmt", "yuv420p", str(out)])


def make_talk_clip(src: Path, out: Path, dur: float, w: int, h: int, fps: int) -> None:
    """Bolte video ko 9:16 mein fit karta hai (audio hata deta hai, asli voice baad mein judti hai)."""
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,format=yuv420p"
    run_ff(["-i", str(src), "-vf", vf, "-t", f"{dur:.2f}", "-r", str(fps), "-an",
            "-threads", "1", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-x264-params", "rc-lookahead=8:ref=2", "-pix_fmt", "yuv420p", str(out)])


def assemble_reel(images: list[Path], audio: Path, ass: Path | None, out: Path, work: Path,
                  voice_dur: float, video_cfg: dict, music: Path | None = None,
                  talking: Path | None = None, talk_dur: float = 0.0) -> Path:
    w, h, fps = int(video_cfg.get("width", 1080)), int(video_cfg.get("height", 1920)), int(video_cfg.get("fps", 30))
    total = voice_dur + float(video_cfg.get("tail_seconds", 0.6))
    clips = []
    if talking and talk_dur > 0:
        # pehla hissa: bolta chehra (lip-sync), baaki: images (cutaway) jab tak voice chalti hai
        talk_dur = min(talk_dur, media_duration(talking), total - 1.0)
        c = work / "clip_talk.mp4"
        make_talk_clip(talking, c, talk_dur, w, h, fps)
        clips.append(c.name)
        rest = images[1:] or images[:1]
        per = max((total - talk_dur) / len(rest), 1.5)
        pool = rest
    else:
        per = max(total / len(images), 1.5)
        pool = images
    for i, img in enumerate(pool):
        c = work / f"clip_{i}.mp4"
        make_clip(img, c, per, i, w, h, fps, float(video_cfg.get("oversample", 1.25)))
        clips.append(c.name)
    (work / "clips.txt").write_text("".join(f"file '{c}'\n" for c in clips), encoding="utf-8")

    def final(with_caps: bool) -> None:
        args = ["-f", "concat", "-safe", "0", "-i", "clips.txt", "-i", str(audio.resolve())]
        vf = f"subtitles={ass.name}" if (with_caps and ass) else "null"
        if music:
            args += ["-stream_loop", "-1", "-i", str(music.resolve())]
            vol = float(video_cfg.get("music_volume", 0.12))
            fc = (f"[0:v]{vf}[v];[2:a]volume={vol}[m];[1:a][m]amix=inputs=2:duration=first:"
                  "dropout_transition=0,volume=2,alimiter=limit=0.95[a]")
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            fc = f"[0:v]{vf}[v]"
            maps = ["-map", "[v]", "-map", "1:a"]
        args += ["-filter_complex", fc, *maps, "-t", f"{total:.2f}", "-r", str(fps),
                 "-threads", "1", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                 "-x264-params", "rc-lookahead=8:ref=2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out.resolve())]
        run_ff(args, cwd=work)

    try:
        final(True)
    except RuntimeError as e:
        if ass and ("subtitles" in str(e).lower() or "filter" in str(e).lower()):
            print("[video] Captions burn nahi ho paye (ffmpeg mein libass nahi). Bina captions ke bana raha hu.")
            final(False)
        else:
            raise
    return out
