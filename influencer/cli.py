from __future__ import annotations

import argparse
import shutil
import sys

from . import images, pipeline
from .config import REFERENCE_IMAGE, ROOT, env, persona


def cmd_doctor(_args) -> int:
    from .video import ffmpeg_exe

    ok = True

    def line(name: str, good: bool, hint: str = "") -> None:
        nonlocal ok
        print(("  OK   " if good else "  MISSING ") + name + ("" if good else f"  -> {hint}"))
        ok = ok and good

    print("Setup check:")
    try:
        line("ffmpeg", bool(ffmpeg_exe()))
    except RuntimeError as e:
        line("ffmpeg", False, str(e))
    line("GEMINI_API_KEY (script/caption)", bool(env("GEMINI_API_KEY")), "free key: aistudio.google.com/apikey (bina key ke template use hoga)")
    prov = env("IMAGE_PROVIDER", "openai")
    line(f"image provider = {prov}", prov != "openai" or bool(env("OPENAI_API_KEY")), "OPENAI_API_KEY bharo ya IMAGE_PROVIDER=pollinations")
    line("character/reference.png", REFERENCE_IMAGE.exists(), "python -m influencer init-character")
    line("IG_USER_ID + IG_ACCESS_TOKEN", bool(env("IG_USER_ID") and env("IG_ACCESS_TOKEN")), "posting ke liye zaroori")
    line("SUPABASE_URL + SUPABASE_SERVICE_KEY", bool(env("SUPABASE_URL") and env("SUPABASE_SERVICE_KEY")), "posting ke liye zaroori (public video URL)")
    print("\nSab OK" if ok else "\nKuch cheezein baaki hain (upar dekho). Bina posting ke tool phir bhi chal sakta hai.")
    return 0


def cmd_init_character(args) -> int:
    who = persona()
    out_dir = ROOT / "character" / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = "placeholder" if args.demo else None
    base = "front-facing upper-body portrait, plain light background, relaxed natural expression"
    for i in range(args.n):
        p = out_dir / f"candidate_{i + 1}.png"
        images.generate(images.full_prompt(who, base), p, seed=100 + i * 7, ref=None, provider=provider)
        print(f"  banayi: {p}")
    print(f"\nAb {out_dir} mein se jo chehra pasand aaye use copy karke\n  {REFERENCE_IMAGE}\nnaam se save karo. "
          "Phir har post isi chehre ke saath banega.")
    return 0


def cmd_make(args) -> int:
    kind = args.kind
    brand = args.brand
    if kind == "ad" and not brand:
        b = pipeline.next_brand()
        brand = str(b) if b else None
    pipeline.run(kind, brand, post=bool(args.post), demo=args.demo, topic=args.topic)
    return 0


def cmd_auto(args) -> int:
    pipeline.run_auto(post=bool(args.post), demo=args.demo, source="cron")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    print(f"Browser mein kholo: http://{args.host}:{args.port}")
    uvicorn.run("influencer.web.app:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="influencer", description="AI influencer automation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="setup check").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("init-character", help="chehre ke candidates banao")
    p.add_argument("-n", type=int, default=4)
    p.add_argument("--demo", action="store_true")
    p.set_defaults(fn=cmd_init_character)

    p = sub.add_parser("make", help="ek post banao (aur --post se publish)")
    p.add_argument("--kind", choices=["reel", "photo", "ad"], default="reel")
    p.add_argument("--brand", help="brands/<file>.yaml (ad ke liye)")
    p.add_argument("--topic", help="topic khud do (optional)")
    p.add_argument("--post", action="store_true", help="Instagram par publish karo")
    p.add_argument("--demo", action="store_true", help="bina API keys ke test (placeholder images, silent voice)")
    p.set_defaults(fn=cmd_make)

    p = sub.add_parser("auto", help="aaj ka planned post (cron ke liye)")
    p.add_argument("--post", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.set_defaults(fn=cmd_auto)

    p = sub.add_parser("serve", help="web dashboard chalao (browser mein)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(env("PORT", "7860")))  # Render PORT env deta hai
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    sys.exit(args.fn(args))
