"""Web dashboard backend (FastAPI). Password-protected, ek time par ek job (kam RAM)."""
from __future__ import annotations

import base64
import hmac
import json
import queue
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import clips as clipstudio, images, persist, pipeline
from ..config import (IST, OUTPUT_DIR, REFERENCE_IMAGE, ROOT, env, load_history, now_ist)

STATIC = Path(__file__).parent / "static"
CAND_DIR = ROOT / "character" / "candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CAND_DIR.mkdir(parents=True, exist_ok=True)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")

# ---------------- job queue (ek saath ek hi kaam) ----------------
jobs: dict[str, dict] = {}
_q: "queue.Queue[tuple[str, object]]" = queue.Queue()


def _worker() -> None:
    while True:
        jid, fn = _q.get()
        job = jobs[jid]
        job["status"] = "running"
        pipeline.progress = lambda m, _j=job: _j["log"].append(str(m))
        try:
            job["result"] = fn()
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 - user ko error dikhana hai
            job["status"] = "error"
            job["error"] = str(e)[:700]
            job["log"].append("Error: " + str(e)[:300])
        finally:
            job["finished"] = time.time()
            for old in sorted(jobs, key=lambda k: jobs[k]["created"])[:-40]:
                jobs.pop(old, None)


def submit(label: str, fn) -> str:
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"id": jid, "label": label, "status": "queued", "log": [], "result": None,
                 "error": None, "created": time.time()}
    _q.put((jid, fn))
    return jid


@asynccontextmanager
async def lifespan(_app: FastAPI):
    n = persist.pull()
    if n:
        print(f"[persist] {n} saved files wapas aayi")
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="AI Influencer", lifespan=lifespan)


# ---------------- auth ----------------
@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path == "/healthz" or request.url.path.startswith("/api/cron/"):
        return await call_next(request)  # token endpoint ke andar check hota hai
    pw = env("APP_PASSWORD")
    if not pw:
        if env("SPACE_ID") or env("REQUIRE_PASSWORD"):
            return Response("Server par APP_PASSWORD secret set karo, tabhi dashboard khulega.", status_code=503)
        return await call_next(request)  # local use (127.0.0.1)
    ok = False
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("basic "):
        try:
            given = base64.b64decode(auth[6:]).decode("utf-8").partition(":")[2]
            ok = hmac.compare_digest(given.encode(), pw.encode())
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        return Response("Login chahiye", status_code=401, headers={"WWW-Authenticate": 'Basic realm="AI Influencer"'})
    return await call_next(request)


# ---------------- helpers ----------------
def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _item(d: Path) -> dict | None:
    res = _read_json(d / "result.json")
    if not res:
        return None
    plan = _read_json(d / "plan.json")
    base = f"/files/{d.name}"
    return {**res, "caption": plan.get("caption", ""), "voiceover": plan.get("voiceover", ""),
            "url": f"{base}/{res.get('file')}", "poster": f"{base}/scene_1.png"}


def _safe_dir(name: str) -> Path:
    d = (OUTPUT_DIR / name).resolve()
    if OUTPUT_DIR.resolve() not in d.parents or not d.is_dir():
        raise HTTPException(404, "Item nahi mila")
    return d


def _validate_yaml(text: str) -> None:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"YAML galat hai: {str(e)[:200]}") from e
    if not isinstance(data, dict):
        raise HTTPException(400, "YAML ek key: value wali file honi chahiye")


CONFIG_FILES = {"persona": ROOT / "character" / "persona.yaml", "settings": ROOT / "settings.yaml"}


# ---------------- API ----------------
@app.get("/api/status")
def status():
    def has(*keys):
        return all(bool(env(k)) for k in keys)

    day = now_ist().strftime("%a").lower()[:3]
    plan = pipeline.settings().get("weekly_plan", {}).get(day, "reel")
    return {
        "checks": [
            {"name": "Gemini key (script/caption)", "ok": has("GEMINI_API_KEY"), "hint": "GEMINI_API_KEY secret. Bina iske simple template use hota hai."},
            {"name": "Image provider", "ok": env("IMAGE_PROVIDER", "openai") != "openai" or has("OPENAI_API_KEY"), "hint": "OPENAI_API_KEY secret ya IMAGE_PROVIDER=pollinations"},
            {"name": "Character reference", "ok": REFERENCE_IMAGE.exists(), "hint": "Character tab mein chehra chuno"},
            {"name": "Instagram (ID + token)", "ok": has("IG_USER_ID", "IG_ACCESS_TOKEN"), "hint": "IG_USER_ID, IG_ACCESS_TOKEN secrets"},
            {"name": "Video hosting (Supabase)", "ok": has("SUPABASE_URL", "SUPABASE_SERVICE_KEY"), "hint": "SUPABASE_URL, SUPABASE_SERVICE_KEY. Isse data bhi restart ke baad bachta hai."},
            {"name": "Daily auto-post token", "ok": has("CRON_TOKEN"), "hint": "CRON_TOKEN secret (roz ki automation ke liye)"},
        ],
        "today": {"day": day, "plan": plan},
        "persist": persist.enabled(),
        "image_provider": env("IMAGE_PROVIDER", "openai"),
        "busy": any(j["status"] in ("queued", "running") for j in jobs.values()),
    }


@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    kind = body.get("kind", "reel")
    if kind not in ("reel", "photo", "ad"):
        raise HTTPException(400, "kind galat hai")
    brand = body.get("brand")
    brand_path = None
    if kind == "ad":
        if brand and NAME_RE.match(brand) and (ROOT / "brands" / f"{brand}.yaml").exists():
            brand_path = str(ROOT / "brands" / f"{brand}.yaml")
        else:
            raise HTTPException(400, "Ad ke liye brand chuno")
    topic = (body.get("topic") or "").strip()[:200] or None
    demo = bool(body.get("demo"))
    ref_url = (body.get("ref_url") or "").strip()[:300] or None
    if ref_url and kind != "reel":
        raise HTTPException(400, "Reference link sirf Reel ke saath chalta hai")
    jid = submit(f"{kind}", lambda: pipeline.run(kind, brand_path, post=False, demo=demo, topic=topic, source="web",
                                                 ref_url=ref_url))
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "Job nahi mila")
    return j


@app.get("/api/items")
def items():
    out = []
    for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if d.is_dir():
            it = _item(d)
            if it:
                out.append(it)
    return out[:60]


@app.post("/api/items/{name}/post")
def post_item(name: str):
    _safe_dir(name)
    return {"job": submit("post", lambda: {"ig_media_id": pipeline.post_output(name)})}


@app.delete("/api/items/{name}")
def delete_item(name: str):
    shutil.rmtree(_safe_dir(name), ignore_errors=True)
    return {"ok": True}







# ---- clip studio (scene prompts -> aapke clips -> reel) ----
@app.post("/api/clips/script")
async def clips_script(request: Request):
    body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    topic = (body.get("topic") or "").strip()[:200] or None
    offline = not (env("GEMINI_API_KEY") or env("OPENAI_API_KEY"))
    return await run_in_threadpool(clipstudio.make_scene_script, topic, pipeline.persona(), clipstudio.N_SCENES, offline)


@app.post("/api/clips/merge")
async def clips_merge(clips: list[UploadFile] = File(...), dialogues: str = Form("[]"),
                      topic: str = Form("Clip Studio reel"), caption: str = Form(""), music: str = Form("1"),
                      speed: str = Form("1.1"), music_volume: str = Form("0.12"),
                      music_file: UploadFile | None = File(None)):
    if not clips or len(clips) > 12:
        raise HTTPException(400, "1 se 12 clips chahiye")
    try:
        dlg = [str(x) for x in json.loads(dialogues)]
    except (json.JSONDecodeError, TypeError):
        dlg = []
    work = clipstudio.new_work_dir(topic)
    paths = []
    for i, f in enumerate(clips):
        data = await f.read()
        if len(data) > 80_000_000:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(400, f"Clip {i + 1} 80 MB se badi hai")
        if len(data) < 1000:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(400, f"Clip {i + 1} khaali ya kharab hai")
        p = work / f"in_{i}.mp4"
        p.write_bytes(data)
        paths.append(p)

    def _num(v, lo, hi, default):
        try:
            return min(max(float(v), lo), hi)
        except (TypeError, ValueError):
            return default

    cfg = dict(pipeline.settings()["video"], clip_speed=_num(speed, 0.8, 1.5, 1.1),
               music_volume=_num(music_volume, 0.0, 0.6, 0.12))
    mus = None
    if music_file is not None and music_file.filename:
        ext = Path(music_file.filename).suffix.lower()
        mdata = await music_file.read()
        if ext not in (".mp3", ".m4a", ".wav", ".aac", ".ogg"):
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(400, "Music mp3, m4a, wav, aac ya ogg hona chahiye")
        if len(mdata) > 25_000_000 or len(mdata) < 1000:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(400, "Music file 25 MB se chhoti honi chahiye")
        mus = work / f"music_in{ext}"
        mus.write_bytes(mdata)
    elif music == "1":
        mus = clipstudio.pick_music(ROOT)

    def work_fn():
        final = clipstudio.merge(paths, dlg, work, cfg, mus, say=pipeline._say)
        for q in paths + ([mus] if mus else []):
            q.unlink(missing_ok=True)
        res = clipstudio.save_item(work, topic[:80], caption or topic, final)
        pipeline._say("Reel tayyar. Library mein dekho.")
        return res

    return {"job": submit("clips", work_fn)}








# ---- character ----
@app.get("/api/character")
def character():
    cands = sorted(p.name for p in CAND_DIR.glob("*.png"))
    return {"reference": REFERENCE_IMAGE.exists(), "ref_mtime": int(REFERENCE_IMAGE.stat().st_mtime) if REFERENCE_IMAGE.exists() else 0,
            "candidates": cands}


@app.post("/api/character/candidates")
async def make_candidates(request: Request):
    body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    n = max(1, min(int(body.get("n", 4)), 6))
    demo = bool(body.get("demo"))
    who = pipeline.persona()
    stamp = int(time.time())

    def work():
        for old in CAND_DIR.glob("*.png"):
            old.unlink(missing_ok=True)
        base = "front-facing upper-body portrait, plain light background, relaxed natural expression"
        for i in range(n):
            pipeline._say(f"Chehra {i + 1}/{n} ban raha hai...")
            images.generate(images.full_prompt(who, base), CAND_DIR / f"cand_{stamp}_{i + 1}.png",
                            seed=100 + i * 7, ref=None, provider="placeholder" if demo else None)
        return {"count": n}

    return {"job": submit("character", work)}


@app.post("/api/character/select")
async def select_candidate(request: Request):
    name = (await request.json()).get("name", "")
    src = (CAND_DIR / name).resolve()
    if CAND_DIR.resolve() not in src.parents or not src.exists():
        raise HTTPException(404, "Candidate nahi mila")
    shutil.copyfile(src, REFERENCE_IMAGE)
    persist.push(REFERENCE_IMAGE)
    return {"ok": True}


@app.post("/api/character/upload")
async def upload_reference(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 8_000_000:
        raise HTTPException(400, "Image 8 MB se chhoti honi chahiye")
    from io import BytesIO

    from PIL import Image

    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "Ye image file nahi lag rahi") from e
    img.thumbnail((1536, 1536))
    img.save(REFERENCE_IMAGE, "PNG")
    persist.push(REFERENCE_IMAGE)
    return {"ok": True}


# ---- config editors ----
@app.get("/api/config/{which}")
def get_config(which: str):
    p = CONFIG_FILES.get(which)
    if not p:
        raise HTTPException(404)
    return {"text": p.read_text(encoding="utf-8")}


@app.put("/api/config/{which}")
async def put_config(which: str, request: Request):
    p = CONFIG_FILES.get(which)
    if not p:
        raise HTTPException(404)
    text = (await request.json()).get("text", "")
    _validate_yaml(text)
    p.write_text(text, encoding="utf-8")
    persist.push(p)
    return {"ok": True}


@app.get("/api/brands")
def brands():
    out = []
    for f in sorted((ROOT / "brands").glob("*.yaml")):
        out.append({"name": f.stem, "text": f.read_text(encoding="utf-8")})
    return out


@app.put("/api/brands/{name}")
async def put_brand(name: str, request: Request):
    if not NAME_RE.match(name):
        raise HTTPException(400, "Naam sirf a-z, 0-9, - aur _ ka ho")
    text = (await request.json()).get("text", "")
    _validate_yaml(text)
    p = ROOT / "brands" / f"{name}.yaml"
    p.write_text(text, encoding="utf-8")
    persist.push(p)
    return {"ok": True}


@app.delete("/api/brands/{name}")
def del_brand(name: str):
    if not NAME_RE.match(name):
        raise HTTPException(400)
    p = ROOT / "brands" / f"{name}.yaml"
    p.unlink(missing_ok=True)
    persist.remove(p)
    return {"ok": True}


# ---- daily automation (cron-job.org yahan ping karega) ----
@app.api_route("/api/cron/auto", methods=["GET", "POST"])
async def cron_auto(request: Request):
    tok = env("CRON_TOKEN")
    given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or request.query_params.get("token", "")
    if not tok or not hmac.compare_digest(given.encode(), tok.encode()):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    today = now_ist().strftime("%Y-%m-%d")
    todays = [h for h in load_history() if h.get("source") == "cron" and str(h.get("date", "")).startswith(today)]
    if any(h.get("posted") for h in todays):
        return {"skipped": "aaj ka post ho chuka hai"}
    if len(todays) >= 2:  # post fail hota rahe to bhi din mein 2 se zyada baar paise kharch nahi honge
        return {"skipped": "aaj 2 koshish ho chuki hain, Library mein check karo"}
    if any(j["label"] == "auto" and j["status"] in ("queued", "running") for j in jobs.values()):
        return {"skipped": "job already chal raha hai"}
    jid = submit("auto", lambda: pipeline.run_auto(post=True, source="cron"))
    return JSONResponse({"job": jid}, status_code=202)


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")
app.mount("/character-files", StaticFiles(directory=ROOT / "character"), name="character-files")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})
