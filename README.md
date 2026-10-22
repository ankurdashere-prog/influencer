# AI Influencer Studio (web based)

Browser mein chalne wala tool: reel, photo aur client ads banao, review karo, ek click se Instagram par post karo, aur roz ki auto-posting chalao.

```
Gemini (script) -> OpenAI image (same face) -> FFmpeg zoom reel -> free voice -> captions
-> Library (review) -> Meta Graph API se Instagram post
```

## Free hosting: Render (browser se hi)

Hugging Face ne free Docker Spaces band kar diye hain (ab PRO chahiye), isliye Render use karte hain.
Render free: 512 MB RAM, 15 minute idle ke baad so jata hai (wapas jagne mein ~1 minute), 750 ghante/mahina. Is tool ka ek reel render lagbhag 250 MB RAM leta hai, to fit hona chahiye. Agar "Out of memory" aaye to `settings.yaml` mein `video.scenes` ko 3 karo.

### Step 1: Supabase (5 minute)
1. supabase.com par free account, New project.
2. Storage, New bucket, naam `influencer-media`, **Public bucket** ON.
3. Project Settings, API se **Project URL** aur **service_role key** copy karo.

### Step 2: Code GitHub par daalo
1. github.com par free account, New repository, naam `ai-influencer`, **Private**.
2. "uploading an existing file" link dabao, zip extract karke uske **andar ki saari files aur folders** drag-drop karo, Commit karo.
   (`.env` file mat daalo, keys sirf Render mein jayengi.)

### Step 3: Render par deploy
1. render.com par "Sign in with GitHub" se account banao.
2. New, **Web Service**, apni `ai-influencer` repo chuno (GitHub access de do).
3. **Language: Docker**, **Instance type: Free**.
4. **Environment Variables** mein ye daalo:

| Name | Value |
|---|---|
| `APP_PASSWORD` | dashboard ka login password (lamba rakho) |
| `CRON_TOKEN` | koi lamba random text |
| `GEMINI_API_KEY` | Gemini key (aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | OpenAI key |
| `IG_USER_ID`, `IG_ACCESS_TOKEN` | Meta Graph API |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Step 1 se |

Optional: `IMAGE_PROVIDER=pollinations` (free image, par same chehra guarantee nahi), `OPENAI_IMAGE_QUALITY=low`, `TTS_VOICE`, `GRAPH_VERSION`.

5. **Create Web Service** dabao. Build mein 5-8 minute lagte hain. Upar URL milega: `https://<naam>.onrender.com`.

### Step 4: Dashboard use karo
URL kholo. Username kuch bhi, password = `APP_PASSWORD`.
1. **Settings** tab: status dekho.
2. **Character** tab: "4 chehre banao", ek chuno.
3. **Create** tab: pehle "Test mode" ON karke, phir asli generate karo.
4. **Library** tab: dekho, theek ho to "Instagram par post karo". Pehle ek **Photo** post karke check karo, phir reel.

### Step 5: Roz ki automation (cron-job.org, free)
1. cron-job.org par account banao, Create cronjob.
2. URL: `https://<naam>.onrender.com/api/cron/auto`
3. Schedule: roz subah 10:00, Timezone Asia/Kolkata.
4. Advanced: Request method **POST**, Header `Authorization` = `Bearer <CRON_TOKEN>`.
5. **Retry on failure ON** rakho, kyunki so chuka server pehli request par jagta hai aur wo request timeout ho sakti hai.

Roz `settings.yaml` ke weekly plan ke hisaab se reel ya photo banke post hoga. Ads kabhi auto-post nahi hote, wo Library mein approval ke liye rukte hain.

## Zaroori baatein
- **Data save:** Render ki disk bhi temporary hai. Supabase set hai to character, settings, brands aur history wahin save hoti hain. Bani hui videos restart ke baad ja sakti hain, kaam ki videos download kar lo.
- **Cost:** sirf OpenAI image paid hai (per image).
- **Voice:** pehle edge-tts, na chale to Google TTS fallback.
- **Rules:** caption mein "AI-generated" label khud lagta hai, hatana mat. Kisi real insaan ka chehra use mat karo. Ads mein `#ad` khud lagta hai.
- Graph API version purana ho jaye to `GRAPH_VERSION` update karo.

## Apne laptop par chalao (hosting ke bina)
```
pip install -r requirements.txt
copy .env.example .env      (keys bharo)
python -m influencer serve  (browser mein http://127.0.0.1:7860)
```
Terminal commands: `python -m influencer make --kind reel --demo`, `doctor`, `auto`.

## Aur ek free automation option
`extras/github-actions-daily.yml` ko `.github/workflows/` mein rakho. GitHub khud roz post banata hai, hosting ya cron-job.org ki zaroorat nahi (keys GitHub Secrets mein). Dashboard ke bina chalta hai.

## Files
- `influencer/web/` dashboard (backend + `static/index.html`)
- `influencer/` pipeline: ideas, images, tts, captions, video, storage, publish, persist
- `character/persona.yaml`, `brands/*.yaml`, `settings.yaml` (Settings tab se bhi edit hote hain)

## Bolta hua influencer (lip-sync) — paid, optional

Render par GPU nahi hai, isliye lip-sync fal.ai (hosted API) se hota hai. Per-reel paisa lagta hai.
Render Environment mein ye daalo (keys kabhi chat/GitHub par nahi):

| Key | Value |
|---|---|
| `TALKING_PROVIDER` | `fal` (band karne ke liye `none`) |
| `FAL_KEY` | fal.ai dashboard se API key |
| `FAL_MODEL` | optional, default `fal-ai/kling-video/v1/standard/ai-avatar` |
| `FAL_EXTRA_JSON` | optional, model ke extra fields, jaise `{"resolution":"480p"}` |

`settings.yaml` ka `video.talking_seconds` batata hai shuru ke kitne second chehra bolega (baaki cutaway images + voice).
Zyada seconds = zyada cost. Lip-sync fail ho to reel apne aap slideshow mode mein ban jaati hai.
