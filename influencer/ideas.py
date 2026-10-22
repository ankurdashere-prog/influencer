"""Content plan: topic, voiceover, scene prompts, caption, hashtags."""
from __future__ import annotations

import random
import re

import yaml

from .llm import LLMError, gemini_json


def _pillars(persona: dict) -> list[tuple[str, str]]:
    out = []
    for p in persona.get("pillars", []):
        if isinstance(p, dict):
            out.extend(p.items())
        else:
            out.append((str(p), ""))
    return out or [("lifestyle", "")]


def _prompt(kind, persona, cfg, avoid, brand, topic, n_scenes, pillar) -> str:
    words = cfg["video"].get("voice_words", 48)
    persona_view = {k: persona.get(k) for k in ("name", "age", "city", "niche", "tone")}
    lines = [
        "You write content for an AI virtual influencer (the audience knows she is AI).",
        f"PERSONA: {yaml.safe_dump(persona_view, allow_unicode=True)}",
        f"FORMAT: {kind}",
        f"CONTENT PILLAR for this post: {pillar[0]} - {pillar[1]}",
    ]
    if topic:
        lines.append(f"TOPIC (given by user, use it): {topic}")
    lines += [
        "",
        "Return ONE JSON object with keys: topic, hook, voiceover, scenes, caption, hashtags.",
        "- topic: short English label (max 8 words).",
        "- hook: scroll-stopping first line, max 8 words, Hinglish.",
        f"- voiceover: ~{words} words, Hinglish (Hindi in Roman script), casual Gen-Z Indian tone, "
        "short sentences, first sentence = the hook, NO emojis, NO hashtags."
        if kind != "photo"
        else "- voiceover: empty string.",
        f"- scenes: list of exactly {n_scenes} objects with key image_prompt. image_prompt is English and "
        "describes ONLY setting, outfit, pose, lighting and camera angle. NEVER describe face, hair or skin "
        "(that is added separately). Each scene must use a different location or pose. Vertical framing.",
        "- caption: Hinglish, 2-4 short lines, 1-3 emojis, end with a question that invites comments.",
        "- hashtags: 8-12 strings without # and without spaces, mix of broad and niche.",
        "RULES: no medical or guaranteed-result claims, no fake personal experiences of illness, "
        "nothing political or adult, never pretend to be a real human.",
    ]
    if avoid:
        lines.append("DO NOT repeat these recent topics: " + "; ".join(avoid))
    if brand:
        lines += [
            "",
            "THIS IS A PAID AD for a client. BRAND BRIEF (the ONLY facts you may use):",
            yaml.safe_dump(brand, allow_unicode=True),
            "Ad rules: use only facts in the brief; do NOT invent prices, discounts, ingredients, results "
            "or reviews; respect do_not_say; the voiceover must end with the brief's cta; keep it sounding "
            "like an honest personal recommendation; product must be visible in scene prompts "
            "(follow image_style).",
        ]
    return "\n".join(lines)


def _normalize(plan: dict, kind: str, n_scenes: int, persona: dict, brand: dict | None) -> dict:
    scenes = [s.get("image_prompt", "") if isinstance(s, dict) else str(s) for s in plan.get("scenes", [])]
    scenes = [s for s in scenes if s.strip()]
    if not scenes:
        raise LLMError("scenes khaali aaye")
    while len(scenes) < n_scenes:
        scenes.append(scenes[len(scenes) % len(scenes)])
    scenes = scenes[:n_scenes]

    tags = []
    for t in plan.get("hashtags", []):
        t = re.sub(r"[^\w]", "", str(t))
        if t and t.lower() not in [x.lower() for x in tags]:
            tags.append(t)
    for t in persona.get("hashtags", []):
        if t.lower() not in [x.lower() for x in tags]:
            tags.append(t)
    if brand:
        tags.insert(0, "ad")
    tags = tags[:28]

    caption = str(plan.get("caption", "")).strip()
    footer = persona.get("disclosure", "")
    parts = [caption]
    if brand:
        parts.append("#ad Paid partnership with " + brand.get("name", "brand"))
    if footer:
        parts.append(footer)
    parts.append(" ".join("#" + t for t in tags))
    return {
        "kind": kind,
        "topic": str(plan.get("topic", "post"))[:80],
        "hook": str(plan.get("hook", "")),
        "voiceover": re.sub(r"\s+", " ", str(plan.get("voiceover", ""))).strip(),
        "scenes": scenes,
        "caption": "\n\n".join(p for p in parts if p)[:2150],
        "hashtags": tags,
    }


_FALLBACK = {
    "fashion_ootd": (
        "Budget OOTD idea",
        "Aaj ka outfit simple hai par look mast hai. Ek white shirt, high waist jeans aur ek statement jhumka. "
        "Bas itna hi chahiye. Is combo ko tum kisi bhi din pehen sakte ho. Tumhara favourite basic kya hai?",
        ["standing in a sunlit street wearing a white shirt and high waist jeans",
         "close up mirror selfie showing statement earrings", "walking in a cafe street with a tote bag",
         "sitting on cafe steps holding iced coffee", "twirling at a park in the golden hour"],
    ),
    "fitness_tips": (
        "Easy home workout",
        "Gym nahi jaa pa rahe? Koi baat nahi. Das squats, das pushups, aur ek minute plank. Teen round karo. "
        "Sirf pandrah minute. Roz karo, farak khud dikhega. Tum kitne round karoge?",
        ["stretching on a yoga mat in a bright living room wearing sportswear",
         "doing squats at home near a window", "holding a plank on a yoga mat",
         "drinking water bottle after workout, smiling", "tying shoelaces ready for a morning walk"],
    ),
    "trending_reel": (
        "Relatable morning mood",
        "Alarm bajta hai aur main sochti hu, bas paanch minute aur. Phir yaad aata hai ki aaj Monday hai. "
        "Chai pi, mood set kiya, aur chal padi. Tumhari morning kaisi hoti hai?",
        ["lying in bed with messy hair holding a phone", "making tea in a kitchen in pyjamas",
         "looking at the camera while sipping chai on a balcony", "getting ready in front of a mirror",
         "walking out of a building door with a bag"],
    ),
    "daily_life": (
        "Cafe day in the city",
        "Aaj ek chhota sa cafe mila jaha ki coffee kamaal thi. Bheed bhi kam thi aur vibe bhi achhi. "
        "Aise hi chhoti chhoti jagah dhoondhna mujhe pasand hai. Tumhara favourite cafe kaun sa hai?",
        ["entering a cozy cafe with plants and warm lights", "sitting by the window with a latte",
         "laughing while looking at the menu", "taking a photo of dessert on the table",
         "walking outside the cafe at sunset with a coffee cup"],
    ),
}


def _fallback(kind, persona, n_scenes, brand, pillar, topic) -> dict:
    if brand:
        name = brand.get("name", "brand")
        product = brand.get("product", "ye product")
        usp = (brand.get("usp") or ["mujhe pasand aaya"])[0]
        offer = brand.get("offer", "")
        cta = brand.get("cta", "")
        voice = (
            f"Skin ya daily routine mein kuch simple chahiye? Main {name} ka {product} use kar rahi hu. "
            f"{usp}. {offer}. {cta}"
        )
        style = brand.get("image_style", "natural light, product visible in hand")
        scenes = [f"holding the {product}, {style}"] * n_scenes
        plan = {"topic": f"{name} ad", "hook": "Simple routine chahiye?", "voiceover": voice,
                "scenes": [{"image_prompt": s} for s in scenes],
                "caption": f"{product} ke saath meri routine ✨ Tum kya use karte ho?",
                "hashtags": ["skincare", "dailyroutine"]}
    else:
        key = pillar[0] if pillar[0] in _FALLBACK else random.choice(list(_FALLBACK))
        t, v, sc = _FALLBACK[key]
        plan = {"topic": topic or t, "hook": v.split(".")[0], "voiceover": v if kind != "photo" else "",
                "scenes": [{"image_prompt": sc[i % len(sc)]} for i in range(n_scenes)],
                "caption": v.split(".")[0] + ". 💛 Tum kya sochte ho?",
                "hashtags": ["ootd", "lifestyle", "fitindia", "reels"]}
    return plan


def make_plan(kind: str, persona: dict, cfg: dict, history: list[dict],
              brand: dict | None = None, topic: str | None = None, offline: bool = False) -> dict:
    n_scenes = 1 if kind == "photo" else int(cfg["video"].get("scenes", 5))
    pillar = random.choice(_pillars(persona))
    avoid_n = int(cfg.get("ideas", {}).get("history_to_avoid", 30))
    avoid = [h.get("topic", "") for h in history[-avoid_n:] if h.get("topic")]
    if not offline:
        try:
            raw = gemini_json(_prompt(kind, persona, cfg, avoid, brand, topic, n_scenes, pillar))
            return _normalize(raw, kind, n_scenes, persona, brand)
        except LLMError as e:
            print(f"[ideas] Gemini nahi chala ({e}). Built-in template use kar raha hu.")
    return _normalize(_fallback(kind, persona, n_scenes, brand, pillar, topic), kind, n_scenes, persona, brand)
