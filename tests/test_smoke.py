"""Offline smoke tests (koi API key ya internet nahi chahiye): python -m unittest discover -s tests"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencer import captions, ideas, pipeline, tts  # noqa: E402
from influencer.config import OUTPUT_DIR, persona, settings  # noqa: E402
from influencer.video import media_duration  # noqa: E402


class SmokeTests(unittest.TestCase):
    def test_estimated_words_cover_duration(self):
        w = tts.estimate_words("namaste dosto aaj ka video", 5.0)
        self.assertEqual(len(w), 5)
        self.assertAlmostEqual(w[-1].end, 5.0, places=3)
        self.assertTrue(all(a.end <= b.start + 1e-9 for a, b in zip(w, w[1:])))

    def test_ass_has_one_dialogue_per_word(self):
        words = tts.estimate_words("ek do teen char paanch chhe", 6.0)
        with tempfile.TemporaryDirectory() as d:
            p = captions.build_ass(words, Path(d) / "c.ass", {"words_per_group": 3})
            text = p.read_text(encoding="utf-8")
        self.assertEqual(text.count("Dialogue:"), 6)
        self.assertIn("PlayResX: 1080", text)

    def test_offline_plan_shape(self):
        cfg = settings()
        plan = ideas.make_plan("reel", persona(), cfg, [], offline=True)
        self.assertEqual(len(plan["scenes"]), cfg["video"]["scenes"])
        self.assertTrue(plan["voiceover"])
        self.assertIn("#", plan["caption"])

    def test_ad_plan_is_marked_as_ad(self):
        brand = {"name": "TestBrand", "product": "serum", "usp": ["halka"], "offer": "", "cta": "link bio"}
        plan = ideas.make_plan("ad", persona(), settings(), [], brand=brand, offline=True)
        self.assertIn("#ad", plan["caption"])

    def test_demo_reel_renders(self):
        res = pipeline.run("reel", demo=True)
        out = OUTPUT_DIR / res["dir"] / res["file"]
        self.assertTrue(out.exists())
        dur = media_duration(out)
        self.assertGreater(dur, 5)


if __name__ == "__main__":
    unittest.main()
