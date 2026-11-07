import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from influencer import story


def _jpg() -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (2000, 1500), (200, 30, 30)).save(b, "JPEG")
    return b.getvalue()


class StoryBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p = [mock.patch.object(story, "LIB_DIR", self.tmp / "lib"), mock.patch.object(story, "LIB_FILE", self.tmp / "lib.json"),
                  mock.patch.object(story.persist, "push"), mock.patch.object(story.persist, "remove")]
        for p in self.p:
            p.start()
        self.doc = story.add_item("character", "Dr Ashish", "plastic surgeon", "a.png", _jpg(), True)
        self.pat = story.add_item("character", "Patient", "", "b.jpg", _jpg(), True)
        self.loc = story.add_item("location", "Clinic", "warm reception", "c.jpg", _jpg(), False)

    def tearDown(self):
        for p in self.p:
            p.stop()


class LibraryTests(StoryBase):
    def test_photo_is_shrunk_to_jpeg(self):
        f = story.LIB_DIR / self.doc["file"]
        self.assertEqual(f.suffix, ".jpg")
        self.assertLessEqual(max(Image.open(f).size), 1280)

    def test_character_needs_consent(self):
        with self.assertRaises(story.StoryError):
            story.add_item("character", "X", "", "a.jpg", _jpg(), False)

    def test_bad_type_and_name(self):
        with self.assertRaises(story.StoryError):
            story.add_item("location", "", "", "a.jpg", _jpg(), False)
        with self.assertRaises(story.StoryError):
            story.add_item("location", "L", "", "a.gif", _jpg(), False)
        with self.assertRaises(story.StoryError):
            story.add_item("clip", "C", "", "a.jpg", b"x", False)
        with self.assertRaises(story.StoryError):
            story.add_item("location", "L", "", "a.jpg", b"not an image", False)

    def test_delete(self):
        f = story.LIB_DIR / self.loc["file"]
        self.assertTrue(f.exists())
        self.assertTrue(story.delete_item(self.loc["id"]))
        self.assertFalse(f.exists())
        self.assertFalse(story.delete_item(self.loc["id"]))


class PlanTests(StoryBase):
    BRIEF = "Dr clinic me enter karte hain. Phir patient ko dekhte hain. Phir unse baat karte hain."

    def test_scene_count(self):
        self.assertEqual([story.scene_count(s) for s in (20, 40, 60, 5, 500)], [3, 5, 8, 1, 8])

    def test_offline_template(self):
        p = story.plan_story(self.BRIEF, 24, [self.doc["id"], self.pat["id"]], [self.loc["id"]], [], offline=True)
        self.assertEqual(p["n_scenes"], 3)
        self.assertEqual(p["source"], "template")
        self.assertTrue(any("AI key" in w for w in p["warnings"]))
        for sc in p["scenes"]:
            self.assertIn("IDENTITY LOCK", sc["flow_prompt"]) if sc["type"] == "talking" else None
            self.assertIn("LOCATION LOCK", sc["flow_prompt"])
            self.assertIn("ONE SINGLE IMAGE", sc["keyframe_prompt"])

    def test_needs_consented_character(self):
        with self.assertRaises(story.StoryError):
            story.plan_story(self.BRIEF, 24, [], [self.loc["id"]], [], offline=True)
        items = story._load()
        for i in items:
            if i["id"] == self.pat["id"]:
                i["consent"] = False
        story._save(items)
        with self.assertRaises(story.StoryError):
            story.plan_story(self.BRIEF, 24, [self.pat["id"]], [], [], offline=True)

    def test_short_brief(self):
        with self.assertRaises(story.StoryError):
            story.plan_story("hi", 24, [self.doc["id"]], [], [], offline=True)

    def test_llm_plan_is_normalised(self):
        d, pa = self.doc["id"], self.pat["id"]
        raw = {"title": "T", "caption": "c", "hashtags": ["#a"], "scenes": [
            {"type": "talking", "characters": [d, "bogus"], "speaker": d, "location": self.loc["id"], "camera": "Close-up",
             "action": "smiles", "line": "Namaste, main aapka doctor hu aur aaj aapko ek zaroori baat batana chahta hu"},
            {"type": "talking", "characters": [], "speaker": "zzz", "location": "nope", "line": ""},
            {"type": "broll", "visual": "clinic hallway", "line": "Sahi jaankari pehla kadam hai aapki sehat ke liye hamesha"}]}
        with mock.patch.object(story, "gemini_json", return_value=raw):
            p = story.plan_story(self.BRIEF, 24, [d, pa], [self.loc["id"]], [], offline=False)
        self.assertEqual(p["source"], "ai")
        s1, s2, s3 = p["scenes"]
        self.assertEqual(s1["characters"], [d])
        self.assertEqual(s2["type"], "broll")          # speaker nahi mila
        self.assertEqual(s2["location"], "")
        self.assertTrue(any("khaali" in w for w in p["warnings"]))
        self.assertIn("No one speaks", s3["flow_prompt"])
        self.assertIn("speaks in", s1["flow_prompt"])
        self.assertEqual(p["hashtags"], ["a"])

    def test_llm_failure_falls_back(self):
        with mock.patch.object(story, "gemini_json", side_effect=RuntimeError("boom")):
            p = story.plan_story(self.BRIEF, 16, [self.doc["id"]], [], [], offline=False)
        self.assertEqual(p["source"], "template")
        self.assertTrue(any("AI se plan nahi bana" in w for w in p["warnings"]))

    def test_rebuild_prompts_uses_edited_line(self):
        p = story.plan_story(self.BRIEF, 16, [self.doc["id"]], [self.loc["id"]], [], offline=True)
        sc = p["scenes"][0]
        sc["type"], sc["speaker"], sc["characters"] = "talking", self.doc["id"], [self.doc["id"]]
        sc["line"] = "Diwali ki shubhkamnayein aap sabko"
        story.rebuild_prompts(p)
        self.assertIn("Diwali ki shubhkamnayein", sc["flow_prompt"])
        self.assertEqual(sc["words"], 5)


if __name__ == "__main__":
    unittest.main()
