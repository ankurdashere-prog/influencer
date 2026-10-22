"""Talking-head tests: koi asli API call nahi (fake fal session)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencer import images, pipeline, talking, tts  # noqa: E402
from influencer.config import settings  # noqa: E402
from influencer.video import assemble_reel, media_duration, run_ff  # noqa: E402


class Resp:
    def __init__(self, data=None, content=b"", code=200):
        self._d, self.content, self.status_code, self.text = data, content, code, str(data)

    def json(self):
        return self._d


class FakeFal:
    def __init__(self, video_bytes, fail=False):
        self.video, self.fail, self.calls, self.polls = video_bytes, fail, [], 0

    def post(self, url, headers=None, json=None, timeout=0):
        self.calls.append((url, headers, json))
        return Resp({"request_id": "r1", "status_url": "http://s", "response_url": "http://r"})

    def get(self, url, headers=None, timeout=0):
        if url == "http://s":
            self.polls += 1
            return Resp({"status": "FAILED" if self.fail else ("COMPLETED" if self.polls > 1 else "IN_PROGRESS")})
        if url == "http://r":
            return Resp({"video": {"url": "http://v/out.mp4"}})
        return Resp(content=self.video)


def make_video(path, secs=4):
    run_ff(["-f", "lavfi", "-i", f"testsrc=size=512x512:rate=25:duration={secs}", "-pix_fmt", "yuv420p", str(path)])


class TalkingTests(unittest.TestCase):
    def test_pick_cut_on_word_boundary(self):
        words = tts.estimate_words("ek do teen char paanch chhe saat aath nau das", 12.0)
        cut = talking.pick_cut(words, 12.0, 8)
        self.assertLessEqual(cut, 8.05)
        self.assertGreaterEqual(cut, 6.0)
        self.assertAlmostEqual(talking.pick_cut(words, 2.0, 8), 2.0, places=1)

    def test_generate_with_fake_fal(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"FAL_KEY": "k", "FAL_MODEL": "x/y"}):
            d = Path(d)
            vid = d / "src.mp4"
            make_video(vid)
            img, aud = d / "f.png", d / "a.mp3"
            images._placeholder("t", img, 1)
            run_ff(["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "2", str(aud)])
            fal = FakeFal(vid.read_bytes())
            out = talking.generate(img, aud, d / "o.mp4", prompt="hi", poll=0, session=fal)
            self.assertGreater(out.stat().st_size, 1000)
            url, hdr, body = fal.calls[0]
            self.assertTrue(url.endswith("/x/y"))
            self.assertEqual(hdr["Authorization"], "Key k")
            self.assertTrue(body["image_url"].startswith("data:image/png;base64,"))
            self.assertTrue(body["audio_url"].startswith("data:audio/"))
            with self.assertRaises(talking.TalkingError):
                talking.generate(img, aud, d / "o2.mp4", poll=0, session=FakeFal(b"", fail=True))
        with mock.patch.dict(os.environ, {"FAL_KEY": ""}), self.assertRaises(talking.TalkingError):
            talking.generate(Path("a"), Path("b"), Path("c"))

    def test_reel_uses_talking_clip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            vid = d / "talk.mp4"
            make_video(vid, 4)
            imgs = []
            for i in range(3):
                p = d / f"s{i}.png"
                images._placeholder("t", p, i)
                imgs.append(p)
            aud = d / "v.mp3"
            run_ff(["-f", "lavfi", "-i", "sine=frequency=300", "-t", "10", str(aud)])
            cfg = dict(settings()["video"], width=360, height=640, fps=15)
            out = assemble_reel(imgs, aud, None, d / "r.mp4", d, 10.0, cfg, None, talking=vid, talk_dur=4.0)
            self.assertAlmostEqual(media_duration(out), 10.6, delta=0.5)
            self.assertTrue((d / "clip_talk.mp4").exists())

    def test_pipeline_falls_back_when_fal_fails(self):
        msgs = []
        with mock.patch.object(pipeline, "progress", msgs.append), \
                mock.patch.dict(os.environ, {"TALKING_PROVIDER": "fal", "FAL_KEY": ""}):
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                aud = d / "a.mp3"
                run_ff(["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "10", str(aud)])
                words = tts.estimate_words("ek do teen char paanch chhe saat aath", 10.0)
                clip, dur = pipeline._talking_head({"look_prompt": "x"}, {"hook": "h"}, settings(), aud, words, 10.0, d, None)
        self.assertIsNone(clip)
        self.assertEqual(dur, 0.0)
        self.assertTrue(any("Slideshow" in m for m in msgs))


if __name__ == "__main__":
    unittest.main()
