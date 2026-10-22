import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from influencer import ideas, reference  # noqa: E402
from influencer.config import persona, settings  # noqa: E402
from influencer.video import run_ff  # noqa: E402


class R:
    ok, status_code, text = True, 200, ""

    def __init__(self, d):
        self._d = d

    def json(self):
        return self._d


class RefTests(unittest.TestCase):
    def test_url_validation(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(reference.ReferenceError):
            reference.download("http://evil.example/x", Path(d))

    def test_download_failure_is_clear(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("influencer.reference.subprocess.run", return_value=mock.Mock(returncode=1, stderr="blocked", stdout="")):
            with self.assertRaises(reference.ReferenceError) as c:
                reference.download("https://www.instagram.com/reel/abc/", Path(d))
            self.assertIn("download nahi", str(c.exception))

    def test_analyze_and_prompt(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            d = Path(d)
            src, small = d / "a.mp4", d / "s.mp4"
            run_ff(["-f", "lavfi", "-i", "testsrc=size=640x1136:rate=25:duration=3", "-pix_fmt", "yuv420p", str(src)])
            reference.shrink(src, small)
            notes = {"topic": "budget tips", "hook": "question in 2s", "structure": ["problem 0-3s"]}
            sess = mock.Mock()
            sess.post.return_value = R({"candidates": [{"content": {"parts": [{"text": json.dumps(notes)}]}}]})
            out = reference.analyze_video(small, session=sess)
            self.assertEqual(out["topic"], "budget tips")
            self.assertIn("inline_data", json.dumps(sess.post.call_args.kwargs["json"])[:200])
        p = ideas._prompt("reel", persona(), settings(), [], None, None, 5, ("a", "b"), "hook: question in 2s")
        self.assertIn("INSPIRATION", p)
        self.assertIn("ORIGINAL", p)
        self.assertNotIn("INSPIRATION", ideas._prompt("reel", persona(), settings(), [], None, None, 5, ("a", "b")))


if __name__ == "__main__":
    unittest.main()
