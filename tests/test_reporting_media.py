"""Report generation for media (video/image) use cases: the report must surface
generated media instead of ignoring it, and must not mislabel a media column as
an empty numeric signal in the Distributions section."""
import io
import unittest

import pandas as pd

from weightslab import reporting
from weightslab.data import media_store


def _png(color=(200, 60, 60)):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(b, "PNG")
    return b.getvalue()


class MediaReportTests(unittest.TestCase):
    def setUp(self):
        media_store.clear()
        self.addCleanup(media_store.clear)
        for sid in range(3):
            media_store.put("pred_video", sid, data=b"FAKE", mime="video/mp4",
                            kind="video", poster=_png())
        self.df = pd.DataFrame(
            {"media:pred_video": [media_store.descriptor_json(media_store.get("pred_video", s))
                                  for s in range(3)],
             "signals//train/fm_loss": [0.5, 0.3, 0.9]},
            index=pd.Index([0, 1, 2], name="sample_id"),
        )

    def test_compute_media_examples_finds_field_and_posters(self):
        examples = reporting.compute_media_examples(self.df)
        self.assertEqual(len(examples), 1)
        ex = examples[0]
        self.assertEqual(ex["field"], "pred_video")
        self.assertEqual(ex["kind"], "video")
        self.assertEqual(ex["count"], 3)
        self.assertTrue(ex["thumbnails"], "expected poster thumbnails")
        self.assertTrue(ex["thumbnails"][0]["poster_uri"].startswith("data:image/"))

    def test_media_examples_empty_without_media(self):
        plain = pd.DataFrame({"signals//loss": [1.0, 2.0]})
        self.assertEqual(reporting.compute_media_examples(plain), [])

    def test_media_section_html_renders_and_is_empty_when_none(self):
        html = reporting._media_section_html(reporting.compute_media_examples(self.df))
        self.assertIn("Generated Media", html)
        self.assertIn("pred_video", html)
        self.assertIn("data:image/", html)
        self.assertEqual(reporting._media_section_html([]), "")

    def test_distribution_on_media_column_is_flagged_not_empty_numeric(self):
        entries = reporting.compute_distribution_entries(self.df, ["pred_video"], plt=None)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].get("is_media"))
        card = reporting._distribution_card_html(entries[0], "b0")
        self.assertIn("is a media column", card)
        self.assertNotIn("No numeric values logged", card)


if __name__ == "__main__":
    unittest.main()
