import json
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from modules import coco_import, dataset_explorer as dx
from tests import helpers


def _png(color) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, "PNG")
    return buf.getvalue()


class CocoImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.fresh_data_root()
        self.imgs = self.tmp / "images"
        self.imgs.mkdir()
        images, anns, inst, cats = [], [], [], [{"id": 1, "name": "traffic light"}, {"id": 2, "name": "dog"}]
        for i in range(1, 7):
            name = f"{i:06d}.jpg"
            (self.imgs / name).write_bytes(_png((i * 30, 20, 100)))
            images.append({"id": i, "file_name": name, "coco_url": f"http://example.invalid/{name}"})
            for j in range(3):
                anns.append({"id": i * 10 + j, "image_id": i, "caption": f"  caption {j} of image {i}  with   extra words" if j else f"short {i}"})
            inst.append({"id": 100 + i, "image_id": i, "category_id": 1 if i % 2 else 2, "area": 50.0})
            inst.append({"id": 200 + i, "image_id": i, "category_id": 2 if i % 2 else 1, "area": 5.0})
        images.append({"id": 99, "file_name": "nocaption.jpg", "coco_url": "http://example.invalid/x"})   # no captions -> ignored
        self.cap_json, self.inst_json = self.tmp / "captions.json", self.tmp / "instances.json"
        self.cap_json.write_text(json.dumps({"images": images, "annotations": anns}))
        self.inst_json.write_text(json.dumps({"categories": cats, "annotations": inst}))

    def tearDown(self):
        helpers.cleanup(self.tmp)

    def test_local_images_classes_and_captions(self):
        out = self.tmp / "out"
        s = coco_import.build_coco_subset(self.cap_json, out, self.imgs, self.inst_json, max_images=4, seed=1)
        self.assertEqual(s["images"], 4)
        self.assertEqual(set(s["classes"]) <= {"traffic_light", "dog"}, True)     # main category by largest area, spaces -> _
        rows = dx.load_captions(out, dx.scan_dataset(out))
        self.assertEqual(len(rows), 4)
        self.assertTrue((rows["source"] == "captions.csv").all())               # real captions, not the class template
        self.assertTrue(rows["caption"].str.startswith("short").all())          # caption_index 0 = first caption
        self.assertTrue((out / "captions_all.json").exists())
        stats = dx.caption_stats(rows)
        self.assertFalse(stats["templated_only"])                                # the Descriptions tab sees real captions
        self.assertEqual(stats["n_captions"], 4)

    def test_without_instances_single_class_and_whitespace_cleaned(self):
        out = self.tmp / "out2"
        s = coco_import.build_coco_subset(self.cap_json, out, self.imgs, None, max_images=10, caption_index=1)
        self.assertEqual(s["classes"], {"coco": 6})                              # image 99 has no caption, so only 6
        self.assertNotIn("  ", (out / "captions.csv").read_text())

    def test_same_seed_same_subset(self):
        a = coco_import.build_coco_subset(self.cap_json, self.tmp / "a", self.imgs, None, max_images=3, seed=7)
        b = coco_import.build_coco_subset(self.cap_json, self.tmp / "b", self.imgs, None, max_images=3, seed=7)
        self.assertEqual((self.tmp / "a" / "captions.csv").read_text(), (self.tmp / "b" / "captions.csv").read_text())
        self.assertEqual(a["images"], b["images"])

    def test_download_path_and_bad_download_is_skipped(self):
        def fetch(url):
            if url.endswith("000002.jpg"):
                return b"this is not an image"
            return _png((1, 2, 3))
        out = self.tmp / "dl"
        s = coco_import.build_coco_subset(self.cap_json, out, None, None, max_images=6, fetch=fetch)
        self.assertEqual(s["images"], 5)
        self.assertEqual(s["skipped"], 1)
        self.assertFalse((out / "coco" / "000002.jpg").exists())                 # the broken file was removed

    def test_errors(self):
        with self.assertRaises(ValueError):
            coco_import.build_coco_subset(self.cap_json, self.tmp / "x", self.imgs, None, max_images=0)
        with self.assertRaises(ValueError):                                      # images folder has none of the files
            coco_import.build_coco_subset(self.cap_json, self.tmp / "y", self.tmp / "empty", None)


if __name__ == "__main__":
    unittest.main()
