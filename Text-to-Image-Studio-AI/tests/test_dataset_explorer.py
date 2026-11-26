import io
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from modules import dataset_explorer as dx
from tests import helpers


class DatasetExplorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.fresh_data_root()
        self.root = helpers.make_dataset(self.tmp / "ds")

    def tearDown(self):
        helpers.cleanup(self.tmp)

    def test_scan_counts_classes_corruption_and_duplicates(self):
        df = dx.scan_dataset(self.root)
        s = dx.summarize(df)
        self.assertEqual(s["total_images"], 33)          # 30 + 2 duplicates + 1 corrupt file
        self.assertEqual(s["num_classes"], 3)
        self.assertEqual(s["corrupted"], 1)
        self.assertEqual(s["duplicates"], 2)             # two extra copies
        self.assertEqual(s["class_counts"]["cats"], 12)

    def test_unlabeled_mode_has_single_group(self):
        df = dx.scan_dataset(self.root, labeled=False)
        self.assertEqual(df["cls"].nunique(), 1)

    def test_near_duplicate_detection(self):
        a = helpers.noise_image(5)
        a.save(self.tmp / "ds" / "cats" / "orig.png")
        a.resize((50, 42)).save(self.tmp / "ds" / "cats" / "resized.png")   # same content, different bytes
        df = dx.scan_dataset(self.root)
        exact = int(df["is_duplicate"].sum())
        near = int(dx.mark_duplicates(df, hamming=6)["is_duplicate"].sum())
        self.assertGreater(near, exact)

    def test_split_is_reproducible_leakage_safe_and_complete(self):
        df = dx.scan_dataset(self.root)
        df = dx.mark_duplicates(df, 0)
        a = dx.split_dataset(df, (0.7, 0.15, 0.15), seed=7)
        b = dx.split_dataset(df, (0.7, 0.15, 0.15), seed=7)
        c = dx.split_dataset(df, (0.7, 0.15, 0.15), seed=8)
        self.assertTrue(a.equals(b))
        self.assertFalse(a.equals(c))
        usable = df[~df["corrupted"]]
        self.assertTrue((a[usable.index] != "").all())
        self.assertTrue((a[df[df["corrupted"]].index] == "").all())
        for _g, sub in df[df["dup_group"] >= 0].groupby("dup_group"):
            self.assertEqual(a[sub.index].nunique(), 1, "duplicates were split across partitions")
        self.assertGreater((a == "train").sum(), (a == "test").sum())

    def test_split_rejects_bad_ratios(self):
        df = dx.scan_dataset(self.root)
        with self.assertRaises(ValueError):
            dx.split_dataset(df, (0.5, 0.2, 0.2))

    def test_preprocess_modes(self):
        img = helpers.noise_image(1, (80, 50))
        for mode in ("resize", "center_crop", "random_crop"):
            out = dx.preprocess_image(img, dx.PreprocessConfig(size=32, mode=mode))
            self.assertEqual(out.size, (32, 32))
        arr = dx.normalize_array(out, "-1..1")
        self.assertGreaterEqual(arr.min(), -1.0)
        self.assertLessEqual(arr.max(), 1.0)
        flipped = dx.preprocess_image(img, dx.PreprocessConfig(size=32, mode="resize", flip=True, flip_probability=1.0))
        plain = dx.preprocess_image(img, dx.PreprocessConfig(size=32, mode="resize"))
        import numpy as np
        self.assertTrue(np.array_equal(np.asarray(flipped), np.asarray(plain.transpose(Image.FLIP_LEFT_RIGHT))))

    def test_apply_preprocessing_writes_split_folders_and_config(self):
        df = dx.mark_duplicates(dx.scan_dataset(self.root), 0)
        splits = dx.split_dataset(df, (0.7, 0.15, 0.15), 1)
        out = self.tmp / "out"
        n = dx.apply_preprocessing(df, out, dx.PreprocessConfig(size=16, out_format="JPEG"), splits)
        self.assertEqual(n, int((splits != "").sum()))
        self.assertTrue((out / "train").is_dir() and (out / "preprocess_config.json").exists())
        self.assertEqual(dx.training_root(out), out / "train")
        self.assertEqual(dx.class_names_from_root(dx.training_root(out)), ["birds", "cats", "dogs"])
        first = next((out / "train").rglob("*.jpg"))
        with Image.open(first) as im:
            self.assertEqual(im.size, (16, 16))

    def test_remove_corrupted_quarantines_files(self):
        df = dx.scan_dataset(self.root)
        moved = dx.remove_corrupted(df, self.root)
        self.assertEqual(moved, 1)
        self.assertFalse((self.root / "dogs" / "broken.jpg").exists())
        self.assertEqual(dx.summarize(dx.scan_dataset(self.root))["corrupted"], 0)

    def test_zip_extraction_blocks_path_traversal_and_skips_non_images(self):
        buf = io.BytesIO()
        png = io.BytesIO(); helpers.noise_image(3).save(png, "PNG")
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data/cats/a.png", png.getvalue())
            zf.writestr("../evil.png", png.getvalue())
            zf.writestr("data/readme.txt", "hi")
            zf.writestr("__MACOSX/data/cats/._a.png", "junk")
        dest = self.tmp / "unzipped"
        n = dx.safe_extract_zip(buf.getvalue(), dest)
        self.assertEqual(n, 1)
        self.assertFalse((self.tmp / "evil.png").exists())
        dx.flatten_single_root(dest)
        self.assertTrue((dest / "cats" / "a.png").exists())

    def test_empty_zip_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("notes.txt", "x")
        with self.assertRaises(ValueError):
            dx.safe_extract_zip(buf.getvalue(), self.tmp / "z2")

    def test_reports_and_captions(self):
        df = dx.scan_dataset(self.root)
        rep = dx.build_report(df, "demo")
        self.assertEqual(rep["summary"]["total_images"], 33)
        self.assertIn("Dataset report", dx.report_markdown(rep))
        self.assertEqual(dx.class_caption("003_canterbury_bells"), "a photo of canterbury bells")
        self.assertEqual(len(dx.OXFORD102_NAMES), 102)
        self.assertGreater(len(dx.zip_directory(self.root)), 1000)


if __name__ == "__main__":
    unittest.main()


class CaptionAndShapesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.fresh_data_root()

    def tearDown(self):
        helpers.cleanup(self.tmp)

    def test_templated_captions_are_flagged(self):
        root = helpers.make_dataset(self.tmp / "ds", with_duplicate=False, with_corrupt=False)
        df = dx.scan_dataset(root)
        stats = dx.caption_stats(dx.load_captions(root, df))
        self.assertTrue(stats["templated_only"])
        self.assertEqual(stats["sources"], {"template": 30})
        self.assertEqual(stats["words_min"], stats["words_max"])       # "a photo of cats" -> always 4 words

    def test_caption_priority_csv_then_txt_then_template(self):
        root = helpers.make_dataset(self.tmp / "ds", per_class=3, with_duplicate=False, with_corrupt=False)
        dx.write_captions_csv(root, [{"filename": str(Path("cats") / "cats_000.png"), "caption": "a fluffy grey cat sleeping on a sofa"}])
        (root / "cats" / "cats_001.txt").write_text("tabby cat")
        df = dx.scan_dataset(root)
        caps = dx.load_captions(root, df).set_index("rel")
        self.assertEqual(caps.loc[str(Path("cats") / "cats_000.png"), "source"], "captions.csv")
        self.assertEqual(caps.loc[str(Path("cats") / "cats_001.png"), "caption"], "tabby cat")
        self.assertEqual(caps.loc[str(Path("cats") / "cats_002.png"), "source"], "template")
        stats = dx.caption_stats(caps.reset_index())
        self.assertEqual(stats["words_max"], 8)
        self.assertFalse(stats["templated_only"])

    def test_caption_stats_numbers(self):
        import pandas as pd
        caps = pd.DataFrame({"cls": ["a", "a", "b"], "caption": ["one two", "one two three four", "x"], "source": ["captions.csv"] * 3})
        s = dx.caption_stats(caps)
        self.assertEqual((s["words_min"], s["words_max"], s["n_unique"]), (1, 4, 3))
        self.assertAlmostEqual(s["words_mean"], 7 / 3)
        self.assertEqual(s["per_class"], {"a": 3.0, "b": 1.0})
        self.assertEqual(dx.caption_stats(caps.iloc[0:0])["n_captions"], 0)

    def test_shapes_dataset_structure_and_determinism(self):
        a, b = self.tmp / "a", self.tmp / "b"
        self.assertEqual(dx.generate_shapes_dataset(a, ("square", "circle", "star"), per_class=10, size=32, seed=7), 30)
        dx.generate_shapes_dataset(b, ("square", "circle", "star"), per_class=10, size=32, seed=7)
        self.assertEqual(sorted(p.name for p in a.iterdir() if p.is_dir()), ["circle", "square", "star"])
        self.assertEqual((a / "captions.csv").read_text(), (b / "captions.csv").read_text())       # same seed -> same dataset
        with Image.open(a / "square" / "square_0000.png") as im:
            self.assertEqual(im.size, (32, 32))
        df = dx.scan_dataset(a)
        self.assertEqual((dx.summarize(df)["num_classes"], dx.summarize(df)["corrupted"]), (3, 0))
        stats = dx.caption_stats(dx.load_captions(a, df))
        self.assertEqual(stats["sources"], {"captions.csv": 30})
        self.assertGreater(stats["words_max"], stats["words_min"])                                  # real length variation

    def test_shape_is_actually_drawn_and_matches_caption_colour(self):
        import random
        img, attrs = dx.draw_shape("circle", 64, random.Random(1))
        bg = dx._BACKGROUNDS[attrs["bg"]]
        self.assertEqual(img.getpixel((0, 0)), bg)
        self.assertIn(dx._COLORS[attrs["color"]], {c for _, c in img.getcolors(10000)})

    def test_shapes_validation(self):
        with self.assertRaises(ValueError):
            dx.generate_shapes_dataset(self.tmp / "x", ("square",), per_class=10)
        with self.assertRaises(ValueError):
            dx.generate_shapes_dataset(self.tmp / "x", ("square", "hexagon"), per_class=10)

    def test_preprocessing_keeps_real_captions_only(self):
        src = self.tmp / "shapes"
        dx.generate_shapes_dataset(src, ("square", "circle"), per_class=10, size=48, seed=3)
        df = dx.mark_duplicates(dx.scan_dataset(src), 0)
        out = self.tmp / "out"
        n = dx.apply_preprocessing(df, out, dx.PreprocessConfig(size=32), dx.split_dataset(df))
        out_df = dx.scan_dataset(out)
        caps = dx.load_captions(out, out_df)
        self.assertEqual(len(caps), n)
        self.assertEqual(set(caps["source"]), {"captions.csv"})


class ZipCaptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.fresh_data_root()

    def tearDown(self):
        helpers.cleanup(self.tmp)

    def _zip(self, wrapper: str = "") -> bytes:
        src = self.tmp / "src"
        dx.generate_shapes_dataset(src, ("square", "circle"), per_class=8, size=32, seed=1)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in src.rglob("*"):
                if f.is_file():
                    z.write(f, (Path(wrapper) / f.relative_to(src)).as_posix())
            z.writestr(str(Path(wrapper) / "square" / "evil.exe"), b"MZ")        # must not be extracted
        return buf.getvalue()

    def _check(self, wrapper: str):
        dest = self.tmp / ("dest" + wrapper)
        n = dx.safe_extract_zip(self._zip(wrapper), dest)
        dx.flatten_single_root(dest)
        self.assertEqual(n, 16)                                                   # only images are counted
        self.assertFalse(list(dest.rglob("*.exe")))
        caps = dx.load_captions(dest, dx.scan_dataset(dest))
        self.assertEqual(set(caps["source"]), {"captions.csv"})

    def test_zip_keeps_captions_csv(self):
        self._check("")

    def test_zip_with_wrapper_folder_keeps_captions_csv(self):
        self._check("my_dataset")


class CaptionGrammarTests(unittest.TestCase):
    def test_shape_captions_use_an_before_vowels(self):
        tmp = helpers.fresh_data_root()
        try:
            dx.generate_shapes_dataset(tmp / "s", ("square", "circle"), per_class=60, size=32, seed=3)
            caps = dx.load_captions(tmp / "s", dx.scan_dataset(tmp / "s"))["caption"]
            self.assertFalse(caps.str.contains(r"\ba [aeiou]", regex=True).any())
            self.assertTrue(caps.str.contains(r"\ban orange\b", regex=True).any())      # orange is one of the colours
        finally:
            helpers.cleanup(tmp)
