import json
import random
import unittest
from pathlib import Path

import numpy as np

from config import settings
from modules import evaluation as ev
from modules import image_generator as ig
from modules import lora_registry, metrics, prompt_tools
from modules import text_embeddings as te
from services import gpu_manager
from tests import helpers


class EmbeddingMathTests(unittest.TestCase):
    def test_cosine_matches_formula(self):
        a, b = np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])
        sim = te.cosine_matrix(np.vstack([a, b, -a]))
        self.assertAlmostEqual(sim[0, 1], float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), places=9)
        self.assertAlmostEqual(sim[0, 0], 1.0); self.assertAlmostEqual(sim[0, 2], -1.0)
        self.assertTrue(np.allclose(sim, sim.T))

    def test_scale_invariance_and_zero_vector(self):
        sim = te.cosine_matrix(np.array([[1.0, 2.0], [10.0, 20.0], [0.0, 0.0]]))
        self.assertAlmostEqual(sim[0, 1], 1.0)
        self.assertTrue(np.isfinite(sim).all())

    def test_rank_pairs_orders_by_similarity(self):
        emb = np.array([[1, 0], [0.9, 0.1], [0, 1.0]])
        pairs = te.rank_pairs(te.cosine_matrix(emb), ["a", "b", "c"])
        self.assertEqual((pairs.iloc[0]["prompt A"], pairs.iloc[0]["prompt B"]), ("a", "b"))
        self.assertEqual(len(pairs), 3)

    def test_pca_umap_kmeans_agreement_exports(self):
        rng = np.random.default_rng(0)
        emb = np.vstack([rng.normal(0, .1, (6, 8)), rng.normal(3, .1, (6, 8))])
        self.assertEqual(te.pca_2d(emb).shape, (12, 2))
        labels = te.kmeans_labels(emb, 2)
        self.assertEqual(len(set(labels[:6])), 1); self.assertNotEqual(labels[0], labels[-1])
        with self.assertRaises(ValueError):
            te.umap_2d(emb[:3])
        self.assertAlmostEqual(te.similarity_agreement(te.cosine_matrix(emb), te.cosine_matrix(emb)), 1.0)
        self.assertTrue(te.embeddings_csv(emb[:2], ["x", "y"]).startswith(b"prompt,d0"))
        self.assertEqual(np.load(__import__("io").BytesIO(te.embeddings_npy(emb))).shape, (12, 8))
        self.assertIn("Similarity report", te.similarity_report_md(["a", "b"], te.cosine_matrix(emb[:2]), "m"))


class MetricsTests(unittest.TestCase):
    def test_psnr_ssim_identical_and_different(self):
        a = helpers.noise_image(1, (64, 64)); b = helpers.noise_image(2, (64, 64))
        self.assertAlmostEqual(metrics.ssim(a, a), 1.0, places=5)
        self.assertGreater(metrics.psnr(a, a), 90)
        self.assertLess(metrics.ssim(a, b), 0.2)
        self.assertLess(metrics.psnr(a, b), 20)

    def test_classification_metrics_perfect_and_chance(self):
        y = np.repeat(np.arange(3), 20)
        perfect = np.eye(3)[y]
        m = metrics.classification_metrics(y, perfect, ["a", "b", "c"])
        self.assertEqual(m["accuracy"], 1.0); self.assertEqual(m["macro_auc"], 1.0)
        self.assertEqual(np.array(m["confusion_matrix"]).trace(), 60)
        rng = np.random.default_rng(0)
        rnd = rng.dirichlet(np.ones(3), size=len(y))
        m2 = metrics.classification_metrics(y, rnd, ["a", "b", "c"])
        self.assertLess(m2["accuracy"], 0.7)
        self.assertLess(abs(m2["macro_auc"] - 0.5), 0.2)
        self.assertEqual(set(m2["roc"]), {"a", "b", "c"})

    def test_missing_class_is_skipped_not_faked(self):
        y = np.array([0, 0, 1, 1]); probs = np.array([[.9, .05, .05], [.8, .1, .1], [.1, .8, .1], [.2, .7, .1]])
        m = metrics.classification_metrics(y, probs, ["a", "b", "c"])
        self.assertNotIn("c", m["per_class_auc"])

    def test_fid_refuses_too_few_images(self):
        imgs = [helpers.noise_image(i) for i in range(5)]
        with self.assertRaises(ValueError):
            metrics.fid_score(imgs, imgs)


class PromptToolTests(unittest.TestCase):
    def test_enhancer_keeps_original_adds_detail_no_duplicates(self):
        out = prompt_tools.enhance_prompt("A beautiful house near a lake", random.Random(1))
        self.assertTrue(out.startswith("A beautiful house near a lake, "))
        self.assertGreater(len(out), 60)
        parts = [p.strip().lower() for p in out.split(",")]
        self.assertEqual(len(parts), len(set(parts)))
        self.assertEqual(prompt_tools.enhance_prompt("   "), "")
        self.assertNotIn("golden-hour", prompt_tools.enhance_prompt("a lake at sunset", random.Random(1)).replace("sunset", ""))

    def test_random_and_presets(self):
        self.assertNotEqual(prompt_tools.random_prompt(random.Random(1)), prompt_tools.random_prompt(random.Random(2)))
        self.assertTrue(prompt_tools.apply_preset("a cat", "Anime").startswith("a cat, anime"))
        self.assertEqual(prompt_tools.apply_preset("", "Anime"), settings.STYLE_PRESETS["Anime"])


class RequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.fresh_data_root()

    def tearDown(self):
        helpers.cleanup(self.tmp)

    def test_valid_and_invalid_requests(self):
        ok = ig.GenerationRequest(prompt="a cat")
        self.assertEqual(ig.validate_request(ok), [])
        bad = ig.GenerationRequest(prompt="", width=513, height=100, steps=0, guidance=99, batch_size=20, seed=-5, scheduler="nope")
        errors = " ".join(ig.validate_request(bad))
        for needle in ("Enter a prompt", "multiple of 8", "between 256", "Steps", "Guidance", "Batch", "Seed", "scheduler"):
            self.assertIn(needle, errors)
        self.assertTrue(ig.validate_request(ig.GenerationRequest(prompt="x", width=2048, height=512)))
        self.assertTrue(ig.validate_request(ig.GenerationRequest(prompt="x" * 1001)))

    def test_generate_reports_errors_instead_of_fake_images(self):
        with self.assertRaises(Exception):
            ig.generate(ig.GenerationRequest(prompt=""))

    def test_high_quality_and_aspect(self):
        req = ig.apply_high_quality(ig.GenerationRequest(prompt="x", steps=10, guidance=3, scheduler="DDIM"))
        self.assertEqual((req.steps, req.guidance, req.scheduler), (35, 7.5, "DPM++ 2M Karras"))
        for ratio in settings.ASPECT_RATIOS:
            w, h = ig.aspect_dims(ratio, 512)
            self.assertEqual((w % 8, h % 8), (0, 0))
            self.assertLess(abs(w * h - 512 * 512) / (512 * 512), 0.1)
        self.assertEqual(ig.parse_resolution("1024 × 768"), (1024, 768))
        self.assertGreater(ig.upscale_lanczos(helpers.noise_image(1, (20, 10))).width, 39)

    def test_seed_resolution(self):
        self.assertEqual(ig.resolve_seed(123), 123)
        self.assertTrue(0 <= ig.resolve_seed(-1) < 2 ** 32)


class GpuManagerTests(unittest.TestCase):
    def test_graceful_without_gpu_and_memory_checks(self):
        info = gpu_manager.get_device_info()
        self.assertIn("cuda", info)
        cpu = {"cuda": False, "total_gb": 0, "free_gb": 0, "name": "CPU", "torch_installed": False}
        self.assertEqual(gpu_manager.check_generation_request("sdxl", 1024, 1024, 1, cpu)[0], "error")
        self.assertEqual(gpu_manager.check_generation_request("sd15", 512, 512, 1, cpu)[0], "warn")
        t4 = {"cuda": True, "total_gb": 15.0, "free_gb": 14, "name": "Tesla T4", "torch_installed": True}
        self.assertEqual(gpu_manager.check_generation_request("sd15", 512, 512, 1, t4)[0], "ok")
        self.assertEqual(gpu_manager.check_generation_request("sdxl", 1024, 1024, 8, t4)[0], "error")
        small = {**t4, "total_gb": 8.0}
        self.assertEqual(gpu_manager.check_generation_request("sdxl", 1024, 1024, 1, small)[0], "warn")
        self.assertTrue(gpu_manager.needs_offload("sdxl", 1024, 1024, 1, small))
        self.assertFalse(gpu_manager.needs_offload("sd15", 512, 512, 1, t4))
        self.assertIn("T4", gpu_manager.device_label(t4))
        self.assertIn("SD", gpu_manager.recommended_profile(t4)["default_model"].upper() + "SD")


class LoraRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.fresh_data_root()

    def tearDown(self):
        helpers.cleanup(self.tmp)

    def _adapter(self, name, family):
        d = lora_registry.adapters_dir() / name
        d.mkdir(parents=True)
        (d / "pytorch_lora_weights.safetensors").write_bytes(b"x")
        (d / "adapter_meta.json").write_text(json.dumps({"family": family, "base_model_key": "m", "rank": 8}))
        return d

    def test_compatibility_blocks_wrong_base_model(self):
        sd15, sdxl = self._adapter("a15", "sd15"), self._adapter("axl", "sdxl")
        self.assertTrue(lora_registry.check_compatibility(sd15, "Stable Diffusion 1.5")[0])
        ok, msg = lora_registry.check_compatibility(sd15, "Stable Diffusion XL Base 1.0")
        self.assertFalse(ok); self.assertIn("cannot be used", msg)
        self.assertFalse(lora_registry.check_compatibility(sdxl, "Stable Diffusion 1.5")[0])
        unknown = lora_registry.adapters_dir() / "mystery"; unknown.mkdir(); (unknown / "w.safetensors").write_bytes(b"x")
        self.assertFalse(lora_registry.check_compatibility(unknown, "Stable Diffusion 1.5")[0])

    def test_generation_request_rejects_incompatible_lora(self):
        sdxl = self._adapter("axl", "sdxl")
        errs = ig.validate_request(ig.GenerationRequest(prompt="x", model_key="Stable Diffusion 1.5", lora_dir=str(sdxl)))
        self.assertTrue(any("cannot be used" in e for e in errs))

    def test_list_rename_delete_export(self):
        self._adapter("one", "sd15")
        self.assertEqual([a["name"] for a in lora_registry.list_adapters()], ["one"])
        self.assertEqual(lora_registry.rename_adapter("one", "two words"), "two_words")
        self.assertEqual(lora_registry.read_meta(lora_registry.adapters_dir() / "two_words")["adapter_name"], "two_words")
        self._adapter("dup", "sd15")
        with self.assertRaises(ValueError):
            lora_registry.rename_adapter("dup", "two_words")
        with self.assertRaises(ValueError):
            lora_registry.sanitize_name("***")
        self.assertGreater(len(lora_registry.export_adapter_zip("two_words")), 100)
        lora_registry.delete_adapter("two_words")
        self.assertEqual([a["name"] for a in lora_registry.list_adapters()], ["dup"])

    def test_family_detection_from_real_safetensors(self):
        try:
            from safetensors.numpy import save_file
        except Exception:
            self.skipTest("safetensors not installed")
        for dim, fam in ((768, "sd15"), (2048, "sdxl")):
            d = lora_registry.adapters_dir() / f"w{dim}"; d.mkdir()
            save_file({"unet.down_blocks.0.attn2.to_k.lora_A.weight": np.zeros((4, dim), dtype=np.float32)}, str(d / "a.safetensors"))
            self.assertEqual(lora_registry.detect_family_from_weights(d), fam)


class EvaluationReportTests(unittest.TestCase):
    def test_rank_pdf_csv_repro_curves(self):
        rows = [dict(model="A", clip_score=30.0, time_s=10.0), dict(model="B", clip_score=30.2, time_s=3.0), dict(model="C", clip_score=25.0, time_s=1.0)]
        table = ev.rank_models(rows)
        self.assertEqual([t["model"] for t in table], ["B", "A", "C"])   # B/A tie on quality (within 0.5) -> faster first
        exps = [{"id": 1, "kind": "gan_classifier_eval", "name": "run1", "created_at": "2026-11-07", "config": {}, "metrics": {"accuracy": 0.8123, "macro_auc": 0.95, "ranking": [1]}}]
        pdf = ev.build_pdf_report("T", exps, [helpers.noise_image(1)])
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertIn(b"accuracy", ev.experiments_to_csv(exps))
        rep = ev.reproducibility_report()
        self.assertIn("packages", rep); self.assertIn("metric_methodology", rep)
        png = ev.training_curves_png([{"epoch": 1, "loss_g": 1, "loss_d": 1, "d_real": .8, "d_fake": .2}, {"epoch": 2, "loss_g": .9, "loss_d": 1.1, "d_real": .7, "d_fake": .3}])
        self.assertTrue(png.startswith(b"\x89PNG"))


class NotebookAndStructureTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_notebook_is_valid_and_complete(self):
        nb = json.loads((self.ROOT / "notebooks" / "Text_to_Image_Studio_Colab.ipynb").read_text())
        self.assertEqual(nb["nbformat"], 4)
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        for needle in ("nvidia-smi", "drive.mount", "pip install", "snapshot_download", "streamlit", "trycloudflare", "STUDIO_DATA_DIR"):
            self.assertIn(needle, text)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                compile("\n".join(l for l in "".join(c["source"]).splitlines() if not l.startswith("!")), "cell", "exec")

    def test_required_files_exist(self):
        for f in ["app.py", "requirements.txt", "README.md", ".gitignore", "assets/styles/theme.css", "services/job_manager.py",
                  "modules/gan_models.py", "views/home.py", "notebooks/Text_to_Image_Studio_Colab.ipynb"]:
            self.assertTrue((self.ROOT / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
