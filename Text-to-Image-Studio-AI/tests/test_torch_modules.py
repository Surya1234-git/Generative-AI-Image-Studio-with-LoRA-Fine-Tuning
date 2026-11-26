"""GAN tests. They need PyTorch and are skipped automatically where it is not installed."""
import importlib.util
import unittest

TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(TORCH, "PyTorch/torchvision not installed")
class GanModelTests(unittest.TestCase):
    def test_shapes_for_all_sizes_and_architectures(self):
        import torch
        from modules import gan_models as gm
        for size in (32, 64, 128):
            for arch in gm.ARCHITECTURES:
                G, D = gm.build_gan(arch, 5, 32, 16, 16, size)
                y = torch.randint(0, 5, (3,))
                x = G(torch.randn(3, 32), y)
                self.assertEqual(tuple(x.shape), (3, 3, size, size))
                self.assertLessEqual(float(x.abs().max()), 1.0)
                self.assertEqual(tuple(D(x, y).shape), (3,))

    def test_attention_only_in_attention_models_and_maps_are_valid(self):
        import torch
        from modules import gan_models as gm
        plain, _ = gm.build_gan("CGAN", 4, 16, 8, 16, 64)
        attn, d_attn = gm.build_gan("Attention CGAN", 4, 16, 8, 16, 64)
        has = lambda m: any(isinstance(x, gm.SelfAttention2d) for x in m.modules())
        self.assertFalse(has(plain)); self.assertTrue(has(attn)); self.assertTrue(has(d_attn))
        self.assertIsNone(gm.attention_overlay(plain, torch.randn(1, 16), torch.tensor([0])))
        images, maps, res = gm.attention_overlay(attn, torch.randn(2, 16), torch.tensor([0, 1]))
        self.assertEqual(res, 16)
        self.assertEqual(tuple(maps.shape), (2, 3, 16, 16))
        self.assertTrue(torch.allclose(maps.sum(dim=(2, 3)), torch.ones(2, 3), atol=1e-4))   # softmax rows sum to 1

    def test_text_conditioning_accepts_new_embeddings(self):
        import torch
        from modules import gan_models as gm
        table = torch.nn.functional.normalize(torch.randn(4, 24), dim=-1)
        G, D = gm.build_gan("CGAN", 4, 16, 8, 16, 32, table)
        img = G.from_text_embedding(torch.randn(2, 16), torch.randn(2, 24))
        self.assertEqual(tuple(img.shape), (2, 3, 32, 32))
        label_G, _ = gm.build_gan("CGAN", 4, 16, 8, 16, 32)
        with self.assertRaises(ValueError):
            label_G.from_text_embedding(torch.randn(1, 16), torch.randn(1, 24))

    def test_attention_gate_starts_at_zero(self):
        import torch
        from modules import gan_models as gm
        layer = gm.SelfAttention2d(32)
        x = torch.randn(2, 32, 8, 8)
        self.assertTrue(torch.allclose(layer(x), x))


@unittest.skipUnless(TORCH, "PyTorch/torchvision not installed")
class GanTrainingTests(unittest.TestCase):
    def test_train_checkpoint_resume_stop_and_generate(self):
        import time
        from modules import dataset_explorer as dx
        from modules import gan_trainer as gt
        from services.job_manager import JobManager
        from tests import helpers
        tmp = helpers.fresh_data_root()
        try:
            root = helpers.make_dataset(tmp / "ds", per_class=12, with_duplicate=False, with_corrupt=False)
            cfg = gt.GANConfig(arch="Attention CGAN", epochs=2, batch_size=8, img_size=32, base_channels=16, z_dim=16, sample_every=1,
                               ckpt_every=1, dataset_name="ds", dataset_path=str(root), run_name="r1")
            jm = JobManager()
            job = jm.start("gan", "t", gt.train_gan, cfg=cfg, resume=False)
            while job.is_active():
                time.sleep(0.2)
            self.assertEqual(job.status, "completed", job.error)
            run = tmp / "checkpoints" / "gan" / "r1"
            self.assertTrue((run / "latest.pt").exists() and (run / "history.json").exists())
            self.assertEqual(len(job.history), 2)
            import numpy as np
            self.assertTrue(all(np.isfinite([h["loss_d"], h["loss_g"]]).all() for h in job.history))
            self.assertTrue(list((run / "samples").glob("*.png")))
            cfg2 = gt.GANConfig(**{**cfg.__dict__, "epochs": 3})
            job2 = jm.start("gan", "t2", gt.train_gan, cfg=cfg2, resume=True)
            while job2.is_active():
                time.sleep(0.2)
            self.assertEqual(job2.status, "completed", job2.error)
            self.assertEqual(len(job2.history), 3)             # resumed from epoch 2, trained 1 more
            imgs, labels = gt.generate_images("r1", [0, 2], 3, seed=1)
            self.assertEqual((len(imgs), imgs[0].size), (6, (32, 32)))
            a, _ = gt.generate_images("r1", [0], 2, seed=5); b, _ = gt.generate_images("r1", [0], 2, seed=5)
            self.assertTrue(np.array_equal(np.asarray(a[0]), np.asarray(b[0])))   # same seed -> same image
        finally:
            helpers.cleanup(tmp)


if __name__ == "__main__":
    unittest.main()
