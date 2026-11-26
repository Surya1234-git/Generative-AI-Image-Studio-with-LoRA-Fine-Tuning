import os
import time
import unittest
from datetime import date

from config import settings
from services import storage_manager as st
from services.job_manager import JobBusyError, JobManager
from tests import helpers


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = helpers.fresh_data_root()
        st.init_db()

    def tearDown(self):
        helpers.cleanup(self.tmp)

    def _save(self, prompt, model="Stable Diffusion 1.5", seed=1, gen_type="txt2img", size=(40, 30)):
        return st.save_generated_image(helpers.noise_image(seed, size), dict(prompt=prompt, model=model, seed=seed, steps=30, guidance=7.5,
                                                                             scheduler="Euler", gen_type=gen_type, negative_prompt="blurry"))

    def test_save_and_metadata_roundtrip(self):
        rec = self._save("a red fox", seed=42)
        self.assertTrue(os.path.exists(rec["path"]))
        got = st.get_image(rec["id"])
        self.assertEqual((got["prompt"], got["seed"], got["steps"], got["width"], got["height"]), ("a red fox", 42, 30, 40, 30))
        from PIL import Image
        self.assertIn("a red fox", Image.open(rec["path"]).text["parameters"])

    def test_filters_use_real_metadata(self):
        a = self._save("a red fox", "Stable Diffusion 1.5", 1)
        b = self._save("a blue whale", "Stable Diffusion XL Base 1.0", 2, "variation", (80, 60))
        self.assertEqual([r["id"] for r in st.list_images(search="whale")], [b["id"]])
        self.assertEqual([r["id"] for r in st.list_images(models=["Stable Diffusion 1.5"])], [a["id"]])
        self.assertEqual([r["id"] for r in st.list_images(gen_types=["variation"])], [b["id"]])
        self.assertEqual([r["id"] for r in st.list_images(min_width=60)], [b["id"]])
        st.set_favorite(a["id"], True)
        self.assertEqual([r["id"] for r in st.list_images(favorites_only=True)], [a["id"]])
        st.set_tags(a["id"], ["Animals", " red "])
        self.assertEqual([r["id"] for r in st.list_images(tags=["animals"])], [a["id"]])
        self.assertEqual(len(st.list_images(date_from=date.today(), date_to=date.today())), 2)
        self.assertEqual(len(st.list_images(date_to=date(2000, 1, 1))), 0)
        self.assertEqual(st.count_images(search="a"), 2)

    def test_rename_delete_and_missing_file(self):
        rec = self._save("x")
        name = st.rename_image(rec["id"], "my pic")
        self.assertEqual(name, "my_pic.png")
        self.assertTrue(os.path.exists(st.get_image(rec["id"])["path"]))
        path = st.get_image(rec["id"])["path"]
        self.assertTrue(st.delete_image(rec["id"]))
        self.assertFalse(os.path.exists(path))
        self.assertIsNone(st.get_image(rec["id"]))
        self.assertFalse(st.delete_image(9999))

    def test_persistence_across_reconnect(self):
        rec = self._save("persist me")
        settings.set_data_root(self.tmp)            # simulates restarting the app on the same folder
        self.assertEqual(st.list_images()[0]["id"], rec["id"])

    def test_datasets_experiments_prompts_search_backup(self):
        st.register_dataset("flowers", self.tmp / "datasets" / "flowers", "labeled", 100, 5, {"a": 1})
        st.register_dataset("flowers", self.tmp / "datasets" / "flowers", "labeled", 120, 5)   # upsert
        self.assertEqual(st.get_dataset("flowers")["num_images"], 120)
        self.assertEqual(len(st.list_datasets()), 1)
        eid = st.add_experiment("fid", "test", {"k": 1}, {"fid": 12.5})
        self.assertEqual(st.list_experiments("fid")[0]["metrics"]["fid"], 12.5)
        st.add_prompt_history("hello world"); st.add_prompt_history("hello world")
        self.assertEqual(len(st.list_prompt_history()), 1)
        self._save("flower field")
        res = st.global_search("flower")
        self.assertTrue(res["images"] and res["datasets"])
        self.assertTrue(st.backup_database().exists())
        self.assertGreater(st.storage_usage()["generated_images"], 0)
        st.delete_experiment(eid)
        self.assertEqual(st.list_experiments("fid"), [])

    def test_purge_keeps_favorites(self):
        a = self._save("old one"); b = self._save("old fav", seed=2)
        st.set_favorite(b["id"], True)
        with st.connection() as c:
            c.execute("UPDATE images SET created_at='2000-01-01T00:00:00'")
        self.assertEqual(st.purge_images(30), 1)
        self.assertIsNone(st.get_image(a["id"]))
        self.assertIsNotNone(st.get_image(b["id"]))


class JobTests(unittest.TestCase):
    def test_runs_reports_progress_and_blocks_second_exclusive_job(self):
        jm = JobManager()
        def work(job, n):
            for i in range(n):
                job.check(); job.progress = (i + 1) / n; time.sleep(0.02)
            return "ok"
        j = jm.start("gan", "first", work, n=10)
        with self.assertRaises(JobBusyError):
            jm.start("gan", "second", work, n=1)
        while j.is_active():
            time.sleep(0.02)
        self.assertEqual((j.status, j.result, round(j.progress, 2)), ("completed", "ok", 1.0))
        jm.start("gan", "third", work, n=1)      # allowed once the first finished

    def test_pause_resume_stop(self):
        jm = JobManager()
        ticks = []
        def work(job):
            for _ in range(1000):
                job.check(); ticks.append(1); time.sleep(0.01)
        j = jm.start("gan", "p", work)
        time.sleep(0.1); j.pause(); time.sleep(0.15)
        self.assertEqual(j.status, "paused")
        n = len(ticks); time.sleep(0.15)
        self.assertEqual(len(ticks), n, "job kept running while paused")
        j.resume()
        for _ in range(50):
            if j.status == "running": break
            time.sleep(0.02)
        self.assertEqual(j.status, "running")
        j.stop()
        for _ in range(100):
            if not j.is_active(): break
            time.sleep(0.02)
        self.assertEqual(j.status, "stopped")

    def test_failure_is_captured_not_raised(self):
        jm = JobManager()
        def boom(job): raise RuntimeError("bad")
        j = jm.start("x", "boom", boom)
        time.sleep(0.2)
        self.assertEqual(j.status, "failed")
        self.assertIn("bad", j.error)

    def test_checkpoint_request_flag(self):
        jm = JobManager()
        def work(job):
            time.sleep(0.1)
            return job.checkpoint_requested()
        j = jm.start("x", "c", work)
        j.request_checkpoint()
        while j.is_active(): time.sleep(0.02)
        self.assertTrue(j.result)


if __name__ == "__main__":
    unittest.main()
