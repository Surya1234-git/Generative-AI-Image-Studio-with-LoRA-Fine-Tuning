# Testing report (what is covered and how it was checked)

## Automated tests: `python -m unittest discover -s tests`
65 tests. **Without PyTorch, 57 run and 8 skip automatically; on a Colab GPU runtime all 65 are meant to run** (run the notebook's unit-test cell first).

Covered by the tests that run without a GPU:
* Dataset Explorer logic: scan, class / duplicate / corrupt detection, near-duplicate hashing, leakage-safe stratified split, preprocessing modes,
  split-folder export, ZIP safety (path traversal), quarantine of corrupted files, **caption-length statistics, the synthetic shapes dataset,
  caption priority (captions.csv > .txt > class template) and captions kept through ZIP upload and preprocessing**.
* Storage: image + metadata round trip (PNG text chunk + SQLite), search and filters, rename / delete, persistence, experiments and prompt history, backup, retention purge.
* Job manager: progress, exclusive-job guard, pause / resume / stop, failure capture, checkpoint requests.
* Maths and metrics: cosine similarity, ranking, PCA / k-means / Spearman, CSV / NPY export, PSNR / SSIM, accuracy / confusion matrix / ROC-AUC
  (perfect, chance and missing-class cases), FID refusal on too few images.
* Request validation, high-quality preset, aspect ratios, GPU memory checks, LoRA compatibility (SD 1.5 vs SDXL), adapter rename / delete / export,
  PDF / CSV / JSON reports, notebook structure.
* COCO captions importer (`tests/test_coco_import.py`): local and downloaded images, category folders, caption choice, seeded subset, bad downloads skipped, and the result opening in the Dataset Explorer's caption logic. The real COCO download itself was not run here (no internet).
* `tools/static_check.py` (undefined names, bad internal imports, misspelled module attributes) over all Python files: clean.

Tests that need PyTorch / diffusers / peft (skipped without them):
* `tests/test_torch_modules.py`: GAN output shapes, attention only in the attention model, attention rows sum to 1, text conditioning,
  full train -> checkpoint -> resume -> generate smoke test.
* `tests/test_lora_scaling.py`: a saved LoRA adapter loads at the same strength it was trained at, also when alpha differs from rank (regression test).

## Exercised by hand on a Google Colab T4 GPU during development
Stable Diffusion 1.5 and SDXL generation with the image actions (regenerate, variation, reuse settings), Dataset Explorer (synthetic shapes, preprocessing, split),
GAN Studio training of CGAN and Attention CGAN, the Evaluation Center (reference classifier, GAN evaluation, SD 1.5 vs SDXL comparison, CLIPScore, SSIM / PSNR / LPIPS),
and LoRA training. The automated tests cannot cover the Streamlit pages themselves (layout, widget behaviour); those were checked in the browser.

## Known limits
* GAN quality depends on training length: short runs on small datasets can collapse to near-identical samples and score at chance level in the classifier
  evaluation. Train longer (150 to 200 epochs at image size 32), try the `hinge` loss, and judge GANs by accuracy, the confusion matrix and the sample images.
* FID with fewer than 2048 images per set is only a rough comparison.
* Colab starts every session with a fresh disk: keep the project ZIP on Google Drive and re-run the launcher notebook after a disconnect.
