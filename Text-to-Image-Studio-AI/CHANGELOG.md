# Changelog

## 2.0.2
* Shapes dataset captions fixed for grammar ("an orange square", not "a orange square"); regression test added.
* New `modules/coco_import.py` and `tools/import_coco_subset.py`: build a COCO captions subset (real human captions, optional
  object-category folders) that the Dataset Explorer opens. Needs no GPU. Five unit tests (`tests/test_coco_import.py`).
* README and About page: LoRA training is documented as supporting Stable Diffusion 1.5 **and** SDXL (as the code does).
* Added `docs/ARCHITECTURE.md` and a `results/` folder with `RESULTS.md` and the real screenshots collected so far.

## 2.0.1
* LoRA adapter strength fixed: the adapter is saved with its rank / alpha metadata, so a loaded adapter runs at the strength it
  was trained with (regression test `tests/test_lora_scaling.py`).
* Outdated Colab `torchao` hidden from peft / diffusers (`config/__init__.py`), notebook uninstalls it.

## Known limitation: GAN quality at small scale
Short runs on small datasets can mode-collapse (near-identical samples, chance-level classifier accuracy). Train longer, try the
`hinge` loss, 32 px images, and judge GANs by accuracy, the confusion matrix, diversity, FID and the samples themselves.
