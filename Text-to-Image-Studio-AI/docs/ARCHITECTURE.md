# Architecture

## Layers
| Layer | Folder | Role |
|---|---|---|
| UI | `app.py`, `views/` | Streamlit pages: Home, AI Image Generator, Dataset Explorer, Text Embeddings, GAN Studio, Evaluation Center, LoRA Fine-tuning, Image Gallery, About, Settings |
| Logic | `modules/` | The actual ML and analysis code, free of UI code so it can be unit-tested |
| Services | `services/` | Background jobs (`job_manager`), GPU checks (`gpu_manager`), model loading (`model_manager`), storage (`storage_manager`: files + SQLite) |
| Config | `config/settings.py` | Model registry (SD 1.5, SDXL), paths, defaults |
| Tests | `tests/`, `tools/static_check.py` | 57 unit tests and a static name/import checker |

## Modules and tasks
| Module | What it does | Task |
|---|---|---|
| `modules/lora_training.py`, `lora_registry.py` | LoRA fine-tuning of Stable Diffusion 1.5 / SDXL on a custom image + caption dataset; adapter management | 1 |
| `modules/gan_models.py` | `CGANGenerator`, `CGANDiscriminator` (label or text conditioning, projection discriminator); `SelfAttention2d` and the `AttnCGAN*` classes (Attention variants); `attention_overlay` for attention maps | 2, 5 |
| `modules/text_embeddings.py` | CLIP tokenizer and text encoder (Hugging Face Transformers), Sentence-Transformers, cosine similarity, PCA / UMAP, k-means, rank agreement; `class_text_table` builds the caption embeddings used to condition the GAN | 3 |
| `modules/dataset_explorer.py`, `modules/coco_import.py` | Import (ZIP, images, folder, Oxford-102, synthetic shapes, COCO captions subset via `tools/import_coco_subset.py`), statistics (classes, resolution, caption length), duplicates / corruption checks, preprocessing, leakage-safe split | 4 |
| `modules/gan_trainer.py` | Training loop (BCE or hinge loss, pause / resume / stop, checkpoints), `generate_images`, `text_to_image` | 2, 5, 6 |
| `modules/evaluation.py`, `modules/metrics.py` | Reference classifier accuracy / confusion matrix / ROC-AUC, FID, CLIPScore, SSIM, PSNR, LPIPS, reports | all |
| `modules/image_generator.py` | Stable Diffusion text-to-image, img2img, upscaling, LoRA loading | 1 |

## Text-to-image pipeline (Task 6)
```
text prompt
   |  CLIPTokenizer                    (Hugging Face)      -> token IDs (77 tokens max)
   v
CLIP text encoder                      (Hugging Face)      -> 512-d embedding (pooled vector)
   |  condition encoder inside the generator
   v
generator:  noise z (100-d)  +  condition  ->  transposed-convolution stack  ->  64x64 image
                                                  (Attention CGAN: self-attention block at 16x16)
   v
image  ->  Image Gallery  ->  Evaluation Center (classifier metrics, FID)
```
Training: the generator is trained against a discriminator that sees the image and its condition. With text conditioning the
condition is the CLIP embedding of the class caption, so a typed prompt is mapped to the nearest learned caption in embedding
space. With label conditioning the condition is the class index.

## Attention (Task 5)
`SelfAttention2d` is a SAGAN-style block. It is the only difference between `CGAN` and `Attention CGAN`, so the two can be
compared fairly with identical settings. Cross-attention between text tokens and image features is not implemented.

## Training data flow
Dataset Explorer writes a dataset (class folders + optional `captions.csv`). GAN Studio and LoRA Fine-tuning read it, write
checkpoints / adapters and sample grids, and the Gallery stores generated images with their prompt, model, seed and settings.
