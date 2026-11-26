# Text-to-Image Studio AI 2.0

An all-in-one Generative AI web app built with **Streamlit** and run on **Google Colab (free T4 GPU)**.
It covers the full text-to-image workflow in one place: Stable Diffusion image generation, dataset analysis,
text embeddings, conditional GANs (with and without attention), model evaluation, LoRA fine-tuning and a persistent image gallery.

Built for the **ElevanceSkills "Real Time Text-to-Image Generator" internship** (six tasks, see the task table below).

---

## Table of contents

1. [What you can do with it](#1-what-you-can-do-with-it)
2. [Internship tasks and where to find them](#2-internship-tasks-and-where-to-find-them)
3. [Repository layout](#3-repository-layout)
4. [What you need](#4-what-you-need)
5. [Run it on Google Colab, step by step](#5-run-it-on-google-colab-step-by-step)
6. [Where your outputs are saved](#6-where-your-outputs-are-saved)
7. [Guided tour: how to use every page and view the output](#7-guided-tour-how-to-use-every-page-and-view-the-output)
8. [Run it on your own computer (optional)](#8-run-it-on-your-own-computer-optional)
9. [Automated tests](#9-automated-tests)
10. [Troubleshooting](#10-troubleshooting)
10b. [Architecture and results](#architecture-and-results)
11. [Honest notes and limitations](#11-honest-notes-and-limitations)
12. [Credits and licences](#12-credits-and-licences)

---

## 1. What you can do with it

| Page | What it does |
|---|---|
| **AI Image Generator** | Stable Diffusion 1.5 / SDXL text-to-image, image-to-image, variations, upscaling, LoRA adapters, prompt helpers |
| **Dataset Explorer** | Upload or create a dataset, then analyse it: classes, image resolution, duplicates, corrupted files, caption length, preprocessing, train/validation/test split, export |
| **Text Embeddings** | See how text becomes tokens and vectors (CLIP / Sentence-Transformers), compare prompts by cosine similarity, 2D plots |
| **GAN Studio** | Train a conditional GAN (CGAN) and an attention-based CGAN, watch training live, generate images, view attention maps, run the text-to-GAN pipeline |
| **Evaluation Center** | Measure models with numbers: classifier accuracy, confusion matrix, ROC-AUC for GANs; SD 1.5 vs SDXL comparison; CLIPScore, FID, SSIM / PSNR / LPIPS; PDF / CSV reports |
| **LoRA Fine-tuning** | Fine-tune Stable Diffusion 1.5 on your own captioned images, compare before / after, manage adapters |
| **Image Gallery** | Every image you generate is saved with its settings; search, filter, favourite, compare, back up |
| **Settings / About** | GPU and performance options, project information, task matrix |

---

## 2. Internship tasks and where to find them

| # | Task | Where in the app |
|---|---|---|
| 1 | Fine-tune a pre-trained text-to-image model on a custom dataset | **LoRA Fine-tuning** page |
| 2 | Conditional GAN that draws shapes from text labels | **GAN Studio → Train** (architecture `CGAN`); use the built-in **Synthetic shapes** dataset (square / circle / triangle) |
| 3 | Tokenise and embed text with Hugging Face Transformers | **Text Embeddings** page |
| 4 | Load and analyse a public dataset (classes, description length, image resolution, images with their text) | **Dataset Explorer** (Overview, Visualize and **Descriptions** tabs) |
| 5 | Improve a GAN with attention | **GAN Studio → Train** (architecture `Attention CGAN`) and the **CGAN vs Attention CGAN** tab |
| 6 | End-to-end pipeline: text → preprocessing → embedding → GAN image | **GAN Studio → Text → GAN pipeline** (train with CLIP text conditioning) |

Task 5 is measured in **Evaluation Center → GAN classifier metrics**, so you can compare the two GANs with numbers instead of eyeballing images.

---

## 3. Repository layout

```
Text-to-Image-Studio-AI/
├── README.md                 <- this file
├── app.py                    <- Streamlit entry point
├── requirements.txt          <- Python dependencies
├── config/                   <- settings, model registry, data folder logic
├── modules/                  <- core logic: generation, datasets, embeddings, GANs, evaluation, LoRA
├── views/                    <- one file per page of the web app
├── components/  services/  assets/   <- UI pieces, GPU / job / storage helpers, styles
├── notebooks/                <- Google Colab launcher notebook
├── sample_data/              <- ready-to-use sample datasets (see below)
├── docs/ARCHITECTURE.md      <- architecture and pipeline description
├── results/                  <- real results and screenshots (see results/RESULTS.md)
├── tests/  tools/  TESTING.md  CHANGELOG.md   <- automated tests, helper scripts, testing report, changes
└── data/  models/            <- empty folders (on Colab the real data goes to Google Drive)
```

---

## 4. What you need

| Item | Details |
|---|---|
| Google account | For Google Colab and Google Drive |
| Colab GPU | Free tier is enough: **Runtime → Change runtime type → T4 GPU** |
| Google Drive space | About **15 GB free** if you download both Stable Diffusion models (SD 1.5 about 4 GB, SDXL about 7 GB, CLIP about 0.6 GB). SD 1.5 alone is enough for most pages |
| Internet in Colab | Needed once to download models and to open the public link |
| Time | First start: 10 to 20 minutes (installs and model downloads). Later starts: a few minutes |

---

## 5. Run it on Google Colab, step by step

### Step 1: Put the project folder on Google Drive

The notebook looks for the project as a **ZIP file in the root of "My Drive"**.

1. If you have this project as **`Text-to-Image-Studio-AI.zip`**, use that file as it is.
   If you only have the folder, zip it so the ZIP is named exactly `Text-to-Image-Studio-AI.zip`
   (the folder `Text-to-Image-Studio-AI` must be *inside* the ZIP).
2. Open [drive.google.com](https://drive.google.com) and **upload `Text-to-Image-Studio-AI.zip` to My Drive** (not inside any sub-folder).

### Step 2: Open the notebook in Colab

Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** and choose
`notebooks/Text_to_Image_Studio_Colab.ipynb` from this project (or upload it to Drive, right-click it → **Open with → Google Colaboratory**).

### Step 3: Choose a GPU

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
Without a GPU the image and GAN pages are extremely slow or fail.

### Step 4: Run the cells in order

Run each cell from top to bottom (click the play button, or Shift + Enter). Wait for each cell to finish before starting the next.

| # | Cell | What it does | What you should see |
|---|---|---|---|
| 1 | GPU and CUDA check | Confirms a GPU is attached | GPU name (for example Tesla T4), memory, `CUDA available: True` |
| 2 | Mount Google Drive | Connects Drive so your work survives disconnects | A Google permission pop-up (allow it), then `Data directory: /content/drive/MyDrive/TextToImageStudio` |
| 3 | Get the project | Unzips the project from Drive (or clones from GitHub) | `Extracted ZIP to /content` or `Project already present` |
| 4 | Install dependencies | Installs the Python packages | Quiet output; finishes without red errors |
| 5 | *(Optional)* Pre-download models | Downloads SD 1.5 and CLIP ahead of time | `Model cache ready.` |
| 6 | *(Optional)* Run unit tests | Checks the code on your GPU | The last lines should say `OK` |
| 7 | **Launch the app** | Starts Streamlit and opens a public tunnel | A link like `https://something-random.trycloudflare.com` |

### Step 5: Open the app

Click the link printed by the last cell. The app opens in a new browser tab with the sidebar on the left
(Home, AI Image Generator, Dataset Explorer, Text Embeddings, GAN Studio, Evaluation Center, LoRA Fine-tuning, Image Gallery, About).
The top bar should show **GPU Connected (T4)**.

> **Important:** keep the Colab tab open while you use the app. The link changes every time you run the launch cell,
> and it stops working when the Colab session ends. Anything saved to Drive stays.

### Coming back after a disconnect

Colab deletes the temporary disk when the session ends, but your data on Drive stays.
Re-run the **GPU check, Drive mount, Get the project, Install dependencies and Launch** cells, then open the new link.

---

## 6. Where your outputs are saved

On Colab everything is stored in **Google Drive → My Drive → `TextToImageStudio`**:

| Folder or file | What is inside |
|---|---|
| `generated_images/` | Every image you generate (PNG with its settings embedded) |
| `datasets/` | Datasets you upload, create or process in the Dataset Explorer |
| `checkpoints/` | GAN training runs: checkpoints and sample images |
| `lora_models/` | LoRA adapters you train |
| `reports/` and `exports/` | PDF / CSV / JSON evaluation reports and exported files |
| `cache/` | Temporary working files |
| `studio.db` | Database behind the Gallery, experiment history and prompt history |
| `settings.json` | Your settings |

You can also download from inside the app: the **Image actions** buttons (PNG / JPEG), **Generate & export** (GAN samples),
**Reports & export** (PDF, CSV, JSON), and the Gallery backup.

---

## Sample data

`sample_data/lora_style_dataset.zip` holds 24 captioned images (512x512, one consistent flat-vector landscape style, captions end with the trigger phrase `sks flat vector style`).
Use it to test **LoRA Fine-tuning** quickly: upload the ZIP on the *Dataset & captions* tab.
For the GAN pages, create the shapes dataset inside the app (**Dataset Explorer → Upload → Synthetic shapes**); no download is needed.
`tools/make_style_dataset.py` regenerates the sample set (`python tools/make_style_dataset.py <output-folder>`).

---

## 7. Guided tour: how to use every page and view the output

Follow this order for a complete walkthrough. Each step says what to do and what you should see.

### 7.1 Home

Shows the status of the app and shortcuts to each page. Check that **GPU Connected (T4)** is shown at the top.

### 7.2 Dataset Explorer (Task 4)

1. **Upload** tab → Source **Synthetic shapes** → pick square, circle and triangle, about 200 images per shape, 64 px → name it `shapes_dataset` → **Upload / Import Dataset**.
   (Other sources: a ZIP of image folders, individual images, a folder on Drive, or the built-in Oxford-102 Flowers.)
2. **Overview**: number of classes, number of images, resolution, corrupted and duplicate files.
3. **Visualize**: class balance charts and sample images.
4. **Descriptions**: caption-length statistics (words and characters), a length histogram, and images shown next to their captions.
   Put a `captions.csv` (columns `filename`, `caption`) next to your images to use real descriptions.
   Datasets such as Oxford-102 only have templated captions, and the page tells you when that is the case.
   **A public dataset with real captions (COCO):** run `tools/import_coco_subset.py` in a Colab cell (it needs internet, which Colab has):
   ```
   !wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip
   !python tools/import_coco_subset.py --captions annotations/captions_val2017.json --instances annotations/instances_val2017.json --out /content/coco_subset --max-images 500
   ```
   Then choose Source **Folder path on this machine / Drive** and enter `/content/coco_subset`. (Official COCO download address; check cocodataset.org if it has moved. The download is about 240 MB and was not run when this README was written.)
5. **Preprocess & Split**: resize / crop, then split into train / validation / test without letting duplicates leak across the splits.
6. **Export**: download the processed dataset or its report.

### 7.3 Text Embeddings (Task 3)

1. Type one or more prompts.
2. **Vectors** tab: the tokens, token ids and the embedding shape produced by the Hugging Face tokenizer and encoder.
3. **Similarity**: cosine similarity between your prompts, plus 2D plots (PCA / UMAP).
4. Download the embeddings as CSV or NPY.

### 7.4 GAN Studio (Tasks 2, 5 and 6)

**Train tab**

1. Choose the dataset (for example `shapes_dataset`).
2. Architecture `CGAN`, Condition **Class label**.
3. Recommended starting point for the shapes dataset: **image size 32, 150 to 200 epochs, batch size 32**.
   GANs need enough updates before shapes appear: in our tests, 50 epochs on 600 images produced images that a classifier could not tell apart.
4. **Start training**. The dashboard shows status, epoch, generator and discriminator loss, loss curves and discriminator confidence.
   You can pause, resume or stop; checkpoints are saved to Drive.
5. Train a second run with architecture **Attention CGAN** and **identical settings**.
6. Train a third run with Condition **CLIP text embedding of the class caption** for the pipeline tab.

**Generate & export tab**: pick a run and generate images for each class, then download them. After training you should see rough but recognisable squares, circles and triangles.

**CGAN vs Attention CGAN tab**: side-by-side sample grids and **attention maps** (where the attention layer looks).

**Text → GAN pipeline tab**: type a sentence and click **Run pipeline** to see the tokens, the embedding and the generated image.
This run only saw the trained class captions, so it works best with sentences close to them, such as "a circle".

### 7.5 Evaluation Center (measure everything)

* **GAN classifier metrics**: choose the dataset and image size, click **Train reference classifier**
  (it should score close to 100% on real images), pick your GAN runs and click **Evaluate selected runs**.
  You get accuracy, a confusion matrix and one-vs-rest ROC-AUC per run. For 3 classes, chance level is 33%, so a working GAN scores clearly above that.
* **Diffusion comparison**: run the same prompts and seed on SD 1.5 and SDXL; compares time, GPU memory and CLIPScore.
  Run it twice and use the second run for fair timings, because the first run includes model loading.
* **Image metrics**: CLIPScore (text vs image), FID (needs at least 100 real and 100 generated images, and a single gallery type selected), SSIM / PSNR / LPIPS (two images of the same content).
* **Experiment history**: every evaluation you ran.
* **Reports & export**: download the PDF report, raw metrics (CSV), configuration (JSON) and a reproducibility report.

### 7.6 LoRA Fine-tuning (Task 1)

1. **Dataset & captions**: upload a ZIP of 15 to 30 images plus a `captions.csv`, or loose images, and name the set. Edit captions in the table and click **Save captions**.
   Use the same trigger phrase in every caption (for example `sks flat vector style`).
2. **Train**: base model Stable Diffusion 1.5, rank 8, alpha 16, learning rate 1e-4, resolution 512, batch size 1, about 10 epochs. The monitor shows progress and the loss curve.
3. **Test: before / after**: enter a prompt that contains your trigger phrase; the app generates the same prompt and seed with and without the adapter so you can compare.
4. **Manage adapters**: rename, delete or export adapters.

### 7.7 AI Image Generator

1. Enter a prompt, choose a model (SD 1.5 is fastest), resolution, steps and seed, then click **Generate Image**.
2. Under the image, **Image actions** let you download (PNG / JPEG), **Regenerate**, make a **Variation** (image-to-image),
   **Copy prompt**, **Reuse settings** (fills the form with the settings of that image), **Upscale**, view **Full screen**, mark **Favorite** or delete it.
3. Optional: apply a trained LoRA adapter from the same page.

### 7.8 Image Gallery

Browse every saved image with its metadata. Search, filter by model, date, size, tags or favourites, compare images, and back up the database.

---

## 8. Run it on your own computer (optional)

You need Python 3.10+ and an NVIDIA GPU for reasonable speed.

```bash
unzip Text-to-Image-Studio-AI.zip
cd Text-to-Image-Studio-AI

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# install PyTorch for your CUDA version first (see pytorch.org), for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
streamlit run app.py
```

Open the address Streamlit prints (usually `http://localhost:8501`).
Data is stored in `./data`. Set the environment variable `STUDIO_DATA_DIR` to use another folder.

---

## 9. Automated tests

From the project folder:

```bash
python -m unittest discover -s tests
```

* On a Colab GPU runtime, run the notebook's unit-test cell (cell 6). The run should end with `OK`.
* Without PyTorch, the GPU-dependent tests skip automatically.
* `TESTING.md` lists exactly what the tests cover.

---

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| No public link appears | Re-run the launch cell. Then run the last cell of the notebook to see the Streamlit log. Alternative tunnel: `!npx localtunnel --port 8501` |
| The link stopped working | The Colab session ended or disconnected. Re-run the cells listed under *Coming back after a disconnect*; you get a new link |
| `ImportError: Found an incompatible version of torchao` | Colab ships an old optional `torchao`. The notebook uninstalls it; if you still see the error, run `!pip uninstall -y torchao` and restart the app cell |
| `IndexError: index N is out of bounds ... with size N` while generating | A scheduler problem in some library versions. The app retries automatically with other schedulers and shows a warning. You can also choose `Euler` or `DDIM` in the scheduler list |
| Out of GPU memory | Use Stable Diffusion 1.5, a smaller resolution, batch size 1, or Settings → Performance → CPU offload = always |
| Streamlit shows red `StreamlitAPIException` / widget errors | Make sure you are using the latest version of this project folder, restart the app cell and open the new link in a fresh tab |
| Evaluation says "Train at least one GAN in GAN Studio first" | Train a GAN first; the page lists only completed runs |
| GAN accuracy is about 33% | The GAN is under-trained or has not learned its labels. Train longer (150 to 200 epochs), use image size 32, and check the sample images |
| FID says it needs at least 100 images | Generate more images, and select only **one** gallery type (for example `gan:CGAN`) in the FID tab |
| Variation or first generation is slow | The first run of each model includes downloading and loading it. Later runs are much faster |
| Colab stops the session when idle | Keep the tab open and active; free Colab limits GPU time per day |

If something fails, the page shows an error message with **Technical details**; copy it when asking for help.

---

## Architecture and results
Architecture, modules and the text-to-image pipeline (text -> CLIP tokenizer -> CLIP embedding -> conditional GAN -> image) are
described in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The exported reports, raw metrics (CSV), configuration (JSON),
reproducibility report and screenshots from the runs below are in [`results/`](results/) (see also `results/RESULTS.md`).
All numbers were read from the app on a Google Colab T4 GPU.

### Results summary

**Task 2 and 5: CGAN vs Attention CGAN** (dataset `shapes_dataset`: 3 classes, 600 images; class-label conditioning; 200 epochs,
32 px). Classifier evaluation = Evaluation Center -> GAN classifier metrics, 50 generated images per class, seed 0. The reference
classifier reaches 97.8% accuracy on real held-out images, which is the ceiling for generated images.

| Model | Accuracy | Macro AUC | FID | Generated images used for FID |
|---|---|---|---|---|
| CGAN (`cgan_1125_1304`) | 66.7% | 0.833 | 327.74 | 144 (600 real) |
| Attention CGAN (`attn_1125_1314`) | 100% | 1.000 | 306.92 | 192 (600 real) |

How to read these numbers (honest caveats):
* Each class collapsed to one picture: the label-conditioned runs ignore the noise input, so all samples of a class look the same.
  Because of this, accuracy only moves in steps of one class (66.7% = 2 of 3 classes recognised, 100% = 3 of 3) and AUC is equally
  coarse. The label does control the output (circle, square and triangle samples are clearly different shapes).
* The Attention CGAN scores better on both measures, but with this little variety and a gap of about one class it is not strong
  evidence that attention improves image quality.
* FID values above 300 mean the generated distribution is still far from the real images. The two FIDs used different numbers of
  generated images (144 vs 192) and fewer than 2048 per set, so they are a rough comparison only and should not be compared with
  published FIDs.

**Task 6: Text -> GAN pipeline** (Attention CGAN trained with CLIP text conditioning, run `attn_1125_1355`). Prompts typed in the
Text -> GAN pipeline tab are tokenized (`<|startoftext|> a photo of a triangle <|endoftext|>`), encoded by the CLIP ViT-B/32 text
encoder into a `[1, 512]` embedding, matched to the nearest training caption, and the generator draws from that embedding.
"a photo of a circle", "a photo of a triangle" and "a photo of a square" each matched the right class (cosine 0.97 for "triangle")
and produced the matching shape. Unlike the label-conditioned runs, the four samples per prompt differ in colour, size, background
and position. The images are small and blurry.

**Task 1: LoRA fine-tuning** (Stable Diffusion 1.5, 24 captioned training images): rank 8, alpha 16, learning rate 0.0001,
30 epochs, batch size 1, gradient accumulation 4, resolution 512, seed 42; training took 510 s and trained 1.59 M parameters.
The Test: before / after tab generates the base model and the adapter side by side with the same prompt and seed (checked at 30,
50 and 60 steps). The adapter clearly changes the output (clothing, face markings, ears), but for the general prompt used the
style change is subtle. A prompt from the training domain with the trigger phrase shows the effect best. The loss curve is noisy
and roughly flat (about 0.02 to 0.045 per epoch), as is usual for diffusion fine-tuning.

**Task 3 and 4:** see `results/RESULTS.md` (tokens, embeddings, similarity matrix, PCA map, wording test, and the dataset
statistics). The Experiment history tab lists every run above (reference classifier, both GANs, the FID runs and the LoRA
comparisons), and Reports & export produced the PDF report, raw metrics CSV, configuration JSON and reproducibility report.

---

## 11. Honest notes and limitations

* LoRA training supports **Stable Diffusion 1.5 and SDXL** (SDXL needs more VRAM: resolution 512 and batch size 1 on a 15 GB T4). An adapter only works with the model family it was trained on.
* Task 5 uses **self-attention** (a SAGAN block in the generator and discriminator). Cross-attention between text tokens and image features is not implemented; the task allows either.
* Oxford-102 has class labels, not human captions, so its captions are templated ("a photo of a ...").
* GAN images are small (32 to 64 px) and blurry. Judge them with the classifier metrics, FID and the sample grids, not the loss values.
* The label-conditioned GANs show no variety inside a class (mode collapse), which makes their classifier accuracy and AUC very coarse; the text-conditioned run produces varied samples. Reducing this collapse (for example with a mode-seeking loss term, longer or differently regularised training) is future work.
* The LoRA before / after difference is real but modest on a general prompt; more training steps or a larger and more varied dataset would strengthen it.
* Free-text generation in the Text → GAN pipeline only works well for sentences close to the captions the GAN was trained on.
* FID with fewer than 2048 images per set is only a rough comparison, and the app flags this.
* Results depend on training length and data size; small datasets give noisy scores.
* Unit tests run without a GPU; GPU features are meant to be run on a Colab T4. See `TESTING.md` for the details.

---

## 12. Credits and licences

* Developer: **Surya Prakash M**
* Built for the ElevanceSkills "Real Time Text-to-Image Generator" internship. The training-course project was only a conceptual reference; the code in this repository is a new implementation.
* Models keep their own licences: Stable Diffusion (CreativeML OpenRAIL-M / OpenRAIL++-M), CLIP (MIT).
* COCO (optional, via `tools/import_coco_subset.py`): annotations are licensed CC BY 4.0; the images come from Flickr and keep their own terms. Check cocodataset.org before redistributing any images.
