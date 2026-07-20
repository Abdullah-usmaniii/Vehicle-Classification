# 🚗 Vehicle Classification & Image Enhancement Toolkit

A two-part computer vision toolkit for **detecting and classifying vehicles** in images and **cleaning up noisy or blurry images** before or after detection — all with an interactive viewer built in.

---

## 📦 What's Inside

| Tool | Script | What It Does |
|------|--------|--------------|
| **Vehicle Classifier** | `test_model.py` | Runs a fine-tuned YOLO model to detect and label vehicles in your images |
| **Noise Control** | `Noise_Control.py` | Cleans up noisy, blurry images and shows a side-by-side before/after comparison |

---

## ⚙️ Setup (One Time Only)

**Requirements:** Python 3.9 or higher

```bash
pip install -r requirements.txt
```

> This installs `opencv-python`, `numpy`, `matplotlib`, `scipy`, and `ultralytics` (for YOLO).

---

## 🚗 Part 1 — Vehicle Classifier

### What It Does

Loads a custom-trained **YOLO detection model** (`my_model.pt`) and runs it on every image in your `Images/` folder. For each image it:

- Draws **bounding boxes** around detected vehicles with class labels and confidence scores
- Saves an **annotated copy** of each image to a timestamped folder inside `Processed_Images/`
- Saves a **text file** per image listing detected classes and bounding box coordinates
- Opens an **interactive viewer** so you can flip through all results with your keyboard

### How to Run

1. Drop your images into the `Images/` folder (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` all work)
2. Run:
   ```bash
   python test_model.py
   ```
3. Wait while all images are processed — a progress counter prints in your terminal
4. The interactive viewer opens automatically when done

### Navigating the Viewer

| Key | Action |
|-----|--------|
| `→` / `Space` / `Enter` | Next image |
| `←` / `Backspace` | Previous image |
| `Q` / `Esc` | Quit viewer |

### Where Results Are Saved

```
Processed_Images/
└── HH-MM-SS_YYYY-MM-DD/
    ├── image1_processed.jpg    ← annotated image with bounding boxes
    ├── image1_processed.txt    ← class names + coordinates (one detection per line)
    ├── image2_processed.jpg
    └── image2_processed.txt
```

Each run creates its own timestamped subfolder — previous results are never overwritten.

### What the Model Detects

The model was trained on **YOLO11n** and fine-tuned for **100 epochs** on a vehicle dataset. It draws labeled bounding boxes around vehicle types in each image with a confidence score shown on each label.

> **Good to know:** The model was trained at `800×800` image resolution with strong augmentations (flips, HSV shifts, mixup). It performs best on clear, well-lit vehicle photos but handles varied conditions reasonably well.

---

## 🧹 Part 2 — Noise Control (Image Enhancer)

### What It Does

Takes noisy, blurry, or low-quality images and intelligently cleans them up using a multi-step enhancement pipeline — then shows you a **side-by-side before/after viewer** so you can judge the result yourself.

### Enhancement Pipeline (What Happens Under the Hood)

```
Input Image
    │
    ▼
① Upscale (2.5×) — Lanczos4 interpolation preserves edge sharpness
    │
    ▼
② Convert to YCrCb colour space
   Y  = Luminance (structure / detail)
   Cr = Red chroma
   Cb = Blue chroma
    │
    ▼
③ Chroma denoising — Gaussian blur kills colour noise without softening edges
    │
    ▼
④ Bilateral filter on luminance — smooths flat areas while protecting edges
    │
    ▼
⑤ Unsharp masking — sharpens blurry transitions without creating harsh halos
    │
    ▼
⑥ Recombine channels → final enhanced image
```

> The result: noise removed, edges sharpened, colours preserved — no charcoal artefacts, no blown-out contrast.

### How to Run

1. Drop your noisy/blurry images into the `noiseImages/` folder
2. Run:
   ```bash
   python Noise_Control.py
   ```
3. All images are processed and saved, then the viewer opens automatically

### Navigating the Before/After Viewer

The viewer shows **Before** (original) on the left and **After** (enhanced) on the right for every image processed in that run.

| Key / Button | Action |
|---|---|
| `→` / `D` / `Space` | Next image |
| `←` / `A` / `Backspace` | Previous image |
| Click **Next ▶** button | Next image |
| Click **◀ Previous** button | Previous image |

### Supported File Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.tif` · `.tiff` · `.webp`

### Where Results Are Saved

```
Processed_Images/
└── YYYY-MM-DD_HH-MM-SS/
    └── enhanced_yourimage.jpg
```

Each run creates its own timestamped folder — originals in `noiseImages/` are never modified.

---

## 📁 Project Structure

```
my_model/
│
├── test_model.py          ← Run the vehicle classifier
├── Noise_Control.py       ← Run the noise cleaner + viewer
├── my_model.pt            ← Fine-tuned YOLO model weights (~5.4 MB)
├── requirements.txt       ← All Python dependencies
│
├── Images/                ← Put images here for the classifier
├── noiseImages/           ← Put images here for noise cleaning
│
├── Processed_Images/      ← All outputs saved here (auto-created per run)
│   └── HH-MM-SS_YYYY-MM-DD/
│       ├── *_processed.jpg
│       └── *_processed.txt
│
└── train/                 ← Training artefacts (curves, confusion matrix, weights)
    ├── results.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── BoxF1_curve.png
    ├── BoxPR_curve.png
    └── weights/
        ├── best.pt
        └── last.pt
```

---

## ⚠️ Known Limitations

### Classifier
- The model detects vehicles that are **visible and reasonably sized** in the frame. Heavily occluded or tiny distant vehicles may be missed.
- Results are for **analytical purposes**. Do not use detections as sole legal or insurance evidence without independent confirmation.

### Noise Control
- The enhancer **recovers detail that's already there** but obscured by noise or blur. It **cannot reconstruct** detail that was genuinely destroyed by extreme downscaling, severe compression, or catastrophic blur — that information is gone at the pixel level.
- If an image still looks unclear after processing, treat it as unreadable — pushing the upscale factor or filter settings further will not help.

---

## 🔧 Dependencies

| Package | Minimum Version |
|---------|----------------|
| `opencv-python` | 4.8 |
| `numpy` | 1.24 |
| `matplotlib` | 3.7 |
| `scipy` | 1.10 |
| `ultralytics` | latest (for YOLO) |

---

## 🧠 Model Training Details

| Property | Value |
|----------|-------|
| Base architecture | YOLO11n |
| Training epochs | 100 |
| Image size | 800 × 800 |
| Optimizer | AdamW |
| Batch size | 16 |
| Dropout | 0.3 |
| Augmentations | Flip LR, HSV shift, Mixup (0.15), RandAugment, Erasing (0.4) |
| IoU threshold | 0.7 |
| Cosine LR schedule | ✅ |
| Transfer learning | Pretrained base, first 10 layers frozen |
