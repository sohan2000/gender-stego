# GenderStego

GenderStego is a research prototype that combines **gait‑based gender recognition** with **privacy‑preserving video processing** and **encrypted video steganography**.

Instead of using faces or raw appearance, GenderStego works only with 2D skeletons extracted from surveillance‑style video, then removes visible identity while still allowing useful analytics on movement and coarse demographics.

> Master’s Project — San José State University, Department of Computer Science (May 2026)  
> Author: Sohan Vallapureddy

---

## What this project does (high level)

Given an input surveillance‑style video, GenderStego:

1. Estimates 2D skeletons for each person and classifies **binary gender from gait** using a Spatial‑Temporal Graph Convolutional Network (ST‑GCN).
2. Removes visible people from the scene and replaces them with **colour‑coded skeletons** (blue = male, red = female, green = unknown), so only motion and gender label remain.
3. **Encrypts and hides the original person regions** inside the person‑free (inpainted) video using DCT‑based steganography and AES‑256‑CTR + HMAC.
4. Supports **authorised reconstruction** of the original footage using the cryptographic keys and stored metadata.

The result is a “public” skeletal video that exposes only motion and gender labels, while the original appearance is encrypted and recoverable only by authorised parties.

---

## Core components

- **Stage 1 – Gait gender classifier**
  - Pose estimation with YOLOv8x‑Pose (17 COCO keypoints per frame) and centroid tracking across frames.
  - Conversion towards the OpenPose joint layout used by DenseGait.
  - ST‑GCN (`GaitSTGCN`, 2,035,905 parameters) for binary gender classification from skeleton sequences.
  - Trained on the **DenseGait** dataset (in‑the‑wild, surveillance‑style skeleton data).

- **Stage 2 – Privacy module**
  - Person segmentation with YOLOv8x‑Seg to obtain person masks and bounding boxes.
  - Background estimation + inpainting to remove people from frames.
  - Skeleton overlay with gender‑coded colours on top of the inpainted background.
  - Produces:
    - A **public skeletal video** (movement + gender, no visible appearance).
    - An **inpainted cover video** (people replaced by background). The Stage 3 notebook rebuilds this cover itself and hides the data in it.

- **Stage 3 – Steganography + reconstruction**
  - **AES‑256‑CTR** encryption of each person crop (fresh random IV per frame), plus **HMAC‑SHA256** over IV‖ciphertext.
  - Frequency‑domain hiding using **4×4 DCT blocks** inside the person bounding box only.
  - Default configuration β = 30, α = 0.05 (“QP27B30”).
  - Reconstruction path that extracts the DCT payload from stego frames, decrypts the person crops and pastes them back.
  - Videos are written with the lossless **FFV1** codec; a lossy codec destroys the hidden data.
  - The HMAC check fails on every honest reconstruction, because extraction never returns the ciphertext exactly. It only distinguishes a substituted video by the garbled output, not by the tag.

---

## Results

- **Gait gender classification (DenseGait, in‑the‑wild skeletons)**
  - Validation accuracy **72.65%** after 20 epochs (train 73.14%, gap 0.49%), on 6,000 clips held out from a random 30,000‑clip training subset (24,000 train). Still improving at epoch 20. History and curves: [`results/training/`](results/training/).
  - **Held‑out test set** (42,706 DenseGait clips never used in training or validation; [`results/evaluation/`](results/evaluation/), produced by `Evaluate_GaitSTGCN.ipynb`):

    | Metric | Value (95% CI) |
    |---|---|
    | Accuracy | **73.1%** (72.7–73.6) |
    | Majority‑class baseline (always "female") | 62.8% |
    | Balanced accuracy | 68.5% (68.0–68.9) |
    | F1 female / male | 0.80 / 0.58 |
    | Macro F1 | 0.69 (0.69–0.70) |
    | Precision / recall, female | 0.75 / 0.87 |
    | Precision / recall, male | 0.69 / 0.50 |
    | ROC‑AUC | 0.774 (0.770–0.779) |

  - The model leans towards the majority class: it finds 87% of female‑labelled clips but only 50% of male‑labelled ones.
  - On the 15,793 test clips with confident labels (DenseGait `Female` score ≤ 0.2 or ≥ 0.8), accuracy is 88.7% and ROC‑AUC 0.92, so a large share of the errors sit on clips whose automatic label is itself uncertain.
  - Balanced accuracy by DenseGait's auto‑generated attributes ranges from 0.52 (`Boots`, n = 212) and 0.58 (`FormalWear`, n = 208; `AgeChild`, n = 341) to 0.75 (`Sandals`, n = 627), and is 0.66 / 0.69 / 0.71 for `BodyThin` / `BodyNormal` / `BodyFat`. See `eval_subgroups.csv`. This is a slice analysis on automatic labels, not a fairness audit; DenseGait has no ethnicity attribute.

- **Steganography (encrypted pipeline in this repo)** — vid2‑1 (2560×1440), first 60 frames, QP27B30; [`results/stego/`](results/stego/)
  - Stego PSNR **43.0 dB** (cover vs. stego), reconstruction PSNR **23.0 dB** (original vs. reconstructed, whole frame).
  - The reconstructed person is recognisable but heavily speckled. Two causes, measured on real frames:
    1. The DC coefficient is left unchanged, so the byte at position (0,0) of every 4×4 block is never hidden; extraction guesses it from its neighbours. That is harmless for a natural image but makes 1 in 16 ciphertext bytes random, and accounts for about 88% of the reconstruction error.
    2. Even the hidden bytes come back only approximately (about 10% exact). XOR decryption turns any wrong ciphertext byte into a random pixel.
  - Saving the stego video as MJPG (as earlier versions did) loses the hidden data almost entirely (19.7 dB on the same frames; the person is not recognisable). The notebook now uses lossless FFV1, at about 7× the file size.

- **Steganography (thesis figures)** — QP27B30, 300 frames of `hall-org`; [`results/Sohan_Code_ResultsQP27.xlsx`](results/Sohan_Code_ResultsQP27.xlsx)
  - Stego PSNR 34.48 dB, reconstruction PSNR 36.51 dB, similarity 99.34%. Raising β increases stego PSNR (36.66 dB at β = 100) but lowers reconstruction quality.
  - These come from an earlier **unencrypted** pipeline that is not included. Neither notebook here reproduces them: the encrypted notebook gives the figures above, and the thesis‑era unencrypted notebook (`Stego_and_Reconstructed_Videos.ipynb`) reconstructs vid2‑1 at 11.9 dB (16.8 dB with a lossless codec).

- **Qualitative tests** on real‑world clips: works best for single‑person and small‑group walking scenes; degrades with cyclists (model trained on walking only), sunset lighting (static background model) and crowds (centroid tracker ID swaps).

---

## Repository layout

```
gender-stego/
├── notebooks/
│   ├── data/                 # DenseGait unzip, file check, sample inspection
│   ├── stage1_pose_gender/   # ST-GCN training, model summary, pose estimation
│   ├── stage2_privacy/       # background, inpainting, gender-coloured overlay, blackout video
│   └── stage3_stego/         # AES + DCT steganography, reconstruction, metrics (+ thesis-era unencrypted version)
├── results/
│   ├── training/             # training_history.csv + per-metric plots
│   ├── evaluation/           # held-out test metrics (eval_metrics.json, eval_subgroups.csv)
│   ├── stego/                # encrypted-pipeline metrics (vid2-1, 60 frames)
│   └── Sohan_Code_ResultsQP27.xlsx   # thesis β sweep (30–100) metrics
├── docs/
│   ├── Vallapureddy_Sohan_thesis.pdf
│   ├── GenderStego_slides.pptx
│   └── cs298-project-proposal.pdf
├── requirements.txt
└── LICENSE
```

### Pipeline order

| Step | Notebook | Produces |
|---|---|---|
| 0 | `notebooks/data/Step1_Unzip_DenseGait.ipynb`, `file_check.ipynb` | `DenseGait/poses/*.npy`, `annotations_present.csv` |
| 1a | `notebooks/stage1_pose_gender/stgcn_final_code.ipynb` | `best_model.pth`, `training_history.csv` |
| 1a′ | `notebooks/stage1_pose_gender/Evaluate_GaitSTGCN.ipynb` | `eval_metrics.json` (precision/recall/F1, ROC‑AUC, CIs), `eval_predictions.csv`, `eval_subgroups.csv` |
| 1b | `notebooks/stage1_pose_gender/Multi_Pose_Estimator.ipynb` | `<vid>-pose_estimations.json` (tracked 17‑joint poses) |
| 2a | `notebooks/stage2_privacy/Background_Image_Extractor.ipynb` | `<vid>-BG_frame.jpg` |
| 2b | `notebooks/stage2_privacy/Multi_Object_Inpainting.ipynb` | `<vid>-output_inpainted.mp4` |
| 2c | `notebooks/stage2_privacy/Multi_Pose_Overlay_Gender.ipynb` | `<vid>-pose_gender_overlay.mp4` (public video) |
| 3 | `notebooks/stage3_stego/Multi_Stego_and_Reconstructed_Video_Results.ipynb` | cover, stego and reconstructed videos (FFV1), `*_crypto.npy`, metrics `.xlsx` |

Each notebook starts with a short header describing its inputs, outputs and known issues. All outputs have been stripped.

## Implementation notes (code vs. thesis)

The code is the reference. Where it differs from the thesis text:

**Stage 1 – classifier**
- **Not wired into the video pipeline.** The overlay notebook takes gender per track ID from a hand‑filled `GENDER_BY_ID`; running `GaitSTGCN` on tracked video poses is future work.
- **Training run:** a random 30,000‑clip subset of the 175,247 clips on Drive, unstratified 80/20 split, batch 64, Adam (lr 1e‑3, weight decay 1e‑4), `StepLR(step_size=5, gamma=0.5)`, mixed precision, no gradient clipping (the thesis states batch 32, ReduceLROnPlateau, clipping 1.0, stratified split; the saved checkpoint confirms the code's settings).
- **Sequence length:** clips are not resampled to 64 frames. Each batch is zero‑padded to its longest clip (capped at 300 frames), and the model averages over padded frames, so a clip's score depends slightly on its batch‑mates. `Evaluate_GaitSTGCN.ipynb` scores clips one at a time.
- **Split not recoverable:** the 30k subset was drawn without a seed and its indices were not saved, so the 6,000‑clip validation set cannot be rebuilt; precision/recall/F1 are reported on the separate held‑out test set. DenseGait has no person IDs, so the same person appearing in a training clip and a test clip cannot be ruled out.
- **Skeleton graph:** edge `(16, 18)` refers to a non‑existent joint and is dropped, so joint 16 is a leaf.
- **Pose format at inference:** `Pose_Estimator_18_Joints.ipynb` emits 19 joints (neck and mid‑hip both added) in pixel coordinates, whereas training used 18 neck‑centred DenseGait joints. The overlay draws the 17‑joint COCO skeleton with keypoint confidence 0.5 (thesis: 0.45).

**Stage 2 – privacy module**
- **Background:** per‑pixel BGR median of up to 100 frames sampled across the whole video (thesis: HSV Gaussian mean of the first 40 frames).
- **Segmentation:** YOLOv8x‑Seg default confidence (0.25, thesis: 0.45); elliptical 25×25 close and 15×15 dilate, plus a 21×21 feather when inpainting.
- **Bounding box:** one box per frame around all people together (thesis: one per person); width and height floored to multiples of 4, origin not snapped.

**Stage 3 – steganography**
- `QP = 27` only names files; no codec quantisation is applied in the encrypted notebook. α clips the payload to [α, 1−α] rather than scaling it.
- Data is hidden in the inpainted frame without skeletons.

---

## Setup

The notebooks were written for **Google Colab** (Python 3.12, NVIDIA A100) with data on Google Drive.

1. `pip install -r requirements.txt`
2. Obtain the datasets yourself (see below) and place DenseGait under a `DenseGait/` folder with `poses/` and `annotations.csv`.
3. Edit the path variables at the top of each notebook (`/content/...` or `/content/drive/MyDrive/Gait Project/...`) to match your layout.
4. Supply your own test videos. No videos are distributed here.

The stego notebook generates an AES key and an HMAC key on first run and writes them to `encryption_keys.txt`. The per‑frame IV/HMAC log goes to `*_crypto.npy`. Both are git‑ignored. Keep them private and never commit them.

### Datasets (not included)

- **DenseGait** (Cosma & Rădoi, *Sensors* 2022): ~217K in‑the‑wild skeleton sequences with auto‑generated attributes. Credentialed access, licensed **CC BY‑NC‑ND 4.0**.
- **OU‑ISIR Gait Database with Pose Sequences**: used only for early validation of the ST‑GCN training loop.

You must obtain both directly from their maintainers and respect their terms.

---

## Ethical use and limitations

GenderStego is a research prototype exploring **privacy‑aware video analytics**. It has important limitations:

- It currently supports **binary gender labels only** (Male/Female) and **has not** been audited for fairness across age, ethnicity, body type, or other demographics.
- DenseGait labels are automatically generated and may contain noise.
- The system works best for walking under relatively clean surveillance conditions and is less reliable for:
  - Non‑walking activities (e.g., cycling, running),
  - Crowded scenes with many occlusions,
  - Rapidly changing lighting.
- It is not real‑time (~6–8 FPS on a mid‑range desktop GPU, per the thesis).

Any real‑world deployment would require:

- A thorough **fairness and bias evaluation** across demographic groups,
- A **formal privacy and legal review** (e.g., DPIA under GDPR‑like regimes),
- Clear governance around who can access decryption keys and under what conditions.

For now, this project should be treated as a **research‑only prototype**. It is not production software.

---

## License

Code in this repository is released under the [MIT License](LICENSE). The thesis, slides and proposal in `docs/` remain © 2026 Sohan Vallapureddy, all rights reserved. Datasets are not covered and are not redistributed.

---

## Acknowledgements

This project was completed as part of **CS 298 (Master’s Project)** at San José State University.

- Primary advisor: **Dr. Nada Attar**  
- Committee members / collaborators: **Dr. Noha Elfiky**, **Dr. Safwat Hamad**, **Dr. Ahmed Elhadad**, **Dr. Amith Kamath Belman**

Thanks also to the authors and maintainers of DenseGait, OU‑ISIR, YOLOv8, and the original ST‑GCN works that this project builds upon.
