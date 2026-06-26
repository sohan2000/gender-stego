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
3. **Encrypts and hides the original person regions** inside the privacy‑safe skeletal video using DCT‑based steganography and AES‑256‑CTR + HMAC.
4. Supports **authorised reconstruction** of the original footage using the cryptographic keys and stored metadata.

The result is a “public” skeletal video that exposes only motion and gender labels, while the original appearance is encrypted and recoverable only by authorised parties.

---

## Core components

- **Stage 1 – Gait gender classifier**
  - Pose estimation with YOLOv8x‑Pose (17 COCO keypoints per frame).
  - Conversion to 18‑joint OpenPose format and fixed‑length pose sequences.
  - ST‑GCN (GaitSTGCN) for binary gender classification from skeleton sequences.
  - Trained on the **DenseGait** dataset (in‑the‑wild, surveillance‑style skeleton data).

- **Stage 2 – Privacy module**
  - Person segmentation with YOLOv8x‑Seg to obtain per‑person masks and bounding boxes.
  - HSV Gaussian background modelling + inpainting to remove people from frames.
  - Skeleton overlay with gender‑coded colours on top of the inpainted background.
  - Produces:
    - A **public skeletal video** (movement + gender, no visible appearance).
    - An **inpainted cover video** that acts as the carrier for steganography.

- **Stage 3 – Steganography + reconstruction**
  - **AES‑256‑CTR** encryption of each person crop, plus **HMAC‑SHA256** for tamper detection.
  - Frequency‑domain hiding using **4×4 DCT blocks** inside the inpainted person regions only.
  - Config tuned for imperceptibility and recoverability (e.g., QP=27, β=30, α=0.05).
  - Reconstruction path that:
    - Extracts DCT coefficients from stego frames,
    - Decrypts the person crops,
    - Re‑inserts them into the background to approximate the original video.

---

## Results (brief)

These results are from the associated thesis and experiments; this codebase is a prototype implementation:

- **Gait gender classification (DenseGait, in‑the‑wild skeletons)**
  - Validation accuracy: ~72.65% after 20 epochs.
  - Small train‑validation gap (~0.49%), suggesting no strong overfitting.
  - Model continues to improve at epoch 20 (conservative lower bound).

- **Privacy + steganography**
  - At the recommended configuration (QP=27, β=30, α=0.05):
    - Stego PSNR ≈ 34.5 dB (below human perceptual detection threshold for distortions).
    - Reconstruction PSNR ≈ 36.5 dB, similarity ≈ 99.3% vs original.
  - Qualitative tests on multiple real‑world clips:
    - Works best for single‑person and small‑group, walking scenes.
    - Degrades gracefully in crowded scenes, unusual activities (e.g., cycling), or challenging lighting (e.g., sunsets).

---

## Tech stack

- **Python**
  - PyTorch (ST‑GCN implementation and training)
  - Ultralytics YOLOv8 (pose + segmentation)
  - NumPy / OpenCV for pre‑/post‑processing
  - Cryptography / PyCryptodome for AES‑CTR + HMAC
- **MATLAB**
  - Video pre/post‑processing and DCT‑based steganography experiments
- **Datasets (not included in this repo)**
  - DenseGait (main training set for the gait classifier)
  - OU‑ISIR Gait Database with Pose Sequences (used only for early pipeline validation)

> Note: Due to licensing, the datasets themselves are **not** distributed here. You must obtain them directly from their respective sources and respect their terms.

---

## Repository status

This repository is intended primarily as a **research showcase** and reference implementation for my Master’s project. It is **not a production system**.

Some parts of the pipeline (e.g., training scripts, evaluation notebooks, MATLAB code for steganography) assume a local directory layout and datasets that are not checked in. Small adjustments will likely be required to run the full end‑to‑end system on your own machine.

If you are mainly interested in the ideas and architecture:

- See the **thesis PDF** and **presentation slides** in this repository for a deeper explanation of:
  - Design decisions and trade‑offs,
  - Experimental setup and metrics,
  - Limitations and future work.

---

## Ethical use and limitations

GenderStego is a research prototype exploring **privacy‑aware video analytics**. It has important limitations:

- It currently supports **binary gender labels only** (Male/Female) and **has not** been audited for fairness across age, ethnicity, body type, or other demographics.
- DenseGait labels are automatically generated and may contain noise.
- The system works best for walking under relatively clean surveillance conditions and is less reliable for:
  - Non‑walking activities (e.g., cycling, running),
  - Crowded scenes with many occlusions,
  - Rapidly changing lighting.

Any real‑world deployment would require:

- A thorough **fairness and bias evaluation** across demographic groups,
- A **formal privacy and legal review** (e.g., DPIA under GDPR‑like regimes),
- Clear governance around who can access decryption keys and under what conditions.

For now, this project should be treated as a **research‑only prototype**.

---

## Acknowledgements

This project was completed as part of **CS 298 (Master’s Project)** at San José State University.

- Primary advisor: **Dr. Nada Attar**  
- Committee members / collaborators: **Dr. Noha Elfiky**, **Dr. Safwat Hamad**, **Dr. Ahmed Elhadad**, **Dr. Amith Kamath Belman**

Thanks also to the authors and maintainers of DenseGait, OU‑ISIR, YOLOv8, and the original ST‑GCN works that this project builds upon.
