# mri-quality-gate

# MRI Quality Gate – Synthetic MRI Artifact Generation Pipeline

This project implements a synthetic MRI artifact generation pipeline for developing and evaluating automated image quality assessment models.

The system simulates realistic MRI artifacts on 2D slices and enables controlled dataset generation for training and benchmarking machine learning–based quality control systems.

---

## Overview

The pipeline loads clean MRI slices (ANALYZE `.img` format), applies one or multiple synthetic artifacts, and assigns a final quality label.

Artifacts can be applied individually or in realistic combinations.

For combined artifacts, the final quality label is defined as the worst quality among the applied artifacts.

---

## Implemented Artifacts

The following artifact types are currently supported:

- Gaussian Noise (additive, intensity-based)
- Blur (Gaussian spatial blur)
- Bias Field (multiplicative low-frequency intensity inhomogeneity)
- Motion (FFT-based pseudo motion artifact in k-space)

Artifacts are applied in a realistic order:

1. bias_field  
2. motion_fft  
3. blur  
4. noise  

---

## Quality Labels

Each generated sample is assigned one of three quality classes:

- `ok`
- `borderline`
- `reject`

For multi-artifact samples, quality aggregation follows a worst-case rule.

---

## Dataset Generation Logic

For each MRI slice:

1. Load clean image.
2. Randomly sample number of artifacts (1–3).
3. Randomly select artifact types without repetition.
4. Apply artifacts sequentially.
5. Assign final quality label.

The pipeline supports controlled randomness via a fixed random seed for reproducibility.

---

## Project Structure

```
src/
│
├── artifacts/
│   ├── noise.py
│   ├── blur.py
│   ├── bias_field.py
│   ├── motion_fft.py
│
├── dicom/
│   ├── io.py
│   ├── series.py
│
├── data/
│   ├── build_dataset.py
│   ├── splits.py
├── models/
│   ├── train.py
├── explainability/
│   ├── gradcam.py

```

---

## Requirements

Python 3.10+

Install dependencies inside a virtual environment:

```
pip install -r requirements.txt
```

---

## Running (Debug Mode)

From project root:

```
python -m src.data.build_dataset
```

This currently runs a temporary visualization loop displaying:

- clean image
- corrupted image
- absolute difference

---

## Planned Extensions

- Export generated dataset to `.npy` or `.png`
- Automatic CSV metadata logging
- Subject-level train/validation/test split
- Baseline ML model for quality classification
- Artifact severity calibration
- 3D volume support

---

## Author

Aleksandra Suchanek  
Biomedical Engineering – Medical Imaging and Machine Learning
