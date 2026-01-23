# Study B: Training from Scratch (Resource-Constrained)

## Overview
This directory contains the implementation for **Study B**.

In this study, we investigate resource-constrained scenarios where pre-trained backbones are unavailable. We evaluate **Inconsistency Masks (IM)** against standard SSL baselines (e.g., FixMatch, iMAS, U²PL) when training entirely from scratch.

> **Key Finding:** In this regime, IM acts as a **stabilizer**. While confidence-based methods often suffer from model collapse or stagnation due to early confirmation bias, IM effectively filters uncertainty to ensure robust convergence.

## 📂 Shared Codebase
The core logic is implemented in the shared Python scripts located in this directory. The dataset-specific folders contain the runner scripts that call these functions.

* **`functions.py`**: Contains the core implementations of all SSL methods (IM generation, FixMatch training loop, iMAS training loop, etc.) and augmentation logic.
* **`unet.py`**: Defines the lightweight 1x1 U-Net architecture used for all experiments in this study.
* **`evalnet.py`**: Defines the quality assessment network used for the IM++ variant.
* **`paths.py`**: Central file to configure dataset paths.
* **`config.ini`**: Central configuration for hyperparameters (learning rates, batch sizes, kernels) across all datasets.

## 🧪 Dataset Experiments
Detailed instructions and execution orders are provided in the `README.md` files within each subdirectory:

* **ISIC 2018** (Binary Skin Lesion Analysis)
* **HeLa** (Multi-Label Microscopy)
* **SUIM** (Multi-Class Underwater Imagery)
* **Cityscapes** (Multi-Class Urban Scenes - From Scratch)

## 🛠️ Usage
1. **Install dependencies:**
```bash
pip install -r requirements.txt
```
2. Update `paths.py` to point to your local data.
3. Navigate to a dataset folder (e.g., `cd SUIM/`) and follow the numbered execution steps in its README.
