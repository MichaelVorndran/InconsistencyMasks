# Study A: Enhancing SOTA Methods (Cityscapes Benchmark)

This directory contains the implementation for **Study A** of the paper: *"Inconsistency Masks: Harnessing Model Disagreement for Stable Semi-Supervised Segmentation"*.

In this study, we demonstrate that Inconsistency Masks (IM) act as a **general enhancement framework** for existing state-of-the-art methods. By filtering residual noise from pseudo-labels, IM consistently boosts performance on the Cityscapes benchmark across ResNet-50 and DINOv2 backbones.

## 📂 Structure

*   `Study_B_Cityscapes.ipynb`: The main entry point. A Jupyter Notebook designed to run the training and evaluation loops on Google Colab or a local GPU server.
*   `configs/`: Contains the JSON configuration files with the exact hyperparameters used to run the baselines (Suppervised, iMAS, U2PL, UniMatch, UniMatch v2 and SegKC) and their IM-enhanced versions in the Colab notebook.

## 🛠️ Setup

### Prerequisites
*   **GPU:** A GPU with at least 12GB VRAM (24GB+ recommended for DINOv2).
*   **Python:** 3.8+
*   **Environment:** PyTorch

### Installation
```bash
pip install -r requirements.txt
```

## 📊 Running the Benchmark

### 1. Data Preparation
1. Download the **Cityscapes dataset** from the [official website](https://www.cityscapes-dataset.com/).
2. Extract the following archives:
   - `leftImg8bit_trainvaltest.zip`
   - `gtFine_trainvaltest.zip`
3. Update the dataset path in `Study_B_Cityscapes.ipynb` to point to your local directory.

### 2. Execution
The entire reproduction pipeline is encapsulated in the `Study_B_Cityscapes.ipynb` notebook. This notebook covers:

* **Baseline Training:** Training the base methods (e.g., UniMatch) to generate initial checkpoints.
* **IM Generation:** Loading checkpoints to generate Inconsistency Masks.
* **Student Training:** Training the final student model using IM-filtered supervision.
