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
