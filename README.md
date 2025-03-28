# ViT-Glial-Tumor-Classification

This repository implements a Vision Transformer (ViT) model for classifying histopathological images of glial tumors into three major categories: Glioblastoma (GBM), Astrocytoma (Astros), and Oligodendroglioma (Oligos). The pipeline includes dataset preparation, balanced sampling, model fine-tuning, training/validation with Weights & Biases logging, and evaluation with confusion matrix visualization.

## 🧠 Dataset
- Custom dataset with JPG images grouped by tumor type.
- Balanced sampling: Up to 6000 samples per class.
- Preprocessing: Center crop, normalization, and data augmentation.

## 🧠 Model
- Pretrained Vision Transformer from [Kaiko AI](https://huggingface.co/1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms).
- Classification head fine-tuned on the glial tumor dataset.
- Last 5 transformer blocks are also fine-tuned for improved learning.

## ⚙️ Features
- Stratified train/val/test split
- Training with early stopping
- W&B integration for real-time monitoring
- Confusion matrix visualization on test results

## 📦 Requirements
- Python 3.8+
- PyTorch
- Torchvision
- timm
- Weights & Biases (wandb)
- scikit-learn
- seaborn
- matplotlib

## 🚀 Usage
```bash
# Train the model
python train.py

# Evaluate on test set
python test.py
