# ViT-Glial-Tumor-Classification

This repository implements a Vision Transformer (ViT) model for classifying histopathological images of glial tumors into three major categories: Glioblastoma (GBM), Astrocytoma (Astros), and Oligodendroglioma (Oligos) at **Diamandis Lab**. The pipeline includes dataset preparation, balanced sampling, model fine-tuning, training/validation with Weights & Biases logging, and evaluation with confusion matrix visualization.

## 🧠 Dataset
- Custom dataset with JPG images grouped by tumor type.
- Balanced sampling: Up to 6000 samples per class.

## 🧠 Model
- Pretrained Vision Transformer from [Kaiko AI](https://huggingface.co/1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms).
- Classification head fine-tuned on the glial tumor dataset.
- Last 5 transformer blocks are also fine-tuned for improved learning.
  
## 🏋️ Training

- Stratified train/val/test split (60/20/20).
- Training includes W&B logging and model checkpointing.
- Early stopping is used to avoid overfitting.

---

## 📊 Results

The model achieved **strong classification performance**:

- ✅ **Best Validation Accuracy**: **94.92%** (Epoch 2)  
- 🧪 **Final Test Accuracy**: **94.64%**

### 📈 Training Summary

```text
Epoch 1: Train Loss=0.2780, Train Acc=89.70%, Val Loss=0.1461, Val Acc=94.75%
Epoch 2: Train Loss=0.1255, Train Acc=95.54%, Val Loss=0.1464, Val Acc=94.92%
Epoch 3: Train Loss=0.0878, Train Acc=96.97%, Val Loss=0.1793, Val Acc=93.94%
Epoch 4: Train Loss=0.0585, Train Acc=98.17%, Val Loss=0.1644, Val Acc=94.33%
Epoch 5: Train Loss=0.0424, Train Acc=98.49%, Val Loss=0.1857, Val Acc=94.83%


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


