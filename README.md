# 🧠 Brain Tumor Image Segmentation using Deep Neural Networks

This project implements a deep learning-based pipeline to perform **semantic segmentation of brain tumors** in MRI scans. Leveraging the power of CNN architectures like **U-Net**, this system automatically identifies and segments tumor regions to aid in medical diagnosis.

---

## 📌 Key Highlights

- ⚙️ Model: U-Net with encoder-decoder and skip connections
- 📊 Dataset: BRATS 2020 dataset (Multimodal MRI scans)
- 📈 Metrics: Dice Coefficient, IoU, Precision, Recall
- 🔧 Framework: TensorFlow 2.x / PyTorch
- 🧪 Output: Binary segmentation masks highlighting tumor regions

---

## 🧠 Background

Brain tumor segmentation is critical in neuro-oncology. Manual segmentation is time-consuming and error-prone. This project automates the process using deep convolutional networks, improving accuracy and speed in clinical workflows.

---

## 🗂️ Project Structure
```mermaid
graph TD
    A[data/] --> A1[raw/]
    A --> A2[processed/]
    B[models/]
    C[notebooks/]
    D[src/] --> D1[dataloader.py]
    D --> D2[model.py]
    D --> D3[train.py]
    D --> D4[evaluate.py]
    D --> D5[infer.py]
    E[outputs/]
    F[requirements.txt]
    G[README.md]

