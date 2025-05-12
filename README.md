# 🧠 Brain Tumor Image Segmentation Using Deep Neural Networks

This project utilizes deep learning techniques to automate **semantic segmentation** of brain tumors in **MRI scans**. By leveraging the **U-Net** architecture with encoder-decoder layers and skip connections, this system automatically detects and segments tumor regions, aiding in **neuro-oncology** diagnostics.

---

## 📌 Key Highlights

- **Model Architecture**: U-Net with encoder-decoder and skip connections
- **Dataset**: BRATS 2020 (Multimodal MRI scans for brain tumor segmentation)
- **Evaluation Metrics**: 
  - Dice Coefficient
  - Intersection over Union (IoU)
  - Precision
  - Recall
- **Framework**: TensorFlow 2.x / PyTorch
- **Output**: Binary segmentation masks highlighting tumor regions

---

## 🧠 Background

Brain tumor segmentation plays a critical role in the diagnosis and treatment of brain tumors. Traditional manual segmentation is labor-intensive and prone to inaccuracies. This project automates the process using deep learning models to improve both **accuracy** and **efficiency**, providing a scalable solution for medical professionals.

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
