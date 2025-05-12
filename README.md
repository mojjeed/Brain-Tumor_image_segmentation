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
brain-tumor-segmentation/
├── data/                   # Dataset folder (MRI slices, segmentation masks)
│   ├── raw/                # Raw BRATS data
│   └── processed/          # Preprocessed images and masks
├── models/                 # Trained model checkpoints
├── notebooks/              # Jupyter notebooks for training, EDA, testing
├── src/
│   ├── dataloader.py       # Data loading and augmentation
│   ├── model.py            # U-Net model architecture
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Model evaluation
│   └── infer.py            # Inference script
├── outputs/                # Segmented output images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
