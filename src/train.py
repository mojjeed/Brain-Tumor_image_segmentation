import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_unet
from sklearn.model_selection import train_test_split
import cv2
import argparse

def load_data(image_dir, mask_dir, img_size=(128, 128)):
    images, masks = [], []

    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, img_size) / 255.0
        mask = cv2.resize(mask, img_size) / 255.0

        images.append(img[..., np.newaxis])
        masks.append(mask[..., np.newaxis])

    return np.array(images), np.array(masks)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / (denominator + 1e-6)

def main(args):
    print("[INFO] Loading data...")
    images, masks = load_data("data/processed/images", "data/processed/masks")

    x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    print("[INFO] Building model...")
    model = build_unet(input_shape=(128, 128, 1))
    model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

    checkpoint = ModelCheckpoint("models/unet_brain_tumor.h5", save_best_only=True, verbose=1)
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    print("[INFO] Training model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint, early_stop]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
