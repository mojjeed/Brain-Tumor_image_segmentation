import cv2
import numpy as np
import tensorflow as tf
import argparse
from model import build_unet

def load_image(path, img_size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    return img[..., np.newaxis]

def save_mask(mask, output_path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask)

def main(args):
    print("[INFO] Loading model...")
    model = build_unet(input_shape=(128, 128, 1))
    model.load_weights("models/unet_brain_tumor.h5")

    print(f"[INFO] Reading image: {args.image_path}")
    img = load_image(args.image_path)
    input_tensor = np.expand_dims(img, axis=0)  # shape: (1, H, W, 1)

    print("[INFO] Predicting mask...")
    pred_mask = model.predict(input_tensor)[0]
    pred_mask = (pred_mask > 0.5).astype(np.float32)

    output_path = args.output_path or "outputs/predicted_mask.png"
    save_mask(pred_mask, output_path)
    print(f"[INFO] Saved mask to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to input MRI image")
    parser.add_argument("--output_path", help="Where to save predicted mask")
    args = parser.parse_args()
    main(args)
