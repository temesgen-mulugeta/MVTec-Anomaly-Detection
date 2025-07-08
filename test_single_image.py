import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from processing.utils import load_model_HDF5
from autoencoder.losses import ssim_loss, l2_loss

# --------- USER CONFIGURATION ---------
# Path to the trained model directory (update if needed)
TIMESTAMP = "08-07-2025_12-39-53"
MODEL_DIR = f"saved_models/my_dataset/pump_house/mvtecCAE/ssim/{TIMESTAMP}"
MODEL_PATH = os.path.join(MODEL_DIR, "mvtecCAE_b8_e2.hdf5")

# Path to the image you want to test (update this!)
IMAGE_PATH = "my_dataset/pump_house/test/defect/IMG_1956_aug_1.png"  # <-- CHANGE THIS

# --------- LOAD MODEL AND INFO ---------
model, info, _ = load_model_HDF5(MODEL_PATH)

# --------- PREPROCESS IMAGE ---------
# Read as grayscale
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Could not load image: {IMAGE_PATH}")
    sys.exit(1)

# Resize to model input shape
shape = tuple(info["preprocessing"]["shape"])
img = cv2.resize(img, shape)

# Rescale
rescale = info["preprocessing"]["rescale"]
img = img.astype(np.float32) * rescale

# Add batch and channel dimensions
img_input = np.expand_dims(img, axis=(0, -1))  # shape: (1, H, W, 1)

# --------- RECONSTRUCT ---------
reconstructed = model.predict(img_input)

# --------- COMPUTE ERROR ---------
dynamic_range = info["preprocessing"]["dynamic_range"]
loss_type = info["model"]["loss"]

if loss_type == "ssim":
    error = ssim_loss(dynamic_range)(img_input, reconstructed).numpy()
    error_map = np.abs(img_input[0, :, :, 0] - reconstructed[0, :, :, 0])
    print(f"SSIM error: {error}")
elif loss_type == "l2":
    error = l2_loss(img_input, reconstructed).numpy()
    error_map = np.abs(img_input[0, :, :, 0] - reconstructed[0, :, :, 0])
    print(f"L2 error: {error}")
else:
    error = np.mean(np.abs(img_input - reconstructed))
    error_map = np.abs(img_input[0, :, :, 0] - reconstructed[0, :, :, 0])
    print(f"Mean abs error: {error}")

# --------- VISUALIZE ---------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(img_input[0, :, :, 0], cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Reconstruction")
plt.imshow(reconstructed[0, :, :, 0], cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Abs Error")
plt.imshow(error_map, cmap="hot")
plt.axis("off")

plt.tight_layout()
plt.show() 