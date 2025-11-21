import cv2
import numpy as np
import os
from tqdm import tqdm
import sys
sys.stdout.reconfigure(encoding='utf-8')


# =============================
# CONFIGURATION
# =============================
INPUT_DIR = r"D:\\Landmark_dataset_aug"  # Input dataset (augmented or clean)
OUTPUT_DIR = r"D:\\Landmark_dataset_aug_Salient"   # Output for saliency images
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Method: 'crop' = crop salient region; 'mask' = highlight salient areas
MODE = 'crop'  # change to 'mask' if you want the full image with salient regions emphasized

# Saliency detector (Spectral Residual)
saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

# =============================
# FUNCTION DEFINITIONS
# =============================
def generate_saliency_map(image):
    """Return saliency map for an image."""
    success, saliency_map = saliency_detector.computeSaliency(image)
    if not success or saliency_map is None:
        return None
    saliency_map = (saliency_map * 255).astype("uint8")
    return saliency_map

def extract_salient_region(image, saliency_map, mode='crop'):
    """Return cropped or masked salient region."""
    if mode == 'mask':
        # Blend saliency heatmap on top of original
        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    # Threshold and find bounding box for cropping
    _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image  # fallback to full image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Slightly expand the box for context
    margin = int(0.05 * max(w, h))
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, image.shape[1] - x)
    h = min(h + 2 * margin, image.shape[0] - y)

    cropped = image[y:y+h, x:x+w]
    return cropped


# =============================
# MAIN LOOP
# =============================
print(f"\n Generating saliency-based dataset from: {INPUT_DIR}")

for landmark in tqdm(os.listdir(INPUT_DIR), desc="Processing landmarks"):
    landmark_path = os.path.join(INPUT_DIR, landmark)
    if not os.path.isdir(landmark_path):
        continue

    output_path = os.path.join(OUTPUT_DIR, landmark)
    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(landmark_path):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        input_img_path = os.path.join(landmark_path, file)
        output_img_path = os.path.join(output_path, file)

        # Read image
        image = cv2.imread(input_img_path)
        if image is None:
            print(f" Skipping unreadable file: {input_img_path}")
            continue

        # Generate saliency map
        saliency_map = generate_saliency_map(image)
        if saliency_map is None:
            continue

        # Extract salient area
        salient_img = extract_salient_region(image, saliency_map, MODE)

        # Save result
        cv2.imwrite(output_img_path, salient_img)

print("\n Saliency extraction complete!")
print(f"Processed dataset saved to: {OUTPUT_DIR}")
