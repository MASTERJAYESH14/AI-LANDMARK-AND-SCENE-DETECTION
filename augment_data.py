import os
import random
import shutil
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# =============================
# CONFIG
# =============================
BASE_DIR = r"D:\\Landmark_dataset"          # Original dataset
AUG_DIR = r"D:\\Landmark_dataset_aug" # Output directory
os.makedirs(AUG_DIR, exist_ok=True)

# Target image counts per landmark
def get_target_count(img_count):
    if img_count < 30:
        return 200  # strong augmentation
    elif img_count < 40:
        return 180  # medium
    elif img_count < 60:
        return 180  # light
    else:
        return img_count  # no augmentation if already large enough

# Define augmentation pipelines
augment_gentle = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.85, 1.15],
    shear_range=5,
    horizontal_flip=True,
    fill_mode='nearest'
)



def choose_augmenter(img_count):
    if img_count < 30:
        return augment_gentle
    elif img_count < 40:
        return augment_gentle
    elif img_count < 60:
        return augment_gentle
    else:
        return None


# =============================
# AUGMENTATION FUNCTION
# =============================
def augment_landmark(landmark_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    images = [f for f in os.listdir(landmark_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    img_count = len(images)
    target = get_target_count(img_count)
    augmenter = choose_augmenter(img_count)

    # Copy originals first
    for f in images:
        shutil.copy(os.path.join(landmark_path, f),
                    os.path.join(output_path, f))

    if target <= img_count or augmenter is None:
        return 0, img_count

    n_to_generate = target - img_count
    gen_count = 0
    print(f"\n Augmenting {os.path.basename(landmark_path)} ({img_count} -> {target})")

    while gen_count < n_to_generate:
        random.shuffle(images)  # make sure all originals get used fairly
        for img_file in images:
            if gen_count >= n_to_generate:
                break
            img_path = os.path.join(landmark_path, img_file)
            img = load_img(img_path)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # generate one augmented image per loop
            for batch in augmenter.flow(x, batch_size=1):
                new_name = f"aug_{gen_count}_{os.path.basename(img_file)}"
                array_to_img(batch[0]).save(os.path.join(output_path, new_name))
                gen_count += 1
                break

    return gen_count, target


# =============================
# MAIN SCRIPT
# =============================
log_lines = []

for landmark in tqdm(os.listdir(BASE_DIR), desc="Processing landmarks"):
    landmark_path = os.path.join(BASE_DIR, landmark)
    if not os.path.isdir(landmark_path):
        continue

    output_path = os.path.join(AUG_DIR, landmark)
    augmented, total = augment_landmark(landmark_path, output_path)
    log_lines.append(f"{landmark}: {total} images ({augmented} new)")

# =============================
# SAVE LOG
# =============================
log_file = os.path.join(AUG_DIR, "augmentation_log.txt")
with open(log_file, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("\n Augmentation complete!")
print(f"Log saved to {log_file}")
