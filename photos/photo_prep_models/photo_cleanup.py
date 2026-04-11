from PIL import Image
import os

DATA_DIR = "photos"
removed = 0
kept = 0

for class_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, class_folder)
    if not os.path.isdir(folder_path):
        continue

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            continue

        fpath = os.path.join(folder_path, fname)
        try:
            with Image.open(fpath) as img:
                img.convert("RGB")  # force full load + decode
            kept += 1
        except Exception as e:
            print(f"Removing corrupt file: {fpath} ({e})")
            os.remove(fpath)
            removed += 1

print(f"\nDone. Kept {kept} images, removed {removed} corrupt files.")