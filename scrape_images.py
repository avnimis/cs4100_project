from bing_image_downloader import downloader
import os, shutil

TARGET = 100

SEARCH_QUERIES = {
    "blurry":               "blurry out of focus photograph",
    "clothing_tag":         "clothing label tag closeup",
    "important_photos":     "people portraits high quality photo",
    "lecture_notes":        "handwritten lecture notes notebook",
    "photo_of_paper":       "photo of paper document flat lay",
    "presentation_photo":   "powerpoint presentation slide projected",
    "products":             "product photography item white background",
    "qr_code":              "qr code scan",
    "random_object":        "random everyday household object photo",
    "receipt":              "store receipt paper photo",
    "screenshot_chat":      "iphone text message screenshot",
    "screenshot_wallpaper": "phone home screen wallpaper screenshot",
    "similar_duplicate":    "nearly identical similar photos",
}

BASE_DIR = "photos"
TEMP_DIR = "temp_downloads"

for folder, query in SEARCH_QUERIES.items():
    save_dir = os.path.join(BASE_DIR, folder)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    needed = max(0, TARGET - existing)

    if needed == 0:
        print(f"[{folder}] Already has {existing} images, skipping.")
        continue

    print(f"[{folder}] Has {existing}, downloading {needed} more...")

    # bing_image_downloader saves into its own subfolder structure
    # so we download to temp then move files over
    downloader.download(
        query,
        limit=needed,
        output_dir=TEMP_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=10,
        verbose=False
    )

    # Move downloaded files into the correct photos/ subfolder
    temp_subfolder = os.path.join(TEMP_DIR, query)
    if os.path.exists(temp_subfolder):
        moved = 0
        for fname in os.listdir(temp_subfolder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                shutil.move(
                    os.path.join(temp_subfolder, fname),
                    os.path.join(save_dir, fname)
                )
                moved += 1
        print(f"  → Moved {moved} images to photos/{folder}/")

# Cleanup temp folder
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

print("\nDone! Check your photos/ folders.")
