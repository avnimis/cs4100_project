"""
Local Photo Deletion Pipeline
==============================
Since the Google Photos Library API is restricted for new projects (shut down
for third-party access in 2024), this pipeline runs your trained model on a
local folder of photos exported from Google Takeout.

How to get your photos:
  1. Go to takeout.google.com
  2. Deselect all → select only "Google Photos"
  3. Export and download the zip
  4. Unzip into a folder, e.g. cs4100_project/takeout_photos/

Then run:
  python run_local_pipeline.py --folder takeout_photos/
  python run_local_pipeline.py --folder takeout_photos/ --threshold 0.7

Output:
  - results/suggested_deletions/  ← copies of photos to delete
  - results/report.json           ← full prediction report
  - results/report.html           ← visual report you can open in browser
"""

import json
import shutil
import argparse
from pathlib import Path
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent
MODEL_WEIGHTS = PROJECT_ROOT / "best_photo_classifier.pth"
RESULTS_DIR   = PROJECT_ROOT / "results"

from photo_model import ConvNet, transform, DEVICE

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".gif"}

# Labels matching your training folders — for display in the report
# (the model only predicts keep/delete, but we label delete reasons for clarity)
DELETE_CATEGORIES = {
    "blurry", "clothing_tag", "lecture_notes", "photo_of_paper",
    "presentation_photo", "products", "qr_code", "random_object",
    "receipt", "screenshot_chat", "screenshot_wallpaper", "similar_duplicate"
}


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model() -> ConvNet:
    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_WEIGHTS}\n"
            "Run photo_model.py first to train and save the model."
        )
    model = ConvNet().to(DEVICE)
    model.load_state_dict(torch.load(str(MODEL_WEIGHTS), map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_WEIGHTS}\n")
    return model


# ---------------------------------------------------------------------------
# Predict on a single local image file
# ---------------------------------------------------------------------------
def predict_image(model: ConvNet, img_path: Path) -> float:
    """Returns P(delete) in [0,1], or -1.0 on error."""
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor).item()
            return torch.sigmoid(torch.tensor(logit)).item()
    except Exception as e:
        print(f"  Warning: could not process {img_path.name} ({e})")
        return -1.0


# ---------------------------------------------------------------------------
# Collect all image files from a folder (recursively)
# ---------------------------------------------------------------------------
def collect_images(folder: Path) -> list[Path]:
    images = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    ]
    print(f"Found {len(images)} images in {folder}\n")
    return sorted(images)


# ---------------------------------------------------------------------------
# Generate HTML report
#    Creates a visual page showing each flagged photo with its delete
#    probability — useful for your class presentation demo.
# ---------------------------------------------------------------------------
def generate_html_report(results: list[dict], output_path: Path):
    rows = ""
    for r in sorted(results, key=lambda x: x["prob_delete"], reverse=True):
        color = "#ffcccc" if r["decision"] == "DELETE" else "#ccffcc"
        rows += f"""
        <tr style="background:{color}">
            <td>{r['filename']}</td>
            <td>{r['decision']}</td>
            <td>{r['prob_delete']:.1%}</td>
            <td>{r['prob_keep']:.1%}</td>
        </tr>"""

    n_delete = sum(1 for r in results if r["decision"] == "DELETE")
    n_keep   = len(results) - n_delete

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Photo Deletion Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h1   {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #333; color: white; }}
        tr:hover {{ opacity: 0.85; }}
    </style>
</head>
<body>
    <h1>AI Photo Deletion Suggestions</h1>
    <div class="summary">
        <strong>Total scanned:</strong> {len(results)} &nbsp;|&nbsp;
        <strong>Suggested deletions:</strong> {n_delete} ({n_delete/max(len(results),1):.1%}) &nbsp;|&nbsp;
        <strong>Keep:</strong> {n_keep} ({n_keep/max(len(results),1):.1%})
    </div>
    <table>
        <tr>
            <th>Filename</th>
            <th>Decision</th>
            <th>P(Delete)</th>
            <th>P(Keep)</th>
        </tr>
        {rows}
    </table>
</body>
</html>"""

    output_path.write_text(html)
    print(f"HTML report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(folder: str, threshold: float = 0.5):
    input_folder = Path(folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Folder not found: {input_folder}")

    # Create output directories
    deletions_dir = RESULTS_DIR / "suggested_deletions"
    deletions_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Local Photo Deletion Pipeline")
    print("=" * 60 + "\n")

    model  = load_model()
    images = collect_images(input_folder)

    if not images:
        print("No images found. Check the folder path and supported extensions.")
        return

    results   = []
    n_flagged = 0

    print(f"Running model on {len(images)} photos (threshold={threshold})...\n")

    for i, img_path in enumerate(images):
        prob_delete = predict_image(model, img_path)
        if prob_delete < 0:
            continue  # skip unreadable files

        decision = "DELETE" if prob_delete >= threshold else "KEEP"

        if decision == "DELETE":
            # Copy flagged photo to results/suggested_deletions/
            shutil.copy2(img_path, deletions_dir / img_path.name)
            n_flagged += 1

        results.append({
            "filename":    img_path.name,
            "full_path":   str(img_path),
            "prob_delete": round(prob_delete, 4),
            "prob_keep":   round(1 - prob_delete, 4),
            "decision":    decision,
        })

        # Progress every 20 photos
        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            print(f"  [{i+1}/{len(images)}] — {n_flagged} flagged so far")

    # Save JSON report
    json_out = RESULTS_DIR / "report.json"
    json_out.write_text(json.dumps(results, indent=2))

    # Save HTML report
    html_out = RESULTS_DIR / "report.html"
    generate_html_report(results, html_out)

    # Print summary
    n_keep = len(results) - n_flagged
    print(f"""
{"=" * 60}
  SUMMARY
  Photos scanned  : {len(results)}
  Flagged DELETE  : {n_flagged} ({n_flagged/max(len(results),1):.1%})
  Flagged KEEP    : {n_keep} ({n_keep/max(len(results),1):.1%})

  Suggested deletions copied to:
    {deletions_dir}

  Open the HTML report in your browser:
    {html_out}
{"=" * 60}
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",    type=str,   required=True,
                        help="Path to folder of photos to scan")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Delete probability threshold (default: 0.5)")
    args = parser.parse_args()
    run_pipeline(folder=args.folder, threshold=args.threshold)