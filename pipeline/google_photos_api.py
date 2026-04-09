"""
Google Photos Pipeline
======================
Connects your trained ConvNet photo classifier to your real Google Photos
account. Creates an album called "AI - Suggested Deletions" and populates
it with photos your model predicts should be deleted.

Process:
  1. OAuth2 authentication  → get permission to access the account
  2. Fetch media items       → get list of photos from Google Photos
  3. Run model inference     → download each photo, predict keep/delete
  4. Create album            → make "AI - Suggested Deletions" in Google Photos
  5. Batch add flagged items → add all delete-predicted photos to the album

Setup (one time):
  pip install google-auth google-auth-oauthlib google-auth-httplib2 requests Pillow
  → Place credentials.json in your project root before running.

Usage:
  python pipeline/google_photos_api.py
  python pipeline/google_photos_api.py --max 50 --threshold 0.7
"""

import io
import sys
import json
import argparse
import requests
import torch
from pathlib import Path
from PIL import Image

# Google auth libraries
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ---------------------------------------------------------------------------
# Paths — adjust if your project layout differs
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
CREDENTIALS     = PROJECT_ROOT / "credentials.json"   # downloaded from Google Cloud
TOKEN           = PROJECT_ROOT / "token.json"          # auto-created after first login
MODEL_WEIGHTS   = PROJECT_ROOT / "best_photo_classifier.pth"


# Add models/ folder to path so we can import photo_model
sys.path.insert(0, str(PROJECT_ROOT / "models"))
from photo_model import ConvNet, transform, DEVICE

# ---------------------------------------------------------------------------
# 1. OAuth2 — Scopes
#
#    A "scope" tells Google exactly what permissions your app needs.
#    We request the minimum needed:
#      - readonly:      read the user's photo library
#      - sharing:       create albums and add photos to them
#
#    We deliberately do NOT request a scope that allows deletion,
#    because the Photos API doesn't offer one — and it's safer anyway.
#    Your "Suggested Deletions" album is a review step, not auto-delete.
# ---------------------------------------------------------------------------


SCOPES = [
    "https://www.googleapis.com/auth/photoslibrary.readonly",
    "https://www.googleapis.com/auth/photoslibrary.appendonly",
]

# Base URL for all Photos Library API calls
PHOTOS_API = "https://photoslibrary.googleapis.com/v1"



# ---------------------------------------------------------------------------
# 2. Authentication
#
#    How this works:
#      - First run: no token.json exists → opens browser → you log in →
#        Google returns a token → we save it as token.json
#      - Later runs: token.json exists → load it → if expired, auto-refresh
#        using the refresh_token inside (no browser needed)
#
#    The Credentials object is then attached to every API request as a
#    Bearer token in the Authorization header, proving to Google that
#    the user has granted us access.
# ---------------------------------------------------------------------------
def authenticate() -> Credentials:
    creds = None

    # Try loading existing token first
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)

    # If no valid credentials, run the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Token exists but expired — silently refresh it
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            # First time — open browser for user login
            if not CREDENTIALS.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS}\n"
                    "Download it from Google Cloud Console → APIs & Services "
                    "→ Credentials → OAuth 2.0 Client IDs → Download"
                )
            print("Opening browser for Google login...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS), SCOPES
            )
            # This starts a local server on port 0 (any free port),
            # opens your browser, and waits for Google to redirect back
            creds = flow.run_local_server(port=0)

        # Save token so future runs skip the browser step
        TOKEN.write_text(creds.to_json())
        print(f"Token saved to {TOKEN}")

    print("Authentication successful.\n")
    return creds


# ---------------------------------------------------------------------------
# 3. API Helper
#
#    All Google Photos API calls are just HTTP requests with:
#      - Authorization: Bearer <token>   ← proves who you are
#      - Content-Type: application/json  ← we're sending/receiving JSON
#
#    We wrap this in a helper so we don't repeat the header setup everywhere.
#    If the response isn't 2xx, we raise an error with the full response body
#    so you can see exactly what Google complained about.
# ------------------------------------------------------------------------

def get_headers(creds: Credentials) -> dict:
    """
    Always refresh credentials before building headers.
    This ensures we never use a stale access token — the token
    stored in creds.token can expire mid-session, so we explicitly
    refresh it each time before making a request.
    """
    creds.refresh(Request())
    return {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }


def api_get(creds: Credentials, endpoint: str, params: dict = None) -> dict:
    url = f"{PHOTOS_API}/{endpoint}"
    resp = requests.get(url, headers=get_headers(creds), params=params)
    if not resp.ok:
        raise RuntimeError(f"GET {endpoint} failed {resp.status_code}: {resp.text}")
    return resp.json()


def api_post(creds: Credentials, endpoint: str, body: dict) -> dict:
    url = f"{PHOTOS_API}/{endpoint}"
    resp = requests.post(url, headers=get_headers(creds), json=body)
    if not resp.ok:
        raise RuntimeError(f"POST {endpoint} failed {resp.status_code}: {resp.text}")
    return resp.json()



# ---------------------------------------------------------------------------
# 4. Fetch Media Items
#
#    The Photos API returns photos in pages. Each page has up to 100 items
#    and a nextPageToken. We keep fetching until there's no token left
#    (meaning we've seen all photos) or we hit the user's max limit.
#
#    Each media item has:
#      - id          : unique Google Photos ID (used to add to albums)
#      - filename    : original filename
#      - baseUrl     : temporary URL to download the image (expires in ~1 hour)
#      - mediaMetadata: width, height, creationTime, etc.
# ---------------------------------------------------------------------------
def fetch_media_items(creds: Credentials, max_items: int = 100) -> list[dict]:
    print(f"Fetching up to {max_items} photos from Google Photos...")
    items = []
    page_token = None

    while len(items) < max_items:
        # How many to request this page (API max is 100 per page)
        page_size = min(100, max_items - len(items))
        params = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token

        data = api_get(creds, "mediaItems", params=params)
        batch = data.get("mediaItems", [])
        items.extend(batch)

        page_token = data.get("nextPageToken")
        print(f"  Fetched {len(items)} items so far...")

        # No more pages
        if not page_token:
            break

    print(f"Total fetched: {len(items)} photos\n")
    return items


# ---------------------------------------------------------------------------
# 5. Model Inference
#
#    For each photo:
#      1. Download it from baseUrl (temporary Google CDN URL)
#         We append "=w256-h256" to request a small 256px version —
#         faster to download than full resolution, and our model only
#         needs 128x128 anyway.
#      2. Run it through the same transform pipeline used during training
#      3. Pass through ConvNet → get raw logit → sigmoid → probability
#      4. If prob_delete >= threshold, flag it
#
#    This is the same predict() logic from photo_model.py, but operating
#    on in-memory images from URLs instead of files on disk.
# ---------------------------------------------------------------------------
def load_model(model_path: Path) -> ConvNet:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}\n"
            "Train the model first by running: python photo_model.py"
        )
    model = ConvNet().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {model_path}\n")
    return model


def predict_from_url(model: ConvNet, url: str) -> float:
    """
    Download image from URL and return P(delete) in [0, 1].
    Returns -1.0 if the image can't be downloaded/processed.
    """
    try:
        # Request a small version for speed (=w256 means max width 256px)
        resp = requests.get(url + "=w256-h256", timeout=10)
        resp.raise_for_status()

        # Load into PIL, convert to RGB (handles PNG transparency, CMYK, etc.)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        # Apply the same transform as training (resize, normalize)
        tensor = transform(img).unsqueeze(0).to(DEVICE)  # add batch dim

        with torch.no_grad():
            logit = model(tensor).item()
            prob_delete = torch.sigmoid(torch.tensor(logit)).item()

        return prob_delete

    except Exception as e:
        print(f"    Warning: could not process image ({e})")
        return -1.0


# ---------------------------------------------------------------------------
# 6. Create Album
#
#    Simple POST to /albums with just a title.
#    Returns the new album's ID, which we need to add photos to it.
# ---------------------------------------------------------------------------
def create_album(creds: Credentials, title: str) -> str:
    print(f"Creating album: '{title}'...")
    data = api_post(creds, "albums", {"album": {"title": title}})
    album_id = data["id"]
    print(f"Album created (id: {album_id})\n")
    return album_id


# ---------------------------------------------------------------------------
# 7. Add Photos to Album
#
#    batchAddMediaItems accepts up to 50 IDs at a time.
#    We chunk our flagged list into groups of 50 and send each chunk.
# ---------------------------------------------------------------------------
def add_to_album(creds: Credentials, album_id: str, media_ids: list[str]):
    if not media_ids:
        print("No photos to add to album.")
        return

    print(f"Adding {len(media_ids)} photos to album...")
    # Split into chunks of 50 (API limit per request)
    for i in range(0, len(media_ids), 50):
        chunk = media_ids[i:i + 50]
        api_post(creds, f"albums/{album_id}:batchAddMediaItems", {
            "mediaItemIds": chunk
        })
        print(f"  Added batch {i // 50 + 1} ({len(chunk)} photos)")

    print("Done adding photos.\n")


# ---------------------------------------------------------------------------
# 8. Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(max_photos: int = 100, threshold: float = 0.5):
    """
    Full pipeline:
      authenticate → fetch photos → run model → create album → populate album
    
    Args:
        max_photos: how many photos to scan from your library
        threshold:  P(delete) cutoff. Higher = more conservative (fewer flagged).
                    0.5 is neutral; try 0.7 if you want only high-confidence deletes.
    """
    print("=" * 60)
    print("  Google Photos AI Deletion Suggester")
    print("=" * 60 + "\n")

    # Step 1: Auth
    creds = authenticate()

    # Step 2: Fetch photos
    media_items = fetch_media_items(creds, max_items=max_photos)
    if not media_items:
        print("No photos found in your library.")
        return

    # Step 3: Load model
    model = load_model(MODEL_WEIGHTS)

    # Step 4: Run inference on each photo
    print(f"Running model on {len(media_items)} photos (threshold={threshold})...")
    flagged_ids   = []   # media item IDs to add to the album
    results       = []   # for summary report

    for i, item in enumerate(media_items):
        name = item.get("filename", f"photo_{i}")
        url  = item.get("baseUrl", "")

        prob = predict_from_url(model, url)
        decision = "DELETE" if prob >= threshold else "KEEP"

        if prob >= threshold:
            flagged_ids.append(item["id"])

        results.append({
            "filename": name,
            "prob_delete": round(prob, 4),
            "decision": decision,
        })

        # Progress update every 10 photos
        if (i + 1) % 10 == 0 or (i + 1) == len(media_items):
            n_flagged = sum(1 for r in results if r["decision"] == "DELETE")
            print(f"  [{i+1}/{len(media_items)}] — {n_flagged} flagged so far")

    # Step 5: Create album and populate it
    print()
    if flagged_ids:
        album_id = create_album(creds, "AI - Suggested Deletions")
        add_to_album(creds, album_id, flagged_ids)
    else:
        print("No photos were flagged for deletion at this threshold.\n")

    # Step 6: Print summary
    n_delete = len(flagged_ids)
    n_keep   = len(media_items) - n_delete
    print("=" * 60)
    print(f"  SUMMARY")
    print(f"  Photos scanned : {len(media_items)}")
    print(f"  Flagged DELETE : {n_delete} ({n_delete/len(media_items):.1%})")
    print(f"  Flagged KEEP   : {n_keep}   ({n_keep/len(media_items):.1%})")
    if flagged_ids:
        print(f"\n  → Open Google Photos and check the")
        print(f"    'AI - Suggested Deletions' album to review.")
        print(f"    Nothing has been permanently deleted.")
    print("=" * 60)

    # Optionally save full results to JSON for inspection
    out = PROJECT_ROOT / "pipeline" / "inference_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flag photos for deletion in Google Photos")
    parser.add_argument("--max",       type=int,   default=100,
                        help="Max number of photos to scan (default: 100)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Delete probability threshold (default: 0.5)")
    args = parser.parse_args()

    run_pipeline(max_photos=args.max, threshold=args.threshold)