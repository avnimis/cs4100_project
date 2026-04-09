import requests
from auth import authenticate
from google.auth.transport.requests import Request

def create_deletion_album(creds):
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json"
    }

    body = {
        "album": {
            "title": "Selected For Deletion"
        }
    }

    response = requests.post(
        "https://photoslibrary.googleapis.com/v1/albums",
        headers=headers,
        json=body
    )
    response.raise_for_status()
    album = response.json()
    print(f"Album created: {album['title']}")
    print(f"Album ID: {album['id']}")
    return album['id']

if __name__ == "__main__":
    creds = authenticate()
    create_deletion_album(creds)