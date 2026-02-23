from googleapiclient.discovery import build
import pickle

def load_credentials():
    with open('token.pickle', 'rb') as token:
        return pickle.load(token)


def test_google_photos_api(creds):
    print("\n--- Testing Google Photos API ---")
    try:
        print('Attempting service build')

        service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)
        print('Service built')

        # Get albums
        results = service.albums().list(pageSize=10).execute()
        print('Albums retrieved')

        albums = results.get('albums', [])
        
        if albums:
            print(f"Albums found: {len(albums)}")
            for album in albums[:5]:
                print(f"  - {album['title']} ({album.get('mediaItemsCount', 0)} items)")
        else:
            print("No albums found")
        
        print("Google Photos API: Success")
    except Exception as e:
        print(f"Google Photos API: Failed - {e}")

if __name__ == "__main__":
    creds = load_credentials()
    test_google_photos_api(creds)