from googleapiclient.discovery import build
import pickle

def get_drive_files():
    # Load credentials from token.pickle
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
    
    # Create a Drive API service
    service = build('drive', 'v3', credentials=creds)
    
    # Make an API call
    results = service.files().list(pageSize=10).execute()
    files = results.get('files', [])
    
    if files:
        print("Files found:")
        for file in files:
            print(f"  {file['name']}")
    else:
        print("No files found")

if __name__ == "__main__":
    get_drive_files()