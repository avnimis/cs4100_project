from googleapiclient.discovery import build
import pickle

def load_credentials():
    with open('token.pickle', 'rb') as token:
        return pickle.load(token)

def test_gmail_api(creds):
    print("\n--- Testing Gmail API ---")
    try:
        service = build('gmail', 'v1', credentials=creds)
        
        # Get user profile
        profile = service.users().getProfile(userId='me').execute()
        print(f"Email: {profile.get('emailAddress')}")
        print(f"Messages: {profile.get('messagesTotal')}")
        print(f"Unread: {profile.get('messagesUnread')}")
        
        # Get recent messages
        results = service.users().messages().list(userId='me', maxResults=5).execute()
        messages = results.get('messages', [])
        
        if messages:
            print(f"\nRecent messages:")
            for msg in messages[:3]:
                message = service.users().messages().get(userId='me', id=msg['id']).execute()
                headers = message['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No subject')
                print(f"  - {subject}")
        
        print("Gmail API: Success")
    except Exception as e:
        print(f"Gmail API: Failed - {e}")

if __name__ == "__main__":
    creds = load_credentials()
    
    test_gmail_api(creds)