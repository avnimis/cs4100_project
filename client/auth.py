from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle
import os

# Scopes define what data you can access
'''
Gmail: 
    1. View your email messages and settings

Photos: 
    1. Append images into your google photos
    2. Sort photos into folders 
'''

SCOPES =    [
            'https://www.googleapis.com/auth/gmail.modify'
            ]


def authenticate():
    creds = None
    
    print("Starting authentication...")
    
    # Check if we have saved credentials
    if os.path.exists('token.pickle'):
        print("Loading saved credentials from token.pickle")
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
        print("Scopes:", creds.scopes)
        print("Valid:", creds.valid)
        print("Expired:", creds.expired)
    else:
        print("No saved credentials found")
    
    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials")
            creds.refresh(Request())
        else:
            print("Initiating OAuth 2.0 flow...")
            flow = InstalledAppFlow.from_client_secrets_file(
                'auth/credentials.json',
                SCOPES
            )
            creds = flow.run_local_server(port=0)
            print("User authorized successfully")
        
        # Save credentials for next time
        print("Saving credentials to token.pickle")
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    print("Authentication complete")
    return creds

if __name__ == "__main__":
    creds = authenticate()
    print(f"Ready to use API with scopes: {creds.scopes}")