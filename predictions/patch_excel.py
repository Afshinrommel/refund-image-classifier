import os
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CSV_PATH = "predictions/predictions.csv"
DRIVE_FOLDER_ID = "1VUYS-Ldx--bbSbQDoJ5XRj-nFV-aKKnc"
CLIENT_SECRET_FILE = "client_secret.json"

# Scope required to view file metadata
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def update_existing_excel():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found!")
        return

    if not os.path.exists(CLIENT_SECRET_FILE):
        print(f"❌ Error: '{CLIENT_SECRET_FILE}' missing in this directory!")
        return

    print("Initiating local authentication flow...")
    try:
        # Create the flow using the local client_secret.json file
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_FILE, 
            scopes=SCOPES,
            redirect_uri="http://localhost:5000/oauth2callback"
        )
        
        # Force the local server to run on the exact port Google expects
        creds = flow.run_local_server(host="localhost", port=5000, open_browser=True)
        
        print("Successfully authenticated. Connecting to Google Drive...")
        service = build("drive", "v3", credentials=creds) 
        
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(name, createdTime)", pageSize=1000).execute()
        files = results.get("files", [])
        
        drive_timestamps = {f["name"]: f["createdTime"] for f in files}
        
        df = pd.read_csv(CSV_PATH)
        df["gdrive_upload_time"] = df["filename"].map(drive_timestamps)
        
        df.to_csv(CSV_PATH, index=False)
        print("✅ Excel file updated successfully with Google Drive upload timestamps!")
        
    except Exception as e:
        print(f"❌ Connection or authentication error: {e}")

if __name__ == "__main__":
    if os.path.exists("predictions.csv"):
        CSV_PATH = "predictions.csv"
    update_existing_excel()