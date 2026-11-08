import os
import urllib.request

# --- CONFIGURATION ---
PATIENT_ID = "chb01"
BASE_URL = f"https://physionet.org/files/chbmit/1.0.0/{PATIENT_ID}/"
SUMMARY_FILE = f"{PATIENT_ID}-summary.txt"

# Paths
WAREHOUSE_PATH = f"/mnt/d/Seizure_Project/raw_data/{PATIENT_ID}/"
FACTORY_PATH = "./"  # Current folder

# Ensure Warehouse exists
os.makedirs(WAREHOUSE_PATH, exist_ok=True)

def get_summary():
    url = BASE_URL + SUMMARY_FILE
    save_path = os.path.join(WAREHOUSE_PATH, SUMMARY_FILE)
    
    print(f"🕵️  Fetching Summary for {PATIENT_ID}...")
    
    if not os.path.exists(save_path):
        print(f"⬇️  Downloading {SUMMARY_FILE}...")
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return None
    else:
        print("✅ Found existing summary file.")

    return save_path

def parse_summary(file_path):
    print(f"\n READING REPORT: {file_path}")
    print("-" * 40)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    current_file = ""
    seizure_count = 0
    
    for line in lines:
        line = line.strip()
        
        # Keep track of which .edf file we are reading about
        if line.startswith("File Name:"):
            current_file = line.split(": ")[1]
            
        # Detect Seizure Info
        if "Number of Seizures in File: 1" in line:
            print(f"💡 SEIZURE DETECTED in: {current_file}")
            seizure_count += 1
            
        if "Seizure Start Time" in line:
            print(f"  🟢 Starts at: {line.split(': ')[1]} seconds")
            
        if "Seizure End Time" in line:
            print(f"  🔴 Ends at:   {line.split(': ')[1]} seconds")
            print("")

    if seizure_count == 0:
        print("This patient is clean (No seizures found in scanned files).")
    else:
        print(f"⚠️  Found {seizure_count} files with seizures.")

if __name__ == "__main__":
    path = get_summary()
    if path:
        parse_summary(path)

