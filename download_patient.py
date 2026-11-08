import os
import urllib.request
from urllib.error import URLError

# --- CONFIGURATION ---
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/chb01/"
SAVE_DIR = "raw_data/chb01"
os.makedirs(SAVE_DIR, exist_ok=True)

# List of all files we need for Patient 01
FILES_TO_DOWNLOAD = [
    "chb01-summary.txt",
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_15.edf",
    "chb01_16.edf",
    "chb01_18.edf",
    "chb01_21.edf",
    "chb01_26.edf"
]

def download_file_pure_python(filename):
    url = f"{BASE_URL}{filename}"
    save_path = os.path.join(SAVE_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"✅ {filename} already exists. Skipping.")
        return
        
    print(f"⬇️ Downloading {filename}...")
    try:
        # Simple progress reporter callback
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, int(downloaded * 100 / total_size))
                print(f"\rProgress: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")
            else:
                print(f"\rDownloaded: {downloaded // 1024} KB", end="")

        urllib.request.urlretrieve(url, save_path, reporthook=progress_hook)
        print(f"\n✅ Successfully saved to {save_path}")
        
    except URLError as e:
        print(f"\n❌ Error downloading {filename}: {e.reason}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    print(f"🚀 Starting Patient 01 data download to: {os.path.abspath(SAVE_DIR)}")
    for file in FILES_TO_DOWNLOAD:
        download_file_pure_python(file)
    print("\n🏁 All downloads completed!")
