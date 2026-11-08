import mne
import numpy as np
import os

# --- CONFIGURATION ---
# Points dynamically to the folder created by your pure Python downloader
DATA_DIR = os.path.join("raw_data", "chb01")
SAVE_DIR = "processed_data"
WINDOW_SIZE = 4  # 4 seconds per segment
STEP_SIZE = 4    # Set equal to WINDOW_SIZE -> ZERO overlap (NO Data Leakage!)

# Create save folder
os.makedirs(SAVE_DIR, exist_ok=True)

# Standard clinical 18-channel "Double Banana" montage (MNE Auto-suffix version)
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2',  # Includes the MNE duplicate auto-suffix
    'FZ-CZ', 'CZ-PZ'
]

# Alternative clinical montage (fallback if MNE doesn't apply suffix)
ALTERNATIVE_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ'
]

# Files we downloaded for Patient 1 containing marked seizures
SEIZURE_FILES = {
    "chb01_03.edf": [2996, 3036],
    "chb01_04.edf": [1467, 1494],
    "chb01_15.edf": [1732, 1772],
    "chb01_16.edf": [1015, 1066],
    "chb01_18.edf": [1720, 1810],
    "chb01_21.edf": [327, 420],
    "chb01_26.edf": [1862, 1963]
}

all_X = []
all_y = []

print(f"🚀 Starting mass preprocessing of {len(SEIZURE_FILES)} Patient 01 files...")

for filename, (seizure_start, seizure_end) in SEIZURE_FILES.items():
    file_path = os.path.join(DATA_DIR, filename)
    print(f"\n📂 Processing {filename}...")
    
    if not os.path.exists(file_path):
        print(f"❌ Skipped {filename} (File not found on disk). Run your download_patient.py script first.")
        continue
        
    try:
        # Load raw EDF data
        data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"❌ Skipped {filename} (Failed to load/corrupt): {e}")
        continue

    # 1. Montage Standardization (Enforce standard 18 channels)
    available_channels = data.ch_names
    selected_montage = None
    
    if all(ch in available_channels for ch in STANDARD_CHANNELS):
        selected_montage = STANDARD_CHANNELS
    elif all(ch in available_channels for ch in ALTERNATIVE_CHANNELS):
        selected_montage = ALTERNATIVE_CHANNELS
    
    if selected_montage is not None:
        data.pick(selected_montage)  # Uses clean, modern MNE pick function
    else:
        missing = [ch for ch in STANDARD_CHANNELS if ch not in available_channels]
        print(f"⚠️ Channel mismatch in {filename}. Missing: {missing}. Skipping file.")
        continue

    # 2. Bandpass Filtering (0.5 - 40.0 Hz)
    data.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    
    # Extract raw data array & scale to microvolts (uV)
    raw_data = data.get_data() * 1e6
    fs = int(data.info['sfreq'])
    total_samples = raw_data.shape[1]
    window_samples = WINDOW_SIZE * fs
    step_samples = STEP_SIZE * fs

    # Sliding-window segmentation loop
    for start_idx in range(0, total_samples - window_samples, step_samples):
        end_idx = start_idx + window_samples
        current_time = start_idx / fs
        mid_point = current_time + (WINDOW_SIZE / 2)
        
        # Label window as seizure (1) if it falls within the clinical window
        label = 1 if (seizure_start <= mid_point <= seizure_end) else 0
        chunk = raw_data[:, start_idx:end_idx]

        # 3. Z-Score Amplitude Normalization (Channel-Wise)
        chunk_mean = np.mean(chunk, axis=1, keepdims=True)
        chunk_std = np.std(chunk, axis=1, keepdims=True) + 1e-8
        chunk_normalized = (chunk - chunk_mean) / chunk_std

        all_X.append(chunk_normalized)
        all_y.append(label)

# Package and save arrays
X_final = np.array(all_X)
y_final = np.array(all_y)

print("\n📦 Packaging Dataset...")
if len(X_final) > 0:
    print(f"Final Data Shape: {X_final.shape} (Windows, 18 Channels, {WINDOW_SIZE * 256} TimePoints)")
    print(f"Seizures (Class 1): {np.sum(y_final == 1)} | Normal (Class 0): {np.sum(y_final == 0)}")
    
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_final)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_final)
    print("✅ Preprocessing Complete! Datasets successfully saved locally.")
else:
    print("❌ Critical Error: No windows extracted. Check that your raw EDF files are completely loaded.")
