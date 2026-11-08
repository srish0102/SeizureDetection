import mne
import numpy as np
import os

# --- CONFIGURATION ---
file_path = "/mnt/d/Seizure_Project/raw_data/chb01/chb01_03.edf"
WINDOW_SIZE = 4  # Seconds (How big is each "bite" for the AI?)
STEP_SIZE = 4    # Seconds (No overlap for now)

# 1. Load Data
print(f"Loading {file_path}...")
data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

# 2. FILTERING (Cleaning the Dirt)
# We keep only 0.5Hz - 40Hz. This removes power line noise (60Hz) and drift.
print("Applying Bandpass Filter (0.5 - 40 Hz)...")
data.filter(l_freq=0.5, h_freq=40.0)

# 3. SEGMENTATION (Chopping into Windows)
# The summary says Seizure is: 2996s to 3036s
seizure_start = 2996
seizure_end = 3036

# Calculate total number of windows
duration = data.times[-1]
n_windows = int(duration / STEP_SIZE)

print(f"Chopping data into {n_windows} windows of {WINDOW_SIZE} seconds each...")

# We will store our clean data here
X_data = [] # The Brainwaves
y_labels = [] # The Answer Key (0 or 1)

raw_data = data.get_data() * 1e6 # Convert to uV
fs = int(data.info['sfreq'])     # 256 Hz
samples_per_window = WINDOW_SIZE * fs

for i in range(n_windows):
    # Calculate start/end time for this window
    start_sec = i * STEP_SIZE
    end_sec = start_sec + WINDOW_SIZE
    
    # Check if this window is inside the Seizure Zone
    # If the window touches the seizure, we call it a Seizure Window
    if (start_sec >= seizure_start) and (end_sec <= seizure_end):
        label = 1 # Seizure!
    else:
        label = 0 # Normal
    
    # Extract the chunk of data (slicing the array)
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)
    
    window_data = raw_data[:, start_sample:end_sample]
    
    # Safety Check: Ensure window is the exact right shape
    if window_data.shape[1] == samples_per_window:
        X_data.append(window_data)
        y_labels.append(label)

# Convert to Numpy Arrays (The Format AI loves)
X_data = np.array(X_data)
y_labels = np.array(y_labels)

# 4. RESULTS
print("\n--- PREPROCESSING COMPLETE ---")
print(f"Total Windows Created: {len(y_labels)}")
print(f"Seizure Windows (Class 1): {np.sum(y_labels == 1)}")
print(f"Normal Windows (Class 0): {np.sum(y_labels == 0)}")
print(f"Data Shape: {X_data.shape}")
print("(Windows, Channels, TimePoints)")

# 5. Save (Optional Demo)
# np.save("X_demo.npy", X_data)
# np.save("y_demo.npy", y_labels)
