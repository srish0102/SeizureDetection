import mne
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
file_path = "/mnt/d/Seizure_Project/raw_data/chb01/chb01_03.edf"

# 1. Load Data
print("Loading data...")
data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

# 2. Setup Time Window (10s before -> 10s after start)
# Seizure starts at 2996s
start_time = 2986 
duration = 20
data.crop(tmin=start_time, tmax=start_time + duration)

# 3. Pick 5 Key Channels (Left side of brain often shows it best)
channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3']
data.pick_channels(channels)

# 4. Get Data in MicroVolts (uV)
raw_data = data.get_data() * 1e6 
times = data.times + start_time # Adjust time axis to real seconds

# 5. Plot "Stacked" EEG Style
plt.figure(figsize=(15, 10))

# We loop through each channel and plot it higher up the screen
offset = 0
spacing = 300 # Distance between lines (uV)

for i, channel_name in enumerate(channels):
    # Get the single line
    signal = raw_data[i]
    
    # Plot it (signal + offset moves it up)
    plt.plot(times, signal + offset, color='black', linewidth=0.8)
    
    # Add Label on the left
    plt.text(start_time, offset + 50, channel_name, fontweight='bold', color='blue', fontsize=12)
    
    # Move the next line up
    offset += spacing

# 6. DRAW THE RED LINE (The Seizure Mark)
plt.axvline(x=2996, color='red', linestyle='--', linewidth=3, label="SEIZURE START (2996s)")

# 7. Make it pretty
plt.title("Patient 01: Seizure Onset Visualization", fontsize=18)
plt.xlabel("Time (seconds)", fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.yticks([]) # Hide messy numbers on Y-axis
plt.grid(True, axis='x', linestyle=':', alpha=0.5)

# 8. Save
plt.savefig("seizure_mark.png")
print("\n✅ Saved to: seizure_mark.png")
