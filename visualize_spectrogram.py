import mne
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
file_path = "/mnt/d/Seizure_Project/raw_data/chb01/chb01_03.edf"
channel_to_analyze = 'P7-O1' # A good channel to see this specific seizure

# Check if file exists
if not os.path.exists(file_path):
    print(f"❌ Error: Cannot find {file_path}")
    exit()

# 1. Load Data
print("Loading data for heatmap...")
data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

# 2. Select only ONE channel
data.pick_channels([channel_to_analyze])

# 3. Crop to the seizure area (2990s to 3040s)
# Seizure starts at 2996s. We want a bit before and a bit after.
start_crop = 2990
stop_crop = 3040
data.crop(tmin=start_crop, tmax=stop_crop)

# 4. Get the raw numbers for plotting
# data.get_data() returns shape (n_channels, n_times), we need just the 1D array[0]
signals = data.get_data()[0]
sampling_rate = data.info['sfreq']

# 5. Create the Spectrogram (Heat Map)
fig = plt.figure(figsize=(12, 8))
plt.title(f"Seizure Energy Heatmap: Channel {channel_to_analyze}\n(Starts around t=6s in this window)")

# NFFT is the window size for fft. smaller = better time resolution, worse freq resolution
# Fs is sampling frequency
plt.specgram(signals, NFFT=512, Fs=sampling_rate, noverlap=256, cmap='inferno')

plt.ylabel('Frequency (Hz)')
plt.xlabel(f'Time (seconds relative to start time {start_crop}s)')
cbar = plt.colorbar()
cbar.set_label('Intensity (dB)')

# Zoom in on relevant frequencies (0-60Hz usually holds the action)
plt.ylim(0, 60)

# 6. Save
output_image = "spectrogram_heatmap.png"
plt.savefig(output_image)
print(f"\n✅ Heatmap saved to: {output_image}")
print("Go check your Desktop folder!")

