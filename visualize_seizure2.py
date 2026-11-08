import mne
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
file_path = "/mnt/d/Seizure_Project/raw_data/chb01/chb01_03.edf"

# 1. Load Data
print("Loading data...")
data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

# 2. Zoom in on the Seizure (Buffer of 60 seconds)
# Seizure is 2996s to 3036s. We look at 2950s to 3050s to see the "Before" vs "During".
start_time = 2950
stop_time = 3050
data.crop(tmin=start_time, tmax=stop_time)

# 3. Calculate "Global Energy" (GFP - Global Field Power)
# This mathematical trick combines all 23 channels into ONE "Intensity" score.
# We multiply by 1,000,000 to convert from Volts to MicroVolts (easier numbers)
signals = data.get_data() * 1e6 
energy = np.std(signals, axis=0)  # Standard Deviation across all channels
times = data.times  # Time axis

# 4. Plot
plt.figure(figsize=(12, 6))

# Plot the Energy Line
plt.plot(times + start_time, energy, color='black', linewidth=1.5, label='Brain Energy (GFP)')

# Highlight the Seizure Zone (Red Box)
plt.axvspan(2996, 3036, color='red', alpha=0.3, label='Seizure Active (2996s - 3036s)')

# Labels
plt.title("Brain Energy Levels: Calm vs. Seizure", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Energy Amplitude (µV)", fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# 5. Save
plt.savefig("energy_plot.png")
print("\n✅ Simple Graph saved to: energy_plot.png")
