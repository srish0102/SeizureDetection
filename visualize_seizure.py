import mne
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# The file with the known seizure
file_path = "/mnt/d/Seizure_Project/raw_data/chb01/chb01_03.edf"

# Check if file exists first
if not os.path.exists(file_path):
    print(f"❌ Error: Cannot find file at {file_path}")
    print("Did you mount the D: drive? (sudo mount -t drvfs D: /mnt/d)")
    exit()

# 1. Load the Data
print(f"Loading {file_path}...")
data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

# 2. Get Basic Info
print(f"Channels: {data.ch_names[:5]} ... (and more)")
print(f"Duration: {data.times[-1]} seconds")
print(f"Sampling Rate: {data.info['sfreq']} Hz")

# 3. ZOOM IN: The Seizure (Starts at 2996s)
# We plot 10 seconds BEFORE (Normal) and 10 seconds DURING (Seizure)
start_time = 2990 
duration = 20 

# Select 4 specific channels to keep the plot clean
channels_to_plot = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1'] 
data.pick_channels(channels_to_plot)

# 4. Plot the Brainwaves
print("Generating plot...")
fig = data.plot(start=start_time, duration=duration, 
                n_channels=4, show=False, title="Seizure Onset (File 03)")

# 5. Save the image
output_image = "seizure_plot.png"
plt.savefig(output_image)
print(f"\n✅ Plot saved to: {output_image}")
print("Go to your Desktop/Seizure_Project_Fast folder to open it!")
