import mne
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
file_path = "/mnt/d/Seizure_Project/raw_data/chb01/chb01_03.edf"

# 1. Load Data
print("Loading data...")
data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

# 2. Zoom in on the Seizure
# Seizure starts at 2996. Let's look at 2990 to 3010 (20 seconds)
start_time = 2990
duration = 20
data.crop(tmin=start_time, tmax=start_time + duration)

# 3. Filter the Data (Optional but makes it look cleaner)
# We remove slow drift (High-pass 1Hz) and high-frequency noise (Low-pass 40Hz)
data.filter(l_freq=1.0, h_freq=40.0)

# 4. Create the Stack Plot
# MNE has a built-in function 'plot()' that does exactly this "stacking" style.
# We set 'scalings' to 'auto' so it fits nicely.
fig = data.plot(
    duration=duration, 
    n_channels=23, 
    scalings='auto', 
    show=False, 
    title="Full Brain Montage (Seizure Start at 2996s)"
)

# 5. Save
output_file = "stack_plot.png"
fig.savefig(output_file)
print(f"\n✅ Stack plot saved to: {output_file}")
print("Check your Desktop folder. You will see all 23 channels acting up!")
