import mne
import numpy as np

# --- CONFIGURATION ---
file_path = "/mnt/d/Seizure_Project/raw_data/chb01/chb01_03.edf"

print(f"🔍 Inspecting: {file_path}")

try:
    # 1. Load Data
    data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # 2. Grab the first channel (usually FP1-F7)
    channel_name = data.ch_names[0]
    print(f"📡 Channel: {channel_name}")
    
    # 3. Get the Raw Signal (in Volts)
    raw_signal = data.get_data()[0] 
    
    # 4. DIAGNOSTICS
    print(f"\n--- STATISTICS ---")
    print(f"Min Value: {np.min(raw_signal)}")
    print(f"Max Value: {np.max(raw_signal)}")
    print(f"Mean Value: {np.mean(raw_signal)}")
    
    # 5. Check for Zeros
    if np.max(np.abs(raw_signal)) == 0:
        print("\n❌ CRITICAL ERROR: The signal is completely FLAT (All Zeros).")
        print("The file might be corrupted or empty.")
    else:
        print("\n✅ Data Detected! The signal is NOT empty.")
        print("If the plot is blank, it is a scaling/zoom problem.")
        
        # Print first 5 actual numbers
        print(f"First 5 samples: {raw_signal[:5]}")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: Could not read file.")
    print(e)
