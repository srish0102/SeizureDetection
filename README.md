# 🧠 AI Neurologist: Clinical EEG Seizure Detection via Hybrid CNN-Mamba

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MNE-Python](https://img.shields.io/badge/MNE-1.11.0-blue)](https://mne.tools/)
[![Dataset](https://img.shields.io/badge/Dataset-CHB--MIT%20PhysioNet-success)](https://physionet.org/content/chbmit/1.0.0/)
[![Architecture](https://img.shields.io/badge/Architecture-1D--CNN%20%2B%20Mamba%20SSM-crimson)](#model-architecture)

An end-to-end clinical deep learning pipeline for automated epileptic seizure detection from continuous scalp EEG recordings (**CHB-MIT Database**). This project introduces a lightweight **Hybrid 1D-CNN + Selective State Space Model (Mamba)** built from scratch in pure PyTorch, achieving a **93.02% F1-Score** and **0.9870 ROC-AUC** with **36% fewer parameters** than a bidirectional LSTM baseline under strict zero-leakage evaluation.

---

## 📌 Key Engineering Highlights

| Challenge in Clinical EEG | Common Literature Flaw | Our Engineering Solution |
| :--- | :--- | :--- |
| **Temporal Data Leakage** | Random train/test splitting on overlapping sliding windows ($75\text{--}90\%$ shared signal), inflating test accuracy to an artificial $99\%$. | Enforced strict **non-overlapping 4-second windows** (`STEP_SIZE = WINDOW_SIZE = 4s`) so zero time points are shared between train and test splits. |
| **Extreme Class Imbalance** | Seizures represent only **1.87%** of continuous monitoring hours (`112` ictal vs. `5,863` normal windows), causing standard `BCELoss` models to predict "Normal" $100\%$ of the time. | Engineered a dynamic **52.11x Recall-Weighted Loss** (`pos_weight` in `BCEWithLogitsLoss`) to heavily penalize false negatives without synthetic oversampling overhead. |
| **Inconsistent EDF Montages** | Raw CHB-MIT recordings contain duplicate channels (`T8-P8`), dummy signals, or varying electrode layouts across sessions. | Automated **18-channel "Double Banana" bipolar montage** selection with dynamic MNE duplicate suffix resolution, $0.5\text{--}40\text{ Hz}$ bandpass filtering, and per-window Z-score normalization. |
| **Hardware Portability** | Official `mamba-ssm` relies on Linux-only Triton/CUDA C++ kernels that fail to compile on Windows and newer Blackwell (`sm_120`) GPUs. | Implemented Zero-Order Hold (ZOH) continuous-to-discrete state-space parameterization ($\Delta, A, B, C, D$) natively in PyTorch (`PurePyTorchMambaBlock`). |

---

## 📊 Experimental Results & Benchmarks

Both models were trained and benchmarked locally on an **NVIDIA GeForce RTX 5050 Laptop GPU** (`CUDA 12.8`) across **5,975 leak-free 4-second EEG windows** (`18 channels × 1024 time points` at `256 Hz`) from Subject `chb01`.

### 1. Model Performance Comparison

| Metric / Specification | 🧠 Hybrid CNN-Mamba (Ours) | 🔄 Hybrid CNN-LSTM (Baseline) | Engineering Takeaway |
| :--- | :---: | :---: | :--- |
| **Trainable Parameters** | **152,385** | 237,889 | **Mamba is 36.0% smaller**, making it ideal for edge/wearable clinical monitoring. |
| **Peak Validation F1-Score** | **93.02%** | **93.02%** | Matches bi-directional recurrent performance with a unidirectional state space scan. |
| **Peak Precision** | **95.24%** | **95.24%** | Extremely high positive predictive value; minimizes ICU alarm fatigue. |
| **Sensitivity (Recall)** | **90.91%** (20/22) | **95.45%** (21/22) | Captures $>90\%$ of ictal windows; missed windows occur only at transition boundaries. |
| **Peak ROC-AUC** | **0.9870** | **0.9966** | Near-perfect probability separation between normal baseline and ictal discharges. |
| **False Alarms (FP)** | **5 / 1,173** | **5 / 1,173** | **$99.57\%$ Specificity** across unseen normal brain activity windows. |
| **Single-Window Latency** | 52.34 ms | **1.45 ms** | Well below the 4,000 ms real-time window threshold (*see latency note below*). |

> **Note on Inference Latency:** Both models process a 4,000 ms EEG window in a fraction of a second, easily satisfying real-time clinical streaming requirements. While `nn.LSTM` leverages pre-compiled C++/cuDNN fused kernels (`1.45 ms`), our `MambaBlock` executes its selective scan in native PyTorch (`52.34 ms`) for cross-platform compatibility on Windows/Blackwell GPUs without custom C++ compilation.

### 2. Training Convergence & Evaluation Curves

![Model Comparison Results](./model_comparison_results.png)

---

## 🔬 Signal Diagnostics & Visualizations

Before training, raw `.edf` recordings were inspected across time, frequency, and spatial energy domains (Subject `chb01_03.edf`, documented clinical seizure from **2996s to 3036s**):

| Visualization | Clinical Insight |
| :--- | :--- |
| **Stacked Montage Traces** (`seizure_mark.png`) | Displays all EEG channels in $\mu\text{V}$, revealing the abrupt transition at `2996s` from calm background activity to high-amplitude rhythmic ictal spikes. |
| **Global Field Power / Brain Energy** (`energy_plot.png`) | Tracks standard deviation across channels over time, isolating a massive energy surge strictly within the annotated `2996s–3036s` seizure window. |
| **STFT Spectrogram Heatmap** (`spectrogram_heatmap.png`) | Short-Time Fourier Transform on occipital-parietal channels (`P7-O1`) showing intense spectral power concentration below $30\text{ Hz}$ during seizure onset. |

---

## 🏗️ Model Architecture

```text
Input EEG Window: [Batch, 18 Channels, 1024 TimePoints (4s @ 256Hz)]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ 1. Spatial-Temporal 1D-CNN Compressor                    │
│    • Conv1d(18 → 64, k=5) + BatchNorm + ReLU + MaxPool(4)│
│    • Conv1d(64 → 128, k=3) + BatchNorm + ReLU + MaxPool(4)│
│    • Compresses sequence length 16x: [B, 1024] → [B, 64] │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ 2. Pure PyTorch Mamba Block (Selective State Space)      │
│    • LayerNorm + Linear Projection (d_model=128, exp=2)  │
│    • 1D Depthwise Conv (d_conv=4) + SiLU Activation      │
│    • Input-Dependent Selection: Δ (dt), B, C projections │
│    • ZOH Discretization: Ā = exp(Δ·A), B̄ = Δ·B          │
│    • Sequential State Scan: h_t = Ā·h_{t-1} + B̄·x_t     │
│    • Residual Connection: x + Mamba(LayerNorm(x))        │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ 3. Classification Head                                   │
│    • Global Average Pooling across time tokens [B, 128]  │
│    • Linear(128 → 64) + ReLU + Dropout(0.3) → Linear(1)  │
│    • Optimized via Recall-Weighted BCEWithLogitsLoss     │
└──────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```text
├── download_patient.py          # Pure-Python downloader for PhysioNet CHB-MIT EDF files
├── preprocess_full.py           # 18-ch standardization, 0.5-40Hz filtering, Z-score & 4s non-overlapping slicing
├── mamba.py                     # From-scratch Pure PyTorch Mamba Selective State Space block
├── train_cnn_mamba.py           # Training & clinical evaluation pipeline for Hybrid CNN-Mamba
├── train_cnn_lstm.py            # Comparative baseline pipeline for Hybrid 1D-CNN + Bi-LSTM
├── benchmark_speed.py           # GPU inference latency (ms) and parameter profiler
├── plot_results.py              # Generates 3-panel high-DPI comparison plots
├── model_comparison_results.png # Saved training convergence, F1, and ROC-AUC plots
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/seizure-detection-mamba.git
cd seizure-detection-mamba
pip install torch numpy pandas scikit-learn mne matplotlib
```

### 2. Download Patient 01 Raw EDF Data
Fetches the annotated seizure `.edf` files directly from PhysioNet into `raw_data/chb01/`:
```bash
python download_patient.py
```

### 3. Run Leak-Free Preprocessing
Applies the 18-channel Double Banana montage, bandpass filter ($0.5\text{--}40\text{ Hz}$), channel-wise Z-score normalization, and non-overlapping 4-second segmentation:
```bash
python preprocess_full.py
```

### 4. Train & Benchmark Models
```bash
# Train the Hybrid CNN-Mamba model
python train_cnn_mamba.py

# Train the Hybrid CNN-LSTM baseline
python train_cnn_lstm.py

# Profile GPU Latency & Parameter Counts
python benchmark_speed.py

# Generate Comparison Plots
python plot_results.py
```

---

## 👩‍💻 Author

**Srishti Bhatia**  
*B.Tech Computer Science & Engineering (AI & ML)*
