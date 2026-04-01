# EMG Data Analysis for ACLR Classification

This repository contains a machine learning pipeline for analyzing electromyography (EMG) data to classify anterior cruciate ligament reconstruction (ACLR) patients based on muscle activation patterns. The project uses dual-channel EMG signals from vastus lateralis oblique (VLO) and vastus medialis oblique (VMO) muscles.


## Dataset Structure

The dataset is organized into three main categories:
- **ACLR**: Patients with anterior cruciate ligament reconstruction
  - `Contra_Limb/`: Contralateral (uninjured) limb data
  - `Injured_Limb/`: Injured limb data
- **Healthy**: Control group data
  - `Contra_Limb/`: Control limb data
  - `Injured_Matched_Limb/`: Matched limb data

Each category contains EMG recordings from two muscle groups:
- **VLO**: Vastus Lateralis Oblique
- **VMO**: Vastus Medialis Oblique

## Features

### Data Processing
- **Signal Preprocessing**: FFT-based denoising with configurable frequency thresholds
- **Windowing**: Sliding window approach for time-series segmentation (default: 500 samples, 75 stride)
- **Normalization**: Z-score normalization for each signal segment
- **Dual-channel Processing**: Simultaneous processing of VLO and VMO signals

### Machine Learning Models
Four different neural network architectures are implemented:

1. **FCN (Fully Connected Network)**: Simple feedforward network
2. **CNN1D**: 1D Convolutional Neural Network for temporal pattern recognition
3. **LSTM**: Long Short-Term Memory network for sequence modeling
4. **CNN-LSTM**: Hybrid architecture combining convolutional and recurrent layers

### Training Features
- **Cross-validation**: Subject-based train/validation/test splits
- **Early stopping**: Model checkpointing based on validation accuracy
- **Comprehensive metrics**: Accuracy, precision, recall, and F1-score evaluation
- **GPU support**: CUDA acceleration when available


### Requirements
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install scipy matplotlib
pip install jupyter
```


### 1. Data Preprocessing

First, preprocess the raw EMG data:

```bash
python data_process.py --csv_path vlo_vmo_pairs_with_labels.csv --output_dir processed_data
```

This will:
- Apply FFT denoising to remove high-frequency noise
- Create sliding windows from continuous signals
- Normalize each segment using z-score
- Split data into train/validation/test sets
- Save processed arrays as `.npy` files

### 2. Model Training

Train a specific model:

```bash
python baseline.py --model cnn --epochs 50 --lr 0.001 --batch_size 64
```

Available models:
- `fcn`: Fully Connected Network
- `cnn`: 1D Convolutional Neural Network
- `lstm`: LSTM Network
- `cnnlstm`: CNN-LSTM Hybrid

### 3. Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | `cnn` |
| `--epochs` | Training epochs | `20` |
| `--lr` | Learning rate | `1e-3` |
| `--batch_size` | Batch size | `64` |
| `--device` | Device (cuda/cpu) | Auto-detect |
| `--data_dir` | Processed data directory | `processed_data` |
| `--save_path` | Model checkpoint path | `best_model.pt` |





