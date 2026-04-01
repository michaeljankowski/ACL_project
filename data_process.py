
# import pandas as pd
# import numpy as np
# import os
# from scipy.fft import rfft, rfftfreq, irfft
# from sklearn.model_selection import train_test_split

# def fft_denoise(signal, threshold=50000):
#     fft_vals = rfft(signal)
#     freqs = rfftfreq(signal.size, d=1.0)
#     fft_vals[freqs > threshold] = 0
#     return irfft(fft_vals)

# def process_and_slice(file_path, label, window=500, stride=75):
#     try:
#         signal = np.loadtxt(file_path)
#         if signal.ndim > 1:
#             signal = signal[:, 0]
#         signal = fft_denoise(signal)
#         slices, labels = [], []
#         for i in range(0, len(signal) - window + 1, stride):
#             segment = signal[i:i + window]
#             if np.std(segment) == 0:
#                 continue
#             norm_segment = (segment - np.mean(segment)) / np.std(segment)
#             slices.append(norm_segment)
#             labels.append(label)
#         return slices, labels
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return [], []

# def preprocess_from_csv(csv_path, output_dir):
#     # Step 1: load filtered_df
#     filtered_df = pd.read_csv(csv_path)

#     train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42, stratify=filtered_df["numeric_label"])
#     train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df["numeric_label"])

# # 标记 split 列
#     filtered_df["split"] = "none"
#     filtered_df.loc[train_df.index, "split"] = "train"
#     filtered_df.loc[val_df.index, "split"] = "val"
#     filtered_df.loc[test_df.index, "split"] = "test"

#     # def assign_split(subject):
#     #     if subject in train_subjects:
#     #         return "train"
#     #     elif subject in val_subjects:
#     #         return "val"
#     #     elif subject in test_subjects:
#     #         return "test"
#     #     else:
#     #         return None

#     # filtered_df["split"] = filtered_df["subject"].apply(assign_split)

#     # Step 3: process and collect
#     all_data = {"train": [], "val": [], "test": []}
#     all_labels = {"train": [], "val": [], "test": []}

#     for _, row in filtered_df.iterrows():
#         split = row["split"]
#         label = int(row["numeric_label"])
#         slices, labels = process_and_slice(row["file_path"], label)
#         all_data[split].extend(slices)
#         all_labels[split].extend(labels)

#     os.makedirs(output_dir, exist_ok=True)
#     for split in ["train", "val", "test"]:
#         np.save(os.path.join(output_dir, f"X_{split}.npy"), np.array(all_data[split]))
#         np.save(os.path.join(output_dir, f"y_{split}.npy"), np.array(all_labels[split]))
#         print(f"✅ Saved {split} set: {len(all_data[split])} samples")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv_path", type=str, required=True, help="Path to Path_AMI_values.csv")
#     parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files")
#     args = parser.parse_args()
#     preprocess_from_csv(args.csv_path, args.output_dir)



import pandas as pd
import numpy as np
import os
from scipy.fft import rfft, rfftfreq, irfft
from sklearn.model_selection import train_test_split

def fft_denoise(signal, threshold=50000):
    fft_vals = rfft(signal)
    freqs = rfftfreq(signal.size, d=1.0)
    fft_vals[freqs > threshold] = 0
    return irfft(fft_vals)

# def process_and_slice(file_path, label, window=500, stride=75):
#     try:
#         signal = np.loadtxt(file_path)
#         if signal.ndim > 1:
#             signal = signal[:, 0]
#         signal = fft_denoise(signal)
#         slices, labels = [], []
#         for i in range(0, len(signal) - window + 1, stride):
#             segment = signal[i:i + window]
#             if np.std(segment) == 0:
#                 continue
#             norm_segment = (segment - np.mean(segment)) / np.std(segment)
#             slices.append(norm_segment)
#             labels.append(label)
#         return slices, labels
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return [], []

def process_and_slice_dual(vlo_path, vmo_path, label, window=500, stride=75):
    try:
        vlo_signal = np.loadtxt(vlo_path) if isinstance(vlo_path, str) and os.path.exists(vlo_path) else None
        vmo_signal = np.loadtxt(vmo_path) if isinstance(vmo_path, str) and os.path.exists(vmo_path) else None
        if vlo_signal is None or vmo_signal is None:
            return [], []

        # single channel
        if vlo_signal.ndim > 1:
            vlo_signal = vlo_signal[:, 0]
        if vmo_signal.ndim > 1:
            vmo_signal = vmo_signal[:, 0]

        # FFT 
        vlo_signal = fft_denoise(vlo_signal)
        vmo_signal = fft_denoise(vmo_signal)

        slices, labels = [], []
        for i in range(0, min(len(vlo_signal), len(vmo_signal)) - window + 1, stride):
            vlo_seg = vlo_signal[i:i + window]
            vmo_seg = vmo_signal[i:i + window]

            if np.std(vlo_seg) == 0 or np.std(vmo_seg) == 0:
                continue

            vlo_norm = (vlo_seg - np.mean(vlo_seg)) / np.std(vlo_seg)
            vmo_norm = (vmo_seg - np.mean(vmo_seg)) / np.std(vmo_seg)

            # two channel concatnate ：[window, 2]
            dual_segment = np.stack([vlo_norm, vmo_norm], axis=-1)
            slices.append(dual_segment)
            labels.append(label)

        return slices, labels
    except Exception as e:
        print(f"Error reading {vlo_path} or {vmo_path}: {e}")
        return [], []

def preprocess_from_csv(csv_path, output_dir):
    # Step 1: load df
    df = pd.read_csv(csv_path)

    # Split train/val/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["y_label"])
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df["y_label"])

    df["split"] = "none"
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    # Step 2: collect
    all_data = {"train": [], "val": [], "test": []}
    all_labels = {"train": [], "val": [], "test": []}

    for _, row in df.iterrows():
        split = row["split"]
        label = int(row["y_label"])

        # VLO
        # if isinstance(row.get("VLO_path"), str) and os.path.exists(row["VLO_path"]):
        #     slices, labels = process_and_slice(row["VLO_path"], label)
        #     all_data[split].extend(slices)
        #     all_labels[split].extend(labels)

        # # VMO
        # if isinstance(row.get("VMO_path"), str) and os.path.exists(row["VMO_path"]):
        #     slices, labels = process_and_slice(row["VMO_path"], label)
        #     all_data[split].extend(slices)
        #     all_labels[split].extend(labels)

        slices, labels = process_and_slice_dual(row["VLO_path"], row["VMO_path"], label)
        all_data[split].extend(slices)
        all_labels[split].extend(labels)

    # Step 3: save
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        np.save(os.path.join(output_dir, f"X_{split}.npy"), np.array(all_data[split]))
        np.save(os.path.join(output_dir, f"y_{split}.npy"), np.array(all_labels[split]))
        print(f"✅ Saved {split} set: {len(all_data[split])} samples")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to vlo_vmo_pairs_with_labels.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files")
    args = parser.parse_args()
    preprocess_from_csv(args.csv_path, args.output_dir)
