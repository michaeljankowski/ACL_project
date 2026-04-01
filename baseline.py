


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from utils import EMGDataset
# from utils import train_model, evaluate_model
# from torch.utils.data import Dataset, DataLoader
# from model import CNNLSTM, FCN, CNN, LSTMModel
# import argparse
# import torch


# # Function to read and slice a single .dat file into fixed-length windows
# # def slice_dat_file(filepath, window_size=150, stride=75):
# #     try:
# #         data = np.loadtxt(filepath)
# #         if data.ndim > 1:
# #             data = data[:, 0]  # Assume EMG signal is in first column if multidimensional
# #         num_windows = (len(data) - window_size) // stride + 1
# #         windows = [data[i * stride: i * stride + window_size] for i in range(num_windows)
# #                    if i * stride + window_size <= len(data)]
# #         return windows
# #     except Exception as e:
# #         print(f"Error reading {filepath}: {e}")
# #         return []
    
# # Standardize (z-score) each sample individually
# # def zscore_normalize(x):
# #     return (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x















# def get_model(model_name):
#     if model_name == "fcn":
#         return FCN()
#     elif model_name == "cnn":
#         return CNN()
#     elif model_name == "lstm":
#         return LSTMModel()
#     elif model_name == "cnnlstm":
#         return CNNLSTM()
#     else:
#         raise ValueError("Unknown model name")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="cnn", help="Choose from: fcn, cnn, lstm, cnnlstm")
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     args = parser.parse_args()

#     model = get_model(args.model)
#     print(f"Training model: {args.model.upper()}")


#     # Apply slicing and label each sample
#     # window_size = 150
#     # stride = 75
#     # sliced_data = {"train": [], "val": [], "test": []}
#     # sliced_labels = {"train": [], "val": [], "test": []}


#     # filtered_df = pd.read_csv("/home/zip22003/AMI-AI/dat files_emg data/Path_AMI_values.csv")
#     # # Train/val/test split by subject
#     # subjects = filtered_df["subject"].unique()
#     # train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=42)
#     # train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.25, random_state=42)

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

#     # for _, row in filtered_df.iterrows():
#     #     split = row["split"]
#     #     label = int(row["numeric_label"])
#     #     windows = slice_dat_file(row["file_path"], window_size=window_size, stride=stride)
#     #     if windows:
#     #         sliced_data[split].extend(windows)
#     #         sliced_labels[split].extend([label] * len(windows))


#     data_dir = "/home/zip22003/AMI-AI/processed_data"  # e.g. "./processed_emg"


#     X_train = np.load(f"{data_dir}/X_train.npy")
#     y_train = np.load(f"{data_dir}/y_train.npy")

# # 加载验证集
#     X_val = np.load(f"{data_dir}/X_val.npy")
#     y_val = np.load(f"{data_dir}/y_val.npy")

# # 加载测试集
#     X_test = np.load(f"{data_dir}/X_test.npy")
#     y_test = np.load(f"{data_dir}/y_test.npy")



#     # X_train = np.array([zscore_normalize(x) for x in X_train])
#     # X_val = np.array([zscore_normalize(x) for x in X_val])
#     # X_test = np.array([zscore_normalize(x) for x in X_test])

# # 


#     train_dataset = EMGDataset(X_train, y_train)
#     val_dataset = EMGDataset(X_val, y_val)
#     test_dataset = EMGDataset(X_test, y_test)

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64)
#     test_loader = DataLoader(test_dataset, batch_size=64)


#     train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=args.device)

#     print("Evaluating on test set...")
#     evaluate_model(model, test_loader, device=args.device)

# if __name__ == "__main__":
#     main()



import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from utils import EMGDataset, train_model, evaluate_model
from model import FCN, CNN1D, LSTMModel, CNN_LSTM


def get_model(model_name):
    if model_name == "fcn":
        return FCN()
    elif model_name == "cnn":
        return CNN1D()
    elif model_name == "lstm":
        return LSTMModel()
    elif model_name == "cnnlstm":
        return CNN_LSTM()
    else:
        raise ValueError("Unknown model name")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn", help="Choose from: fcn, cnn, lstm, cnnlstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="/home/zip22003/AMI-AI/processed_data")
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    args = parser.parse_args()

    
    model = get_model(args.model)
    print(f"Training model: {args.model.upper()} on {args.device}")

    
    X_train = np.load(f"{args.data_dir}/X_train.npy")
    y_train = np.load(f"{args.data_dir}/y_train.npy")
    X_val = np.load(f"{args.data_dir}/X_val.npy")
    y_val = np.load(f"{args.data_dir}/y_val.npy")
    X_test = np.load(f"{args.data_dir}/X_test.npy")
    y_test = np.load(f"{args.data_dir}/y_test.npy")

    
    train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EMGDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(EMGDataset(X_test, y_test), batch_size=args.batch_size)

    
    train_model(model, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, device=args.device,
                save_path=args.save_path)

    
    print(" Evaluating on test set...")
    evaluate_model(model, test_loader, device=args.device, model_path=args.save_path)


if __name__ == "__main__":
    main()