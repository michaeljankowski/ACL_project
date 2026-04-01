# import torch
# from torch.utils.data import Dataset, DataLoader



# class EMGDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N, 1, 150]
#         self.y = torch.tensor(y, dtype=torch.long)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
    



# import torch
# import torch.nn as nn
# from sklearn.metrics import accuracy_score

# def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda', save_path='best_model.pth'):
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     best_val_acc = 0.0

#     for epoch in range(epochs):
#         model.train()
#         train_losses, train_preds, train_labels = [], [], []

#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             output = model(X)
#             loss = criterion(output, y)
#             loss.backward()
#             optimizer.step()

#             train_losses.append(loss.item())
#             train_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
#             train_labels.extend(y.cpu().numpy())

#         acc = accuracy_score(train_labels, train_preds)
#         print(f"Epoch {epoch+1}: Train Loss={sum(train_losses)/len(train_losses):.4f} Acc={acc:.4f}")

#         # Optional: evaluate on validation set
#         model.eval()
#         val_preds, val_labels = [], []
#         with torch.no_grad():
#             for X, y in val_loader:
#                 X, y = X.to(device), y.to(device)
#                 output = model(X)
#                 val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
#                 val_labels.extend(y.cpu().numpy())
#         val_acc = accuracy_score(val_labels, val_preds)
#         print(f"          Val Acc={val_acc:.4f}")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), save_path)
#             print(f"          🔥 New best model saved to {save_path}")



# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# def evaluate_model(model, test_loader, device='cuda', load_path='best_model.pth'):

#     model.load_state_dict(torch.load(load_path))
#     model.eval()
#     model.to(device)
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for X, y in test_loader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())

#     acc = accuracy_score(all_labels, all_preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

#     print(f"Test Accuracy:  {acc:.4f}")
#     print(f"Test Precision: {precision:.4f}")
#     print(f"Test Recall:    {recall:.4f}")
#     print(f"Test F1 Score:  {f1:.4f}")
#     return acc, precision, recall, f1




import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EMGDataset(Dataset):
    def __init__(self, X, y):
        # X: [N, window, 2], y: [N]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_datasets(data_dir, batch_size=64):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    train_loader = DataLoader(EMGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EMGDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(EMGDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader



import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu", save_path="best_model.pt"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss={np.mean(train_losses):.4f} | Train Acc={train_acc:.4f}")

        # validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"             Val Acc={val_acc:.4f}")

 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f" Best model updated, Val Acc={val_acc:.4f} saved to {save_path}")


def evaluate_model(model, test_loader, device="cpu", model_path="best_model.pt"):
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    print("Test set evaluation")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return acc, precision, recall, f1