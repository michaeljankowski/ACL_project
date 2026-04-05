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
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

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





def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu", save_path="best_model.pt", y_train=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) #L2 regularization

    # BALANCE WEIGHT
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]),y=y_train)
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    #measure by f1 score instead
    best_val_f1 = 0.0

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "val_recall": []} 
    for epoch in range(epochs):
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            #limit the size of gradients during backpropagation to prevent exploding gradients and stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
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
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support( 
            val_labels, val_preds, average='binary', zero_division=0
        )
 
        history["train_loss"].append(np.mean(train_losses))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_rec)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f" Best model updated, Val Acc={val_acc:.4f} saved to {save_path}")
    return history

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