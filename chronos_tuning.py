"""
   Since chronos is not a classifier,
    Using linear probing:
        1. Freeze the Chronos encoder entirely so it never updates weights
        2. Extract embeddings from the frozen encoder for each channel (VLO, VMO)
        3. Add the two channel embeddings into one vector
        4. Train a small MLP classification head on top of those embeddings

    we use Chronos's pretrained encoder to extract features, then train only the final classification head.

    we embed VLO and VMO separately then fuse them. This lets us study whether the
    channel fusion strategy affects performance.

Install:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git

Run:
    python chronos_tuning.py --epochs 50 --lr 1e-4 --batch_size 32
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight  
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 
from chronos import ChronosPipeline
import os

from utils import EMGDataset  # same dataset class as baseline



# We store these so the classifier head automatically sizes itself correctly depending on which model variant is being used.
EMBED_DIMS = {
    "amazon/chronos-t5-tiny":  64,
    "amazon/chronos-t5-mini":  128,
    "amazon/chronos-t5-small": 512,
    "amazon/chronos-t5-base":  768,
}


# Classifier head
# This is the only part that gets trained the Chronos encoder is frozen.
# Same role as the final fc layer in CNN_LSTM from model.py, but here it sits
# on top of pretrained embeddings instead of learned CNN/LSTM features.
class ChronosClassifier(nn.Module):
    """
    embed_dim:   output dimension of the Chronos encoder (model-size dependent)
    n_channels:  2 (VLO + VMO) — we embed each channel separately then concat
    num_classes: 2 (AMI-present / AMI-absent)
    """
    def __init__(self, embed_dim=512, n_channels=2, num_classes=2, dropout=0.1):
        super().__init__()
        fused_dim = embed_dim * n_channels  # VLO+VMO embedding concatenated

        
        # Similar in spirit to the fc layers in CNN_LSTM, but operating on
        # pretrained embeddings instead of raw learned features
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),       
            nn.Linear(fused_dim, 128),      #compress fused embedding to 128
            nn.GELU(),                      #smooth
            nn.Dropout(dropout),            # regularization 
            nn.Linear(128, num_classes),    # final output: 2 classes (AMI / no AMI)
        )

    def forward(self, emb_vlo, emb_vmo):
        # emb_vlo, emb_vmo: [B, embed_dim] — already mean-pooled over time
        fused = torch.cat([emb_vlo, emb_vmo], dim=-1)  # [B, 2*embed_dim]
        return self.head(fused)


# Run the frozen Chronos encoder ONCE before training to get embeddings.
# Because the encoder never updates, we don't need to rerun it every epoch
#no gradients computed
@torch.no_grad()
def extract_embeddings(pipeline, X, device, batch_size=32):

    all_vlo, all_vmo = [], []
    N = len(X)

    for start in range(0, N, batch_size):
        batch = X[start:start + batch_size]  # [B, T, 2]

        #
        vlo_series = [torch.tensor(s[:, 0], dtype=torch.float32) for s in batch]  # VLO 
        vmo_series = [torch.tensor(s[:, 1], dtype=torch.float32) for s in batch]  # VMO

        # pipeline.embed returns (embedding, tokenizer_state)
        emb_vlo, _ = pipeline.embed(vlo_series)  
        emb_vmo, _ = pipeline.embed(vmo_series) 

        
        all_vlo.append(emb_vlo.mean(dim=1).float().cpu())
        all_vmo.append(emb_vmo.mean(dim=1).float().cpu())

    return torch.cat(all_vlo, dim=0), torch.cat(all_vmo, dim=0)



# Wraps pre-computed embeddings + labels for fast DataLoader iteration.
# Since embeddings are already extracted, each training iteration just loads
# two vectors and a label
# Same role as EMGDataset in utils.py but holds embeddings instead of raw data.
class EmbeddingDataset(Dataset):
    def __init__(self, emb_vlo, emb_vmo, labels):
        self.emb_vlo = emb_vlo
        self.emb_vmo = emb_vmo
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emb_vlo[idx], self.emb_vmo[idx], self.labels[idx]



# Same strategy as train_model in utils.py:
#   - AdamW optimizer with weight decay (L2 regularization)
#   - Class weights to handle AMI class imbalance
#   - Gradient clipping to stabilize training
#   - Track val_f1 and val_recall per epoch → saved to history for plotting
#   - Save best model based on val_f1 (not val_acc)
#
#  adds a learning rate scheduler (ReduceLROnPlateau)
# which halves lr if val_f1 doesn't improve for 5 epochs 
def train_chronos(classifier, train_loader, val_loader, epochs, lr,
                  device, save_path, y_train):
    classifier.to(device)

    # AdamW
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=lr, weight_decay=1e-4
    )

    # Class weights 
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.array([0, 1]), y=y_train
    )
    #print(f"[Loss] Class weights — AMI-absent(0): {class_weights[0]:.4f}, "
    #      f"AMI-present(1): {class_weights[1]:.4f}")

    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    #LR scheduler reduces lr by 50% if val_f1 doesn't improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_f1 = 0.0

    history = {"train_loss": [], "train_acc": [],
               "val_acc": [], "val_f1": [], "val_recall": []}

    for epoch in range(epochs):
        classifier.train()
        train_losses, train_preds, train_labels = [], [], []

        for emb_vlo, emb_vmo, y in train_loader:
            emb_vlo = emb_vlo.to(device)
            emb_vmo = emb_vmo.to(device)
            y       = y.to(device)

            optimizer.zero_grad()
            logits = classifier(emb_vlo, emb_vmo)
            loss   = criterion(logits, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss={np.mean(train_losses):.4f} | "
              f"Train Acc={train_acc:.4f}")

        classifier.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for emb_vlo, emb_vmo, y in val_loader:
                logits = classifier(emb_vlo.to(device), emb_vmo.to(device))
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(y.numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        # F1 and recall
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary', zero_division=0
        )
        print(f"             Val Acc={val_acc:.4f} | "
              f"Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}")

      
        history["train_loss"].append(np.mean(train_losses))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_rec)

        # Step scheduler based on val_f1
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(classifier.state_dict(), save_path)
            print(f" Best model saved → Val F1={val_f1:.4f}")

    return history 



#Similair to evaluate_model, loads best checkpoint, runs on
#test set with no gradient updates, returns acc/precision/recall/f1.

def evaluate_chronos(classifier, test_loader, device, model_path):
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():  
        for emb_vlo, emb_vmo, y in test_loader:
            logits = classifier(emb_vlo.to(device), emb_vmo.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    print("\n── Chronos Test Results ─────────────────")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("─────────────────────────────────────────")
    return acc, precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str,
                        default="/Users/michaeljankowski/UConnCS/ACL_Project-main/processed_data")
    parser.add_argument("--model_name", type=str,
                        default="amazon/chronos-t5-small")  # swap for chronos-t5-base for larger model
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--device",     type=str,
                        default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()


    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("histories",  exist_ok=True)

    
    args.save_path = f"checkpoint/best_chronos_{args.model_name.split('/')[-1]}.pt"

    print(f"Device:     {args.device}")
    print(f"Model:      {args.model_name}")

    print("Loading Chronos encoder...")
    pipeline = ChronosPipeline.from_pretrained(
        args.model_name,
        device_map=args.device,
        torch_dtype=torch.float32,  
    )
    pipeline.model.eval()  #encoder stays in eval mode 

    #load data
    X_train = np.load(f"{args.data_dir}/X_train.npy")
    y_train = np.load(f"{args.data_dir}/y_train.npy")
    X_val   = np.load(f"{args.data_dir}/X_val.npy")
    y_val   = np.load(f"{args.data_dir}/y_val.npy")
    X_test  = np.load(f"{args.data_dir}/X_test.npy")
    y_test  = np.load(f"{args.data_dir}/y_test.npy")

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


    print("Extracting train embeddings...")
    train_vlo, train_vmo = extract_embeddings(pipeline, X_train, args.device, args.batch_size)

    print("Extracting val embeddings...")
    val_vlo, val_vmo = extract_embeddings(pipeline, X_val, args.device, args.batch_size)

    print("Extracting test embeddings...")
    test_vlo, test_vmo = extract_embeddings(pipeline, X_test, args.device, args.batch_size)

    train_loader = DataLoader(
        EmbeddingDataset(train_vlo, train_vmo, y_train),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        EmbeddingDataset(val_vlo, val_vmo, y_val),
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        EmbeddingDataset(test_vlo, test_vmo, y_test),
        batch_size=args.batch_size
    )

   
    embed_dim = EMBED_DIMS.get(args.model_name, 512)
    classifier = ChronosClassifier(embed_dim=embed_dim, n_channels=2, num_classes=2)
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier head parameters: {total_params:,}")

  
    history = train_chronos(
        classifier, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=args.device,
        save_path=args.save_path, y_train=y_train
    )

    # test set(comment_)
    print("\nEvaluating on test set...")
    acc, precision, recall, f1 = evaluate_chronos(classifier, test_loader,
                      device=args.device, model_path=args.save_path)
    history["test_f1"]        = f1
    history["test_recall"]    = recall
    history["test_acc"]       = acc
    history["test_precision"] = precision
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs("histories", exist_ok=True)

    hist_path = f"histories/chronos_test_{args.model_name.split('/')[-1]}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_history.npy"
    np.save(hist_path, history)
    print(f"History saved to {hist_path}")
    


if __name__ == "__main__":
    main()