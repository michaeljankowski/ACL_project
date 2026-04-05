import pandas as pd
import os

CSV_IN  = "vlo_vmo_pairs_with_labels.csv"
CSV_OUT = "vlo_vmo_pairs_with_labels_local.csv"

df = pd.read_csv(CSV_IN)

repo_root = os.path.abspath(".")
marker = "dat files_emg data/"

def rewrite(p):
    p = str(p)
    if marker not in p:
        return p
    rel = p.split(marker, 1)[1]
    return os.path.join(repo_root, marker, rel)

df["VLO_path"] = df["VLO_path"].apply(rewrite)
df["VMO_path"] = df["VMO_path"].apply(rewrite)

df.to_csv(CSV_OUT, index=False)
print("Wrote:", CSV_OUT)
print("Example new paths:")
print(df["VLO_path"].iloc[0])
print(df["VMO_path"].iloc[0])
print("Exists checks:")
print(os.path.exists(df["VLO_path"].iloc[0]), os.path.exists(df["VMO_path"].iloc[0]))
