import numpy as np
import matplotlib.pyplot as plt
import os

histories_dir = 'histories'

# toggle if you want to switch
PLOT_MODE = 'every_5'  
#PLOT_MODE = 'every_epoch'

files = {
    'Chronos(small)':  'chronos_test_chronos-t5-base_lr0.0001_bs32_ep50_history.npy',
    'CNN-LSTM':        'test_cnnlstm_lr0.001_bs64_ep50_history.npy',

}

histories = {
    name: np.load(os.path.join(histories_dir, path), allow_pickle=True).item()
    for name, path in files.items()
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metrics = [('val_f1', 'Validation F1 Score'), ('val_recall', 'Validation Recall')]

for ax, (key, title) in zip(axes, metrics):
    for name, hist in histories.items():
        if key not in hist:
            print(f"Warning: '{key}' not found in {name}, skipping.")
            continue

        values = hist[key]
        epochs = np.arange(1, len(values) + 1)

        if PLOT_MODE == 'every_5':
            # full line + dots every 5 epochs
            marker_idx    = [i for i in range(len(epochs)) if (i + 1) % 5 == 0]
            marker_epochs = epochs[marker_idx]
            marker_values = [values[i] for i in marker_idx]
            line, = ax.plot(epochs, values, linewidth=1.5)
            ax.plot(marker_epochs, marker_values, marker='o',
                    color=line.get_color(), linestyle='none', label=name)

        elif PLOT_MODE == 'every_epoch':
            # dot at every epoch
            ax.plot(epochs, values, marker='o', linewidth=1.5,
                    markersize=4, label=name)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(key.replace('_', ' ').title())
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Model Comparison — F1 & Recall', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()