# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Fully Connected Network (FCN)
# class FCN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(500, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, 2)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = x.squeeze(1)  
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.softmax(self.fc3(x))

# # 1D CNN
# class CNN(nn.Module):
#     def __init__(self, in_channels=1, seq_len=1000, num_classes=2):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.pool1 = nn.MaxPool1d(2)

#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.pool2 = nn.MaxPool1d(2)

#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.global_avg_pool(x).squeeze(-1)
#         return F.softmax(self.fc(x), dim=1)

# # LSTM
# class LSTMModel(nn.Module):
#     def __init__(self, in_channels=1, seq_len=1000, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.5):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=in_channels,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         # For bidirectional LSTM, the output size is hidden_dim * 2
#         self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         # (B, C, T) -> (B, T, C)
#         x = x.permute(0, 2, 1)
        
#         # Forward propagate LSTM
#         out, _ = self.lstm(x)
        
#         # Take the last time step's output
#         out = out[:, -1, :]
        
#         # Apply fully connected layers
#         out = F.relu(self.fc1(out))
#         out = self.dropout(out)
#         out = self.fc2(out)
        
#         return F.softmax(out, dim=1)

# # CNN + LSTM
# class CNNLSTM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
#         self.pool = nn.MaxPool1d(2)
#         self.lstm = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)
#         self.fc = nn.Linear(64, 2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # [B, 16, 150]
#         x = self.pool(x)  # [B, 16, 75]
#         x = x.permute(0, 2, 1)  # [B, 75, 16]
#         x, _ = self.lstm(x)
#         return F.softmax(self.fc(x[:, -1, :]), dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F

# === FCN ===
class FCN(nn.Module):
    def __init__(self, input_dim=2, window=500, num_classes=2):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(window * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, window, 2]
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# === CNN ===
class CNN1D(nn.Module):
    def __init__(self, input_channels=2, num_classes=2):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, window, 2] → [batch, 2, window]
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# === LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch, window, 2]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last hidden state
        out = self.fc(out)
        return out

# === CNN + LSTM ===
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=64, num_layers=1, num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # CNN 部分
        x = x.permute(0, 2, 1)  # [batch, 2, window]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [batch, window, channels]

        # LSTM 部分
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out