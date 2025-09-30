import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_channels * num_timepoints, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (channels, time)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Outputs raw scores for CrossEntropyLoss
        return x


class EEG_CNN(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes):
        super(EEG_CNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
        )  # kernel = filter
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Compute the flattened size after convolutions and pooling
        conv_output_size = num_timepoints // (2 * 2 * 2)  # 3 poolings of size 2
        self.fc1 = nn.Linear(256 * conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch_size, channels, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleNN2(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_channels * num_timepoints, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class EEG_CNN2(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=32, kernel_size=7, padding=3
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(2)

        conv_output_size = num_timepoints // 4  # Two max-pooling layers of size 2
        self.fc1 = nn.Linear(64 * conv_output_size, 64)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout3(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EEG_CNN_GRU(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        # GRU
        gru_input_size = num_timepoints // 2  # after pooling
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)

        # Fully connected
        self.fc1 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # GRU expects (batch, seq_len, feature), so we permute
        x = x.permute(0, 2, 1)  # (batch_size, time_steps, features)

        # GRU
        _, h_n = self.gru(x)  # we take the last hidden state
        x = h_n.squeeze(0)  # (batch_size, hidden_size)

        # Fully connected layers
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimpleNN3(nn.Module):  #### Adapted for reg
    def __init__(self, num_channels, num_timepoints, num_classes, hidden_dim=128):
        super().__init__()
        input_size = num_channels * num_timepoints  # Flatten EEG input

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(
            hidden_dim, num_classes
        )  # Output a single value (regression)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input to (batch_size, input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation (linear output for regression)
        return x


class EEG_CNN3(nn.Module):  #### Adapted for reg
    def __init__(self, num_channels, num_timepoints, num_classes):  # ✅ Fix this
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * (num_timepoints // 8), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(
            64, num_classes
        )  # ✅ Ensure this outputs a single regression value

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # ✅ No activation function (linear output for regression)

        return x


class EEG_LSTM(nn.Module):
    def __init__(
        self, num_channels, num_timepoints, num_classes, hidden_dim=128, num_layers=2
    ):
        super(EEG_LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=num_channels * num_timepoints,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len, channels, timepoints = x.shape
        x = x.view(batch_size, seq_len, -1)  # ✅ Flatten (channels, time) per trial

        _, (hidden, _) = self.lstm(x)  # ✅ Retrieve the last hidden state of the LSTM
        x = self.fc(hidden[-1])  # ✅ Final fully connected layer for classification

        return x
