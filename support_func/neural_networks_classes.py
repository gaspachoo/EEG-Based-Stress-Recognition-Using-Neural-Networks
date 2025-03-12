import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, num_channels, num_timepoints, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_channels*num_timepoints, 128)
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

        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=1, padding=3) #kernel = filter
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
        self.fc1 = nn.Linear(num_channels*num_timepoints, 64)
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
        
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(2)
        
        conv_output_size = num_timepoints // 4  # Deux maxpoolings de taille 2
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
