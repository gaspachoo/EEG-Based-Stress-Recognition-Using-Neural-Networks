import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from support_func.import_data import load_dataset
from sklearn.model_selection import train_test_split
from support_func.dataset_class import EEGDataset_reg
from support_func.early_stopping_class import *

# ✅ Load dataset function (EEG data & stress labels)
def reshape_dataset(data_folder, labels_filename):
    data, labels = load_dataset(data_folder, labels_filename)
    data = np.array(data, dtype=np.float32) # Data shape before reshaping:  (40, 9, channels, time)
        
    num_channels, num_timepoints = data.shape[2], data.shape[3]  # (40, 9, channels, time)
    
    # Reshape (40 subjects, 9 trials) → (360, channels, time)
    X = data.reshape(-1, num_channels, num_timepoints)
    
    # Stress levels are already continuous → No need for binning
    y = labels.reshape(-1).astype(np.float32)  # (360,)

    return X, y, num_channels, num_timepoints

# ✅ Data Preparation with Train-Test Split
def train_gen(data_folder, labels_filename, test_size=0.2, sampling_mode=None):
    """Prepares EEG data for training by generating PyTorch DataLoaders for regression.
    
    Args:
        data_folder (str): Name of the data folder.
        labels_filename (str): Name of the labels file.
        test_size (float, optional): Percentage of data used for testing. Default = 0.2.
        sampling_mode (str, optional): 'oversampling' or 'undersampling'. Default = None.

    Returns:
        train_loader, test_loader, num_channels, num_timepoints
    """
    
    # Load dataset
    X, y, num_channels, num_timepoints = reshape_dataset(data_folder, labels_filename)
    
    # Train-Test Split (Stratified split no longer needed)
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_size, random_state=42)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    '''
    # Handle sampling strategies if needed
    if sampling_mode == 'oversampling':
        X_train, y_train = random_oversample(X_train, y_train)
    elif sampling_mode == 'undersampling':
        X_train, y_train = random_undersample(X_train, y_train)
    '''
    # Create PyTorch Datasets
    train_dataset = EEGDataset_reg(X_train, y_train)
    test_dataset = EEGDataset_reg(X_test, y_test)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, num_channels, num_timepoints

# ✅ Train One Epoch (For Regression)
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_data, batch_labels in loader:
        # Move data to GPU
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

# ✅ Training with Early Stopping
def train_with_early_stopping(model, train_loader, val_loader, device, num_epochs=50, patience=7, lr=0.001):
    """Train model with early stopping (Regression).
    
    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int, optional): Max epochs. Default = 50.
        patience (int, optional): Early stopping patience. Default = 7.
        lr (float, optional): Learning rate. Default = 0.001.

    Returns:
        model, history (dict)
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean Squared Error for Regression
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': []}

    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping at epoch {epoch+1}.")
            break

    return model, history

# ✅ Validation Function (Regression)
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            # Move data to GPU
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss