import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from support_func.import_data import load_dataset2
from support_func.dataset_class import EEGDataset_cla
from support_func.early_stopping_class import EarlyStopping
from sklearn.model_selection import train_test_split


def train_gen(
    data_folder,
    labels_filename,
    num_classes,
    test_size=0.2,
    lstm=False,
    sampling_mode=None,
):
    """Generates the train and test data loaders for classification."""

    # Load dataset
    data, labels = load_dataset2(
        data_folder, labels_filename
    )  # data: (40, 9, channels, time), labels: (40, 9)
    data = np.array(data, dtype=np.float32)

    if lstm:
        labels = np.median(labels, axis=1).astype(int)  # # Take the Median of 9 trials
        num_channels, num_timepoints = data.shape[2], data.shape[3]  # (channels, time)
    else:
        # Ensure shape is correct (Flatten subjects & trials)
        samples = data.shape[0] * data.shape[1]  # 40 * 9 = 360 samples
        data = data.reshape(
            samples, data.shape[2], data.shape[3]
        )  # (360, channels, time)
        labels = labels.reshape(samples)  # Flatten labels to match
        num_channels, num_timepoints = data.shape[1], data.shape[2]

    labels = labels - 1

    # Group labels (linspace or quantile binning)
    bins = np.linspace(labels.min(), labels.max(), num_classes + 1)[
        1:-1
    ]  # Distribute on evenly spaced points
    # bins = np.quantile(labels, np.linspace(0, 1, num_classes + 1))[1:-1]  # Evenly distribute data across nomber of samples in each class
    grouped_labels = np.digitize(labels, bins, right=True)
    # Plot class distribution as histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3))
    plt.bar(range(len(np.bincount(grouped_labels))), np.bincount(grouped_labels), color='skyblue')
    plt.title('Class Distribution (train_gen)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Stratified split
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices, stratify=grouped_labels, test_size=test_size, random_state=42
    )

    X_train, y_train = data[train_idx], grouped_labels[train_idx]
    X_test, y_test = data[test_idx], grouped_labels[test_idx]

    # Handle sampling if necessary
    if sampling_mode == "oversampling":
        X_train, y_train = random_oversample(X_train, y_train)
    elif sampling_mode == "undersampling":
        X_train, y_train = random_undersample(X_train, y_train)

    # Create datasets & dataloaders
    train_dataset = EEGDataset_cla(X_train, y_train)
    test_dataset = EEGDataset_cla(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, num_channels, num_timepoints


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_data, batch_labels in loader:
        # Transfert GPU
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_gen2(
    data_folder,
    labels_filename,
    num_classes,
    test_size=0.2,
    lstm=False,
    sampling_mode=None,
):
    """Generates dataloaders for classification."""

    # Data loading
    data, labels = load_dataset2(
        data_folder, labels_filename
    )  # (40, 9, num_segments, channels, time), (40, 9, num_segments)

    # Get dimensions
    num_subjects, num_trials = 40, 9
    num_segments, num_channels, num_timepoints = (
        12,
        32,
        256,
    )  # Based on your print output

    # Preallocate a fixed-size NumPy array instead of using an object array
    data_array = np.zeros(
        (num_subjects, num_trials, num_segments, num_channels, num_timepoints),
        dtype=np.float32,
    )
    labels_array = np.zeros((num_subjects, num_trials, num_segments), dtype=int)

    # Fill the array manually
    for i in range(num_subjects):
        for j in range(num_trials):
            if data[i, j] is not None:
                data_array[i, j] = data[i, j]  # Copy the (12, 32, 256) array
                labels_array[i, j] = labels[i, j]  # Copy the (12,) array
            else:
                print(f"Warning: Missing data at subject {i}, trial {j}")  # Debug info

    # Now, data_array and labels_array can be safely used
    data = data_array
    labels = labels_array

    # Convert data to numpy
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=int)

    if lstm:
        # Aggregate labels over trials AND segments to get (40,)
        labels = np.median(labels, axis=(1, 2)).astype(int)  # (40,)

        # Bin labels
        bins = np.histogram_bin_edges(labels, bins=num_classes)[1:-1]
        grouped_labels = np.digitize(labels, bins, right=True)

        print(
            "Grouped labels shape after binning:", grouped_labels.shape
        )  # Should be (40,)

        # Fix the shape of `data` for LSTM
        seq_len = data.shape[1] * data.shape[2]  # trials * num_segments (9 * 12)
        data = data.reshape(40, seq_len, data.shape[3], data.shape[4])

        num_channels, num_timepoints = (
            data.shape[2],
            data.shape[3],
        )  # Keep correct dimensions

    else:
        # Flatten (40, 9, num_segments, channels, time) → (total_samples, channels, time)
        samples = (
            data.shape[0] * data.shape[1] * data.shape[2]
        )  # (40 * 9 * num_segments)
        data = data.reshape(
            samples, data.shape[3], data.shape[4]
        )  # (total_samples, channels, time)
        labels = labels.reshape(samples)  # Flatten labels

        # Bin labels
        bins = np.histogram_bin_edges(labels, bins=num_classes)[1:-1]
        grouped_labels = np.digitize(labels, bins, right=True)

    # Plot class distribution as histogram
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3))
    plt.bar(range(len(np.bincount(grouped_labels))), np.bincount(grouped_labels), color='skyblue')
    plt.title('Class Distribution (train_gen2)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Ensure `grouped_labels` is 1D
    grouped_labels = grouped_labels.flatten()

    # Check for class imbalance
    min_class_count = np.min(np.bincount(grouped_labels))
    if min_class_count < 2:
        print(
            "Warning: Some classes have less than 2 samples. Disabling stratification."
        )
        stratify = None
    else:
        stratify = grouped_labels

    # Split train/test
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices, stratify=stratify, test_size=test_size, random_state=42
    )

    X_train, y_train = data[train_idx], grouped_labels[train_idx]
    X_test, y_test = data[test_idx], grouped_labels[test_idx]

    # Handle sampling
    if sampling_mode == "oversampling":
        X_train, y_train = random_oversample(X_train, y_train)
    elif sampling_mode == "undersampling":
        X_train, y_train = random_undersample(X_train, y_train)

    # Create DataLoaders
    train_dataset = EEGDataset_cla(X_train, y_train)
    test_dataset = EEGDataset_cla(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, num_channels, num_timepoints


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=50,
    patience=7,
    lr=0.001,
    criterion=nn.CrossEntropyLoss(),
):
    """Train given model with given train loader and validation loader. Please specify the device to use.

    Args:
        model (class): class of a Model
        train_loader (torch.utils.Data.DataLoader): train loader, given by train_gen func
        val_loader (torch.utils.Data.DataLoader): validation loader, given by train_gen func
        num_epochs (int, optional): max number of epochs (can be less if early stopping). Defaults to 50.
        patience (int, optional): patience before early stopping. Defaults to 7.
        lr (float, optional): adam gamma coefficient. Defaults to 0.001.
        criterion (_type_, optional): loss function. Defaults to nn.CrossEntropyLoss().

    Returns:
        model, model_history (ndarray)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Using device: {device}")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping at epoch {epoch+1}.")
            break

    return model, history


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            # ⬇️ Transfert GPU
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)  # Extract class predictions
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def random_oversample(data, labels):

    unique_classes, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    new_data = []
    new_labels = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        repeats = max_count // len(cls_indices)
        remainder = max_count % len(cls_indices)

        # Duplicate samples
        new_data.append(np.repeat(data[cls_indices], repeats, axis=0))
        new_labels.append(np.repeat(labels[cls_indices], repeats))

        if remainder > 0:
            chosen = np.random.choice(cls_indices, remainder, replace=False)
            new_data.append(data[chosen])
            new_labels.append(labels[chosen])

    data_balanced = np.vstack(new_data)
    labels_balanced = np.concatenate(new_labels)
    return data_balanced, labels_balanced


def random_undersample(data, labels):

    unique_classes, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()

    new_data = []
    new_labels = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        selected = np.random.choice(cls_indices, min_count, replace=False)
        new_data.append(data[selected])
        new_labels.append(labels[selected])

    data_balanced = np.vstack(new_data)
    labels_balanced = np.concatenate(new_labels)
    return data_balanced, labels_balanced
