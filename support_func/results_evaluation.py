import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import torch
import seaborn as sns

def plot_history_cla(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f"{model_name} Training History")
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()
    
def plot_history_reg(history, model_name):
    """Plot training & validation loss over epochs for regression."""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10,5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training History - {model_name}')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_with_metrics_gpu(model, loader, device, num_classes):
    model.eval()

    # ✅ Adapt F1-Score and Confusion Matrix to dynamic num_classes
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='weighted').to(device)
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            _, preds = torch.max(outputs, 1)

            f1_metric.update(preds, batch_labels)
            cm_metric.update(preds, batch_labels)

    f1_score_gpu = f1_metric.compute()
    confusion_matrix_gpu = cm_metric.compute()

    print(f"Weighted F1-score (GPU): {f1_score_gpu:.4f}")

    # ✅ Move Confusion Matrix to CPU and convert to NumPy
    cm_cpu = confusion_matrix_gpu.cpu().numpy()

    # ✅ Normalize by actual class (row-wise)
    cm_normalized = cm_cpu.astype('float') / cm_cpu.sum(axis=1, keepdims=True)

    # ✅ Dynamically set class labels
    if num_classes == 2:
        class_labels = ["Low", "High"]
    elif num_classes == 3:
        class_labels = ["Low", "Medium", "High"]
    else:
        class_labels = [str(i+1) for i in range(num_classes)]  # 1,2,3,...,num_classes

    # ✅ Plot Normalized Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar_kws={'label': 'Proportion'})
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()
