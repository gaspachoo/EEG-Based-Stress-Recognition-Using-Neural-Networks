from support_func.model_processing import train_gen,train_with_early_stopping
from support_func.neural_networks_classes import EEG_CNN
from support_func.results_evaluation import evaluate_with_metrics_gpu,plot_history
import torch

if __name__ == "__main__":
    num_classes=3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,test_loader, num_channels, num_timepoints = train_gen('filtered_data','scales.xls',num_classes = num_classes, sampling_mode='oversampling')

    print('Data processed 100 %')
    cnn_model = EEG_CNN(num_channels, num_timepoints, num_classes).to(device)

    print("\n🔹 Training CNN")
    cnn_model, cnn_history = train_with_early_stopping(cnn_model, train_loader, test_loader,device)

    cnn_final_acc = cnn_history['val_acc'][-1]
    print(f"Final Validation Accuracy: {cnn_final_acc:.2f}%")

    plot_history(cnn_history, 'CNN')
    evaluate_with_metrics_gpu(cnn_model, test_loader,device)
    