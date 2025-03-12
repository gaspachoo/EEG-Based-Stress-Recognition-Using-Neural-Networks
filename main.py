from support_func.model_processing import train_gen,train_with_early_stopping
from support_func.NN_classes import EEG_CNN
from support_func.results_evaluation import evaluate_with_metrics_gpu,plot_history
import torch

if __name__ == "__main__":
    data_folder = 'filtered_data'
    labels_file = 'scales.xls'
    num_classes = 3
    test_size = 0.2
    sampling_mode = "'oversampling" # None or "oversampling" or "undersampling"
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,test_loader, num_channels, num_timepoints = train_gen(data_folder, labels_file,num_classes, test_size=0.2, sampling_mode = None)

    print('✅ Data processed')
    cnn_model = EEG_CNN(num_channels, num_timepoints, num_classes).to(device)

    print("⚙️ Training CNN")
    cnn_model, cnn_history = train_with_early_stopping(cnn_model, train_loader, test_loader,device)

    cnn_final_acc = cnn_history['val_acc'][-1]
    print(f"🚀 Final Validation Accuracy: {cnn_final_acc:.2f}%")

    plot_history(cnn_history, 'CNN')
    evaluate_with_metrics_gpu(cnn_model, test_loader,device)
    