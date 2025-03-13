from support_func.model_processing_reg import train_gen, train_with_early_stopping
from support_func.NN_classes import *
from support_func.results_evaluation import evaluate_with_metrics_gpu, plot_history_reg
import torch

if __name__ == "__main__":
    data_folder = 'filtered_data'
    labels_file = 'scales.xls'
    test_size = 0.2
    sampling_mode = None  # None, "oversampling", or "undersampling"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data (Regression Mode)
    train_loader, test_loader, num_channels, num_timepoints = train_gen(
        data_folder, labels_file, test_size=test_size, sampling_mode=sampling_mode
    )

    print('✅ Data processed')

    # Define CNN model for Regression (num_classes = 1)
    cnn_model = SimpleNN3(num_channels, num_timepoints,num_classes=1).to(device)

    print("⚙️ Training CNN for Regression")
    cnn_model, cnn_history = train_with_early_stopping(
        cnn_model, train_loader, test_loader, device
    )

    # Final Validation Loss (MSE)
    cnn_final_loss = cnn_history['val_loss'][-1]
    print(f"🚀 Final Validation MSE: {cnn_final_loss:.4f}")

    # Plot Training History (Now tracking loss instead of accuracy)
    plot_history_reg(cnn_history, 'CNN')
