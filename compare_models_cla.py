#!/usr/bin/env python3
"""Compare classification models defined in `support_func/NN_classes.py`.

This script will:
- load data using `train_gen2` (classification pipeline)
- iterate over a list of model classes, instantiate each with the right
  input shapes, train using `train_with_early_stopping`, evaluate on test set,
  save the model checkpoint and training history, and print a short summary.

Usage:
    python compare_models_cla.py

Adjust the `MODEL_LIST` below to add/remove models to compare.
"""
from pathlib import Path
import torch
import random
import numpy as np
import json

from support_func.model_processing_cla import (
    train_gen2,
    train_with_early_stopping,
)
from support_func import NN_classes
from support_func.results_evaluation import (
    evaluate_with_metrics_gpu,
    plot_history_cla,
)


def get_model_classes():
    # Select the model classes you want to compare (by name)
    # Available names (from NN_classes.py):
    # - SimpleNN
    # - EEG_CNN
    # - SimpleNN2
    # - EEG_CNN2
    # - EEG_CNN_GRU
    # - EEG_LSTM
    # Note: EEG_LSTM is disabled by default because it expects a different
    # data shape. Enable it only if you call `train_gen2(..., lstm=True)`.
    names = [
        "SimpleNN",
        "EEG_CNN",
        "EEG_CNN2",
    ]
    cls_list = []
    for n in names:
        if hasattr(NN_classes, n):
            cls_list.append(getattr(NN_classes, n))
        else:
            print(f"Warning: model {n} not found in NN_classes")
    return cls_list


def main():
    # Config
    # Reproducibility: fixed seed for comparison
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using fixed seed: {seed}")

    data_folder = "filtered_data"
    labels_file = "scales.xls"
    num_classes = 2
    test_size = 0.2
    out_dir = Path("./ComparisonResults")
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (non-LSTM path)
    train_loader, test_loader, num_channels, num_timepoints = train_gen2(
        data_folder, labels_file, num_classes, test_size=test_size, lstm=False
    )

    model_classes = get_model_classes()
    summary = []

    for ModelCls in model_classes:
        model_name = ModelCls.__name__
        print(f"\n=== Training model: {model_name} ===")

        model = ModelCls(num_channels, num_timepoints, num_classes).to(device)

        # Train
        trained_model, history = train_with_early_stopping(
            model, train_loader, test_loader, device, lr=1e-4, num_epochs=100
        )

        # Save checkpoint and history
        ckpt_path = out_dir / f"{model_name}.pth"
        torch.save(trained_model.state_dict(), ckpt_path)

        hist_path = out_dir / f"{model_name}_history.json"
        # Convert history arrays to lists for JSON
        serial_hist = {k: [float(x) for x in v] for k, v in history.items()}
        with open(hist_path, "w") as fh:
            json.dump(serial_hist, fh, indent=2)

        # Plot training history
        try:
            plot_history_cla(history, model_name)
        except Exception as e:
            print(f"Plotting history failed for {model_name}: {e}")

        # Eval
        try:
            metrics = evaluate_with_metrics_gpu(
                trained_model, test_loader, device, num_classes
            )
        except Exception as e:
            print(f"Evaluation failed for {model_name}: {e}")
            metrics = None

        final_val_acc = None
        if len(history.get("val_acc", [])) > 0:
            final_val_acc = history["val_acc"][-1]
        summary.append(
            {
                "model": model_name,
                "checkpoint": str(ckpt_path),
                "history": str(hist_path),
                "final_val_acc": final_val_acc,
                "metrics": metrics,
            }
        )

    # Print compact table
    print("\n=== Comparison summary ===")
    for s in summary:
        print(
            f"{s['model']}: final_val_acc={s['final_val_acc']},"
            f" checkpoint={s['checkpoint']}"
        )

    # Save summary
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
