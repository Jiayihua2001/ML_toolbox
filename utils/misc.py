import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchinfo import summary


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Return 'cuda' if available, else 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)




def plot_training_curves(train_losses, val_losses,
                         train_accuracies=None, val_accuracies=None,
                         title="Training Curves", save_path=None):
    """
    Plot training and validation loss curves and, optionally, accuracy curves.

    Args:
        train_losses (list or array): Loss values for the training set over epochs.
        val_losses (list or array): Loss values for the validation set over epochs.
        train_accuracies (list or array, optional): Accuracy values for the training set.
        val_accuracies (list or array, optional): Accuracy values for the validation set.
        title (str): Plot title.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, train_losses, label="Train Loss", color="tab:blue", marker="o")
    ax1.plot(epochs, val_losses, label="Val Loss", color="tab:blue", linestyle="--", marker="o")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    if train_accuracies is not None and val_accuracies is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Accuracy", color="tab:orange")
        ax2.plot(epochs, train_accuracies, label="Train Acc", color="tab:orange", marker="x")
        ax2.plot(epochs, val_accuracies, label="Val Acc", color="tab:orange", linestyle="--", marker="x")
        ax2.tick_params(axis='y', labelcolor="tab:orange")
        ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()


def compute_classification_metrics(y_true, y_pred, average='macro'):
    """
    Compute commonly used classification metrics.

    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.
        average (str): Averaging method for multi-class data ('macro', 'micro', 'weighted').

    Returns:
        dict: Dictionary with accuracy, precision, recall, and F1 score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics


def print_model_summary(model, input_size, device="cpu"):
    """
    Print a detailed summary of the model architecture using torchinfo.

    Args:
        model (torch.nn.Module): The model to summarize.
        input_size (tuple or list of tuples): Input size(s) for the model (e.g. (1, 3, 224, 224)).
        device (str): Device to perform the summary on (e.g. "cpu" or "cuda").
    """
    print(summary(model, input_size=input_size, device=device))
    

# Example usage:
if __name__ == "__main__":
    # Example: Plotting training curves
    train_losses = [1.0, 0.8, 0.6, 0.5]
    val_losses = [1.1, 0.9, 0.7, 0.55]
    train_acc = [60, 70, 75, 78]
    val_acc = [58, 68, 73, 76]
    plot_training_curves(train_losses, val_losses, train_accuracies=train_acc, val_accuracies=val_acc,
                           title="Example Training Curves", save_path="training_curves.png")

    # Example: Compute metrics for dummy data
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 0, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    print("Classification Metrics:", metrics)

    # Example: Print model summary for a simple CNN
    import torch.nn as nn
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(16 * 16 * 16, 10)  # assuming input images are 32x32
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleCNN().to("cpu")
    print_model_summary(model, input_size=(1, 3, 32, 32))
