import os
import torch
from typing import Dict, Any

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str = "./checkpoint", filename: str = 'checkpoint.pth'):
    """
    Save a checkpoint and, if is_best, also save as the best model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_best_model(model, optimizer=None, scheduler=None, metric: str = 'valid_acc', path: str = "./checkpoint/best_model.pth"):
    """
    Load a previously saved checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    metric_value = checkpoint.get(metric, None)
    return model, optimizer, scheduler, epoch, metric_value
