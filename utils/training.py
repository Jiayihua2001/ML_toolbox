import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def train_step(train_loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0
    batch_bar = tqdm(total=len(train_loader), desc="Train", leave=False)
    scaler = GradScaler()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        with autocast():
            out, out_lengths = model(x, lx)
            loss = criterion(torch.transpose(out, 0, 1), y, out_lengths, ly)
        running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_bar.set_postfix(loss=f"{running_loss/(i+1):.4f}")
        batch_bar.update()
    batch_bar.close()
    return running_loss / len(train_loader)

def val_step(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0
    batch_bar = tqdm(total=len(val_loader), desc="Val", leave=False)
    with torch.inference_mode():
        for i, data in enumerate(val_loader):
            x, y, lx, ly = data
            x, y = x.to(device), y.to(device)
            out, out_lengths = model(x, lx)
            loss = criterion(torch.transpose(out, 0, 1), y, out_lengths, ly)
            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{running_loss/(i+1):.4f}")
            batch_bar.update()
    batch_bar.close()
    return running_loss / len(val_loader)
