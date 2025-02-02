import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from tqdm import tqdm



#logger

def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with the specified name and log file.

    Args:
        name (str): Name of the logger.
        log_file (str): File path to log the information.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# # Example usage
# logger = setup_logger('my_logger', 'training.log')
# logger.info('Logger is set up and ready to log information.')

# data loading

# define audio dataset
class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, partition, PHONEMES):

        self.data_path = data_path

        if partition == 'train':
            self.mfcc_dir = os.path.join(self.data_path, f'{partition}-clean-100/', 'mfcc/')
            self.transcript_dir = os.path.join(self.data_path, f'{partition}-clean-100/', 'transcript/')
        else:
            self.mfcc_dir = os.path.join(self.data_path, f'{partition}-clean/', 'mfcc/')
            self.transcript_dir = os.path.join(self.data_path, f'{partition}-clean/', 'transcript/')

        self.PHONEMES = PHONEMES

        mfcc_names = os.listdir(self.mfcc_dir)
        transcript_names = os.listdir(self.transcript_dir)

        assert len(mfcc_names) == len(transcript_names)

        self.mfccs, self.transcripts = [], []

        for i in range(0, len(mfcc_names)):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            scaler = StandardScaler()   # standscalar
            mfcc = scaler.fit_transform(mfcc)
            transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]))
            # from phonemes to index
            transcript_idx = []
            for i in transcript:
                transcript_idx.append(PHONEMES.index(i))
            transcript = np.array(transcript_idx)
            # add to collection list
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        self.length = len(mfcc_names)
        
    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        mfcc = self.mfccs[idx]
        transcript = self.transcripts[idx]

        return mfcc, transcript

    def pad_and_pack(batch):
        # collect mfcc and transcript within a batch
        batch_mfcc = [torch.tensor(x) for x, y in batch]
        len_mfcc = [len(x) for x in batch_mfcc]
        batch_transcript = [torch.tensor(y) for x, y in batch]
        len_transcript = [len(y) for y in batch_transcript]
        #pad the seq
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(len_mfcc), torch.tensor(len_transcript)


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path,PHONEMES):

        self.data_path = data_path
        self.mfcc_dir = os.path.join(self.data_path, 'test-clean/mfcc/')
        self.PHONEMES = PHONEMES

        mfcc_names = os.listdir(self.mfcc_dir)
        mfcc_names.sort()
        self.mfccs= []

        for i in range(0, len(mfcc_names)):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            scaler = StandardScaler()   # standscalar
            mfcc = scaler.fit_transform(mfcc)
            # add to collection list
            self.mfccs.append(mfcc)

        self.length = len(mfcc_names)
        
    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        mfcc = self.mfccs[idx]
        return mfcc

    def pad_and_pack(batch):
        # collect mfcc and transcript within a batch
        batch_mfcc = [torch.tensor(x) for x, y in batch]
        len_mfcc = [len(x) for x in batch_mfcc]

        #pad the seq
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        return batch_mfcc_pad,torch.tensor(len_mfcc)



# Data Transformations
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class data_transforms():
    def __init__(self,dataset_path,batch_size):
        self.batch_size = batch_size
        self.dataset_path =dataset_path
        dataset = datasets.ImageFolder(root=dataset_path, transform=transforms.ToTensor())
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.calculate_mean_std()
        self.data_transforms = self.get_data_transforms()


    def calculate_mean_std(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Calculate the mean and standard deviation of a dataset.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 64.

        Returns:
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]: Mean and standard deviation for each channel.
        """
        #for rgb - should be 3 channels -3 d matix   
        #for grey plot - should be 1 d
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_images_count = 0

        for images, _ in self.dataloader:
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images_count += batch_samples

        mean /= total_images_count
        std /= total_images_count
        self.mean = mean.numpy()
        self.std = std.numpy()

    def get_data_transforms(self) :
        """
        Get data transformations for training and validation.

        Args:
            dataset (str, optional): Name of the dataset. Defaults to 'cifar10'.

        Returns:
            Dict[str, transforms.Compose]: Dictionary containing train and val transforms.
        """
        # you can add other transforms here

        data_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
         
        return data_transforms
    def get_transformed_dataloader(self):
        dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.data_transforms)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return dataloader

        
# ----------------------------
# Checkpoint Management
# ----------------------------

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str = "./checkpoint", filename: str = 'checkpoint.pth'):
    """
    Save the training checkpoint.

    Args:
        state (Dict[str, Any]): State dictionary to save.
        is_best (bool): Whether this checkpoint is the best so far.
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str, optional): Checkpoint filename. Defaults to 'checkpoint.pth'.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_best_model( model, optimizer= None, scheduler= None, metric= 'valid_acc',path="./checkpoint/best_model.pth"):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]

# ----------------------------
# Training and Validation Epochs
# ----------------------------

def train_step(train_loader, model, optimizer, criterion, scaler):
    model.train()

    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    train_loss = 0

    for i, data in enumerate(train_loader):

        optimizer.zero_grad()
        x, y, lx, ly = data
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast():
            out, out_lengths = model(x, lx)
            loss = criterion(torch.transpose(out, 0, 1), y, out_lengths, ly)

        train_loss += loss

        batch_bar.set_postfix(
            loss=f"{train_loss / (i + 1):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']}"
        )

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()
        batch_bar.update()

    batch_bar.close()
    train_loss /= len(train_loader)

    return train_loss

def val_step(data_loader, model, criterion):
    model.eval()
    val_loss = 0
    val_dist = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')

    for i, data in enumerate(data_loader):
        x, y, lx, ly = data
        x = x.to(device)
        y = y.to(device)
        with torch.inference_mode():
            out, out_lengths = model(x, lx)
            loss = criterion(torch.transpose(out, 0, 1), y, out_lengths, ly)
        val_loss += loss
        batch_bar.set_postfix(
            loss=f"{val_loss / (i + 1):.4f}"
        )
        batch_bar.update()
    batch_bar.close()
    val_loss /= len(data_loader)
    return val_loss



# Metrics
def cls_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities).
        targets (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy in percentage.
    """
    _, preds = torch.max(outputs, dim=1)
    correct = torch.sum(preds == targets).item()
    accuracy = correct / targets.size(0) * 100
    return accuracy



def predict(data_loader, model, decoder, LABELS):
    model.eval()

    preds = []
    test_dist = 0
    test_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Test')

    for i, data in enumerate(data_loader):
        x, y = data
        x = x.to(device)
        with torch.inference_mode():
            out, out_lengths = model(x, y)
            preds.append(cls_accuracy(out,y))
        test_bar.update()
    return preds

# for hw 3 ---calc the dist

import Levenshtein as Lev
def calculate_levenshtein(h, y, lh, ly, decoder, labels, debug=False):
    if debug:
        pass
        # print(f"\n----- IN LEVENSHTEIN -----\n")
        # Add any other debug statements as you may need
        # you may want to use debug in several places in this function

    beam_results, beam_scores, timesteps, out_lens = decoder.decode(h, seq_lens=lh)

    batch_size = beam_results.size()[0]  

    distance = 0  # Initialize the distance to be 0 initially

    for i in range(batch_size):
        h_sliced = beam_results[i][0][:out_lens[i][0]]
        h_string = ''.join([labels[x] for x in h_sliced])

        l_sliced = y[i][0:ly[i]]
        l_string = ''.join([labels[yy] for yy in l_sliced])

        distance += Lev.distance(l_string, h_string)

    distance /= batch_size

    return distance


def val_step_dist(data_loader, model, criterion, decoder, LABELS):
    model.eval()

    val_loss = 0
    val_dist = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')

    for i, data in enumerate(data_loader):
        x, y, lx, ly = data
        x = x.to(device)
        y = y.to(device)
        with torch.inference_mode():
            out, out_lengths = model(x, lx)
            loss = criterion(torch.transpose(out, 0, 1), y, out_lengths, ly)
            distance = calculate_levenshtein(out, y, out_lengths, ly, decoder, LABELS, debug=False)
        val_loss += loss
        val_dist += distance
        batch_bar.set_postfix(
            loss=f"{val_loss / (i + 1):.4f}",
            distance=f"{val_dist / (i + 1):.4f}"
        )
        batch_bar.update()
    batch_bar.close()
    val_loss /= len(data_loader)
    val_dist /= len(data_loader)

    return val_loss, val_dist

def make_output(h, lh, decoder, LABELS):
    beam_results, _,_, out_seq_len = decoder.decode(h, seq_lens=lh)
    batch_size = beam_results.size()[0]
    preds = []
    for i in range(batch_size): 

        h_sliced = beam_results[i][0][:out_seq_len[i][0]]
        h_string = ''.join([LABELS[x] for x in h_sliced])
        preds.append(h_string)

    return preds


def predict_dist(data_loader, model, decoder, LABELS):
    model.eval()

    preds = []
    test_dist = 0
    test_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Test')

    for i, data in enumerate(data_loader):
        x, lx = data
        x = x.to(device)
        with torch.inference_mode():
            out, out_lengths = model(x, lx)
            preds.append(make_output(out, out_lengths, decoder, LABELS))
        test_bar.update()
    return preds



# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    import torchvision.datasets as datasets
    

    logger = setup_logger("my_logger","./logger_test")
    # Configuration
    config = {
        'dataset_path': '/cifar10',
        'batch_size': 128,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'checkpoint_dir': './checkpoints',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }

    # Data Transforms
    train_data_transforms = data_transforms(os.path.join(config['dataset_path'],"/train")).data_transforms
    val_data_transforms = data_transforms(os.path.join(config['dataset_path'],"/val")).data_transforms
    # Datasets and Dataloaders
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_data_transforms)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform= val_data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    # define a model here
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 10)
    ).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(1, config['num_epochs'] + 1):
        train_metrics = train_step(model, train_loader, criterion, optimizer, config['device'])
        val_metrics = val_step(model, val_loader, criterion, config['device'])

        logger.info(f"Epoch [{epoch}/{config['num_epochs']}]")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")

        # Checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
        }
        save_checkpoint(checkpoint, is_best, config['checkpoint_dir'])


