import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from CNNLSTM import *
from ml_utils import *
import torchaudio.transforms as tat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import gc

import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder

import warnings
warnings.filterwarnings('ignore')
import yaml
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())


PHONEMES = CMUdict[:-2]
LABELS = ARPAbet[:-2]


def load_raw(path, name):
    return np.load(os.path.join(path, name))

def phonemes2integers(transcript, phonemes):
    # Return the position of the word in labels
    index = []
    for word in transcript:
        index.append(phonemes.index(word))
    return np.array(index)

class AudioDataset(Dataset):

    def __init__(self, data_path, partition, limit=-1):

        self.data_path = data_path

        if partition == 'train':
            self.mfcc_dir = os.path.join(self.data_path, '{}-clean-100/'.format(partition), 'mfcc/')
            self.transcript_dir = os.path.join(self.data_path, '{}-clean-100/'.format(partition), 'transcript/')
        else:
            self.mfcc_dir = os.path.join(self.data_path, '{}-clean/'.format(partition), 'mfcc/')
            self.transcript_dir = os.path.join(self.data_path, '{}-clean/'.format(partition), 'transcript/')

        self.PHONEMES = PHONEMES

        mfcc_names = os.listdir(self.mfcc_dir)
        transcript_names = os.listdir(self.transcript_dir)

        assert len(mfcc_names) == len(transcript_names)

        self.mfccs, self.transcripts = [], []

        for i in range(0, len(mfcc_names)):
            #   Load a single mfcc
            mfcc = load_raw(self.mfcc_dir, mfcc_names[i])
            scaler = StandardScaler()
            mfcc = scaler.fit_transform(mfcc)
            transcript = load_raw(self.transcript_dir, transcript_names[i])[1:limit]
            transcript = phonemes2integers(transcript, self.PHONEMES)
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        self.length = len(mfcc_names)
        
    def __len__(self):

        return self.length

    def __getitem__(self, ind):

        mfcc = self.mfccs[ind]
        transcript = self.transcripts[ind]

        return mfcc, transcript

    def collate_fn(batch):
        # batch of input mfcc coefficients
        batch_mfcc = [torch.tensor(x) for x, y in batch]
        lengths_mfcc = [len(xx) for xx in batch_mfcc]

        # batch of output phonemes
        batch_transcript = [torch.tensor(y) for x, y in batch]
        lengths_transcript = [len(yy) for yy in batch_transcript]

        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)

        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


class AudioDatasetTest(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.mfcc_dir = os.path.join(self.data_path, 'test-clean/mfcc/')
        self.PHONEMES = PHONEMES
        mfcc_names = os.listdir(self.mfcc_dir)
        mfcc_names.sort()
        self.mfccs = []

        for i in range(0, len(mfcc_names)):
            #   Load a single mfcc
            mfcc = load_raw(self.mfcc_dir, mfcc_names[i])
            scaler = StandardScaler()
            mfcc = scaler.fit_transform(mfcc)
            self.mfccs.append(mfcc)

        self.length = len(mfcc_names)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        mfcc = self.mfccs[ind]

        return mfcc

    def collate_fn(batch):
        # batch of input mfcc coefficients

        batch_mfcc = [torch.tensor(x) for x in batch]
        lengths_mfcc = [len(xx) for xx in batch_mfcc]

        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)

        return batch_mfcc_pad, torch.tensor(lengths_mfcc)


def load_checkpoint(model, optimizer,scheduler, checkpoint_path='./checkpoint.pth'):
    """
    Loads the model and optimizer state from a checkpoint.

    Args:
        model (nn.Module): The model instance to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer instance to load the state into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        tuple: A tuple containing the validation distance and the epoch number.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    val_dist = checkpoint['val_dist']
    epoch = checkpoint['epoch']
    
    print(f"Checkpoint loaded: Epoch {epoch}, Validation Distance {val_dist}")
    return val_dist, epoch


def main():

    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data_path = '/global/cfs/cdirs/m3578/jiayihua/11785/hw3/11785-f24-hw3p2'
    train_data = AudioDataset(data_path, partition='train')
    val_data = AudioDataset(data_path, partition='dev')
    test_data = AudioDatasetTest(data_path)
    # Set random seeds for reproducibility
    seed = 63
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Do NOT forget to pass in the collate function as parameter while creating the dataloader
    train_loader = DataLoader(train_data, config['batch_size'], shuffle=False, collate_fn=AudioDataset.collate_fn,)
    val_loader = DataLoader(val_data, config['batch_size'], shuffle=False, collate_fn=AudioDataset.collate_fn)
    test_loader = DataLoader(test_data, config['batch_size'], shuffle=False, collate_fn=AudioDatasetTest.collate_fn)

    print("Batch size: ", config['batch_size'])
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

    OUT_SIZE = len(LABELS)

    torch.cuda.empty_cache()

    model = CNN_BiLSTM(hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                       dropout=config['lstm_dropout'], out_size=OUT_SIZE, in_channels=28,config=config).to(device)

    criterion = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0)
    decoder = CTCBeamDecoder(
        LABELS,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=config["beam_width"],
        num_processes=4,
        blank_id=0,
        log_probs_input=True
    )

    decoder_pred = CTCBeamDecoder(
        LABELS,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=True
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=3,
                                                           verbose=True)
    
    # Mixed Precision, if you need it
    scaler = torch.cuda.amp.GradScaler()

    import wandb
    wandb.login(key="bc022b99e5a39b97fc6ae8c641ab328e9f52d2e6") 
    run = wandb.init(
        name=config['run_name'],  ## Wandb creates random run names if you skip this field
        reinit=True,  ### Almlows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project="hw3_p2_ablation",  ### Project should be created in your wandb account
        config=config  ### Wandb Config for your run
    )

    torch.cuda.empty_cache()
    gc.collect()

    best_val_dist = 300
    val_dist = 0
    # val_dist, epoch = load_checkpoint(model, optimizer,scheduler)
    for epoch in range(config["epochs"]):

        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_loss = train_step(train_loader, model, optimizer, criterion, scaler)

        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            config['epochs'],
            train_loss,
            curr_lr))
        if (epoch + 1) % 10 == 0:
            val_loss, val_dist = evaluate_dist(val_loader, model, criterion, decoder, LABELS)
            print("Val Loss {:.04f}".format(val_loss))
            print("Val Distance {:.04f}".format(val_dist))
        else:
            val_loss = evaluate(val_loader, model, criterion)
            print("Val Loss {:.04f}".format(val_loss))
        scheduler.step(val_loss)
        # Use the below code to save models
        if val_dist < best_val_dist:
            # path = os.path.join(root_path, model_directory, 'checkpoint' + '.pth')
            print("Saving model")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_dist': val_dist,
                        'epoch': epoch}, './checkpoint.pth')
            best_val_dist = val_dist
            # wandb.save('checkpoint.pth')

        # You may want to log some hyperparameters and results on wandb
        wandb.log(
            {"train_loss": train_loss, 'validation_loss': val_loss, "learning_Rate": curr_lr, 'distance': val_dist})

    run.finish()

    torch.cuda.empty_cache()
    predictions = predict(test_loader, model, decoder_pred, LABELS)
    predictions = sum(predictions, [])
    df = pd.read_csv('random_submission.csv')
    df.label = predictions

    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()




