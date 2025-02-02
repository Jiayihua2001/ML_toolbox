import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio.transforms as tat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import gc
import pandas as pd
from tqdm import tqdm
import os
import datetime
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder


# utilities and models
from CNNLSTM import *
from ml_utils import *

import warnings
import yaml
warnings.filterwarnings('ignore')
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
# define audio dataset
class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, partition):

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
            transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]))[1:-1]
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


class AudioDataset_test(torch.utils.data.Dataset):

    def __init__(self, data_path):

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



def main():

    # load config file
    # with open(r'config.yaml') as file:
    #     config = yaml.load(file, Loader=yaml.FullLoader)
    logger  = setup_logger("hw3_logger","hw3p2_ablation.log")
    run = wandb.init()
    
    # Access hyperparameters
    config = wandb.config
    logger.info(config)
    data_path = config["data_path"]
    train_data = AudioDataset(data_path, partition='train')
    val_data = AudioDataset(data_path, partition='dev')
    test_data = AudioDataset_test(data_path)

    #set random seed
    seed = 2001
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load the data

    train_loader = DataLoader(train_data, config['batch_size'], shuffle=False, collate_fn=AudioDataset.pad_and_pack)
    val_loader = DataLoader(val_data, config['batch_size'], shuffle=False, collate_fn=AudioDataset.pad_and_pack)
    test_loader = DataLoader(test_data, config['batch_size'], shuffle=False, collate_fn=AudioDataset_test.pad_and_pack)


    logger.info("Batch size: %d", config['batch_size'])
    logger.info("Train dataset samples = %d, batches = %d", train_data.__len__(), len(train_loader))
    logger.info("Val dataset samples = %d, batches = %d", val_data.__len__(), len(val_loader))
    logger.info("Test dataset samples = %d, batches = %d", test_data.__len__(), len(test_loader))

    #load the model 
    OUT_SIZE = len(LABELS)
    torch.cuda.empty_cache()
    model = CNN_BiLSTM(hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
                       dropout=config['lstm_dropout'], out_size=OUT_SIZE, in_channels=28,config = config).to(device)
    
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

    # run = wandb.init(
    #     name=config['run_name'],  ## Wandb creates random run names if you skip this field
    #     reinit=True,  ### Almlows reinitalizing runs when you re-run this cell
    #     # run_id = ### Insert specific run id here if you want to resume a previous run
    #     # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    #     project="hw3p2",  ### Project should be created in your wandb account
    #     config=config  ### Wandb Config for your run
    # )


    torch.cuda.empty_cache()
    gc.collect()

    #train_loop
    best_val_dist = 100
    val_dist = 0

    # load model if needed
    if config["load"]:
        model, optimizer, scheduler, epoch, val_dist = load_best_model( model, optimizer, scheduler,metric= 'val_dist')
    for epoch in range(config["epochs"]):

        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_loss = train_step(train_loader, model, optimizer, criterion, scaler)

        logger.info("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            config['epochs'],
            train_loss,
            curr_lr))

        val_loss, val_dist = val_step_dist(val_loader, model, criterion, decoder, LABELS)
        logger.info("Val Loss {:.04f}".format(val_loss))
        logger.info("Val Distance {:.04f}".format(val_dist))

        scheduler.step(val_loss)
        # Use the below code to save models
        if val_dist < best_val_dist:
            # path = os.path.join(root_path, model_directory, 'checkpoint' + '.pth')
            logger.info("Saving model")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_dist': val_dist,
                        'epoch': epoch}, './checkpoint.pth')
            best_val_dist = val_dist

            # wandb.save('checkpoint.pth')

        # wandb log
        wandb.log(
            {"train_loss": train_loss, 'validation_loss': val_loss, "learning_Rate": curr_lr, 'distance': val_dist})

    run.finish()

    torch.cuda.empty_cache()

    predictions = predict_dist(test_loader, model, decoder_pred, LABELS)
    predictions = sum(predictions, [])

    df = pd.read_csv('random_submission.csv')
    df.label = predictions
    df.to_csv('submission.csv', index=False)

    

if __name__=="__main__":
    import wandb
    with open(r'sweep_config.yaml') as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    wandb.login(key="bc022b99e5a39b97fc6ae8c641ab328e9f52d2e6") 
    sweep_id = wandb.sweep(sweep_config, project="hw3_p2_ablation")
    # Run the sweep
    wandb.agent(sweep_id, function=main, count=30)
    






