import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

class AudioDataset(torch.utils.data.Dataset):
    """
    For training/validation data (with transcripts).
    """
    def __init__(self, data_path: str, partition: str, PHONEMES):
        self.data_path = data_path
        if partition == 'train':
            self.mfcc_dir = os.path.join(self.data_path, f'{partition}-clean-100', 'mfcc')
            self.transcript_dir = os.path.join(self.data_path, f'{partition}-clean-100', 'transcript')
        else:
            self.mfcc_dir = os.path.join(self.data_path, f'{partition}-clean', 'mfcc')
            self.transcript_dir = os.path.join(self.data_path, f'{partition}-clean', 'transcript')
        self.PHONEMES = PHONEMES

        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        transcript_names = sorted(os.listdir(self.transcript_dir))
        assert len(mfcc_names) == len(transcript_names), "Mismatch in mfcc and transcript files."

        self.mfccs = []
        self.transcripts = []
        for mfcc_file, transcript_file in zip(mfcc_names, transcript_names):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_file))
            scaler = StandardScaler()
            mfcc = scaler.fit_transform(mfcc)
            transcript = np.load(os.path.join(self.transcript_dir, transcript_file))
            transcript_idx = [self.PHONEMES.index(p) for p in transcript]
            self.mfccs.append(mfcc)
            self.transcripts.append(np.array(transcript_idx))
        self.length = len(mfcc_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.mfccs[idx], self.transcripts[idx]

    @staticmethod
    def pad_and_pack(batch):
        batch_mfcc = [torch.tensor(x) for x, _ in batch]
        len_mfcc = [len(x) for x in batch_mfcc]
        batch_transcript = [torch.tensor(y) for _, y in batch]
        len_transcript = [len(y) for y in batch_transcript]
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(len_mfcc), torch.tensor(len_transcript)

class AudioTestDataset(torch.utils.data.Dataset):
    """
    For test data (without transcripts).
    """
    def __init__(self, data_path: str, PHONEMES):
        self.data_path = data_path
        self.mfcc_dir = os.path.join(self.data_path, 'test-clean', 'mfcc')
        self.PHONEMES = PHONEMES
        mfcc_names = sorted(os.listdir(self.mfcc_dir))
        self.mfccs = []
        for mfcc_file in mfcc_names:
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_file))
            scaler = StandardScaler()
            mfcc = scaler.fit_transform(mfcc)
            self.mfccs.append(mfcc)
        self.length = len(mfcc_names)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.mfccs[idx]
    
    @staticmethod
    def pad_and_pack(batch):
        batch_mfcc = [torch.tensor(x) for x in batch]
        len_mfcc = [len(x) for x in batch_mfcc]
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        return batch_mfcc_pad, torch.tensor(len_mfcc)
