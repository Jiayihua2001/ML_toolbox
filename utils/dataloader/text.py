import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class TextDataset(Dataset):
    """
    Loads and tokenizes text transcripts.
    """
    def __init__(self, partition: str, config: dict, tokenizer):
        self.root = config['root']
        self.subset = config['subset']
        self.partition = partition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN
        self.sos_token = tokenizer.SOS_TOKEN
        self.pad_token = tokenizer.PAD_TOKEN

        self.text_dir = os.path.join(self.root, self.partition)
        self.text_files = sorted(os.listdir(self.text_dir))
        subset_size = max(config['batch_size'], int(self.subset * len(self.text_files)))
        self.text_files = self.text_files[:subset_size]
        self.length = len(self.text_files)
        self.transcripts_shifted = []
        self.transcripts_golden = []

        for file in tqdm(self.text_files, desc=f"Loading transcripts for {partition}"):
            transcript = np.load(os.path.join(self.text_dir, file)).tolist()
            transcript = " ".join(transcript.split())
            tokenized = self.tokenizer.encode(transcript)
            self.transcripts_shifted.append(np.array([self.sos_token] + tokenized))
            self.transcripts_golden.append(np.array(tokenized + [self.eos_token]))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden

    def collate_fn(self, batch):
        batch_shifted = [item[0] for item in batch]
        batch_golden = [item[1] for item in batch]
        lengths = [len(item) for item in batch_shifted]
        batch_shifted_pad = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
        batch_golden_pad = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)
        return batch_shifted_pad, batch_golden_pad, torch.tensor(lengths)
