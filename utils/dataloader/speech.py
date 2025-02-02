import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from scipy.fftpack import dct

class SpeechDataset(Dataset):
    """
    Loads fbank features and transcripts for speech tasks.
    """
    def __init__(self, partition: str, config: dict, tokenizer, isTrainPartition: bool):
        self.config = config
        self.root = config['root']
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN
        self.sos_token = tokenizer.SOS_TOKEN
        self.pad_token = tokenizer.PAD_TOKEN
        self.subset = config['subset']
        self.feat_type = config['feat_type']
        self.num_feats = config['num_feats']
        self.norm = config['norm']

        self.fbank_dir = os.path.join(self.root, self.partition, "fbank")
        self.fbank_files = sorted(os.listdir(self.fbank_dir))
        subset_size = max(config['batch_size'], int(self.subset * len(self.fbank_files)))
        self.fbank_files = self.fbank_files[:subset_size]

        if self.partition != 'test-clean':
            self.text_dir = os.path.join(self.root, self.partition, "text")
            self.text_files = sorted(os.listdir(self.text_dir))[:subset_size]

        self.length = len(self.fbank_files)
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []

        for i in tqdm(range(len(self.fbank_files)), desc=f"Loading data for {self.partition}"):
            feats = np.load(os.path.join(self.fbank_dir, self.fbank_files[i])).T
            if self.feat_type == 'mfcc':
                feats = self.fbank_to_mfcc(feats)
            if self.norm == 'cepstral':
                feats = (feats - np.mean(feats, axis=0)) / (np.std(feats, axis=0) + 1E-8)
            self.feats.append(feats[:, :self.num_feats])
            if self.partition != 'test-clean':
                transcript = np.load(os.path.join(self.text_dir, self.text_files[i])).tolist()
                transcript = "".join(transcript)
                tokenized = self.tokenizer.encode(transcript)
                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])

        if self.partition != 'test-clean':
            assert len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)

        self.global_mean, self.global_std = None, None
        if self.norm == 'global_mvn':
            self.global_mean, self.global_std = self.compute_global_stats()

        # SpecAugment transforms (if enabled) can be applied in collate_fn.
        self.time_mask = None
        self.freq_mask = None
        if config.get("specaug", False):
            import torchaudio.transforms as tat
            if config["specaug_conf"].get("apply_time_mask", False):
                self.time_mask = torch.nn.Sequential(*[tat.TimeMasking(time_mask_param=config["specaug_conf"]["time_mask_width_range"]) for _ in range(config["specaug_conf"]["num_time_mask"])])
            if config["specaug_conf"].get("apply_freq_mask", False):
                self.freq_mask = torch.nn.Sequential(*[tat.FrequencyMasking(freq_mask_param=config["specaug_conf"]["freq_mask_width_range"]) for _ in range(config["specaug_conf"]["num_freq_mask"])])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.feats[idx])
        shifted, golden = None, None
        if self.partition != 'test-clean':
            shifted = torch.LongTensor(self.transcripts_shifted[idx])
            golden = torch.LongTensor(self.transcripts_golden[idx])
        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean.unsqueeze(0)) / (self.global_std.unsqueeze(0) + 1e-8)
        return feat, shifted, golden

    def collate_fn(self, batch):
        batch_feats = [item[0] for item in batch]
        lengths_feats = [len(feat) for feat in batch_feats]
        batch_feats_pad = pad_sequence(batch_feats, batch_first=True)
        if self.partition != 'test-clean':
            batch_transcript = [item[1] for item in batch]
            batch_golden = [item[2] for item in batch]
            lengths_transcript = [len(transcript) for transcript in batch_transcript]
            batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=self.pad_token)
            batch_golden_pad = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)
            return batch_feats_pad, batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_feats), torch.tensor(lengths_transcript)
        else:
            return batch_feats_pad, None, None, torch.tensor(lengths_feats), None

    def fbank_to_mfcc(self, fbank):
        mfcc = dct(fbank, type=2, axis=1, norm='ortho')
        return mfcc

    def compute_global_stats(self):
        all_feats = []
        for file in tqdm(self.fbank_files, desc="Computing global stats"):
            feats = np.load(os.path.join(self.fbank_dir, file))
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, axis=0)
        global_mean = np.mean(all_feats, axis=0)
        global_std = np.std(all_feats, axis=0)
        np.save(os.path.join(self.root, 'global_mean.npy'), global_mean)
        np.save(os.path.join(self.root, 'global_std.npy'), global_std)
        return torch.tensor(global_mean, dtype=torch.float32), torch.tensor(global_std, dtype=torch.float32)
