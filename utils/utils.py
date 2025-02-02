import torch
import torch.nn as nn
import torchaudio.functional as aF
import torchaudio.transforms as tat
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, Dataset, DataLoader
import gc
import os
from transformers import AutoTokenizer
import yaml
import math
from typing import Literal, List, Optional, Any, Dict, Tuple
import random
from torchinfo import summary
import wandb
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct
import seaborn as sns
import matplotlib.pyplot as plt

''' Imports for decoding and distance calculation. '''
import json
import warnings
import shutil



class CharTokenizer():
    ''' A wrapper around character tokenization to have a consistent interface with other tokeization strategies'''

    def __init__(self):
        # Define special tokens for end-of-sequence, padding, and unknown characters
        self.eos_token = "<|endoftext|>"  # Same as EOS_TOKEN
        self.pad_token = "<|padding|>"
        self.unk_token = "<|unknown|>"

        # Initialize vocabulary with uppercase alphabet characters and space
        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")

        # Create vocabulary mapping
        self.vocab = {
            self.eos_token: 0,
            self.pad_token: 1,  # Same ID as EOS_TOKEN
            self.unk_token: 2,
        }

        for idx, char in enumerate(characters, start=3):
            self.vocab[char] = idx

        # Create an inverse mapping from IDs to characters for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Define token IDs for special tokens for easy access
        self.eos_token_id = self.vocab[self.eos_token]
        self.bos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

        self.vocab_size = len(self.vocab)

    def tokenize(self, data:str) -> List[str]:
        # Split input data into a list of characters for tokenization
        return [char for char in data]

    def encode(self, data:str, return_tensors:Optional[Literal['pt']]=None) -> List[int]:
        # Encode each character in data to its integer ID, using unk_token if character is not in vocab
        e = [self.vocab.get(char.upper(), self.unk_token) for char in data]
        # If specified, convert to PyTorch tensor format
        if return_tensors == 'pt':
            return torch.tensor(e).unsqueeze(0)
        return e

    def decode(self, data:List[int]) -> str:
        # Decode list of token IDs back to string by mapping each ID to its character
        try:
            return ''.join([self.inv_vocab.get(j) for j in data])
        except:
            # Handle decoding error by converting data to list, if it's a tensor
            data = data.cpu().tolist()
            return ''.join([self.inv_vocab.get(j) for j in data])



class GTokenizer():

    def __init__(self, token_type: Literal['1k', '10k', '50k', 'char']='char', logger=None):

        self.token_type = token_type
        self.vocab, self.inv_vocab = None, None

        # Where are these files?
        if token_type == '1k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_1k", use_fast=False)
        elif token_type == '10k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_10k", use_fast=False)
        elif token_type == '20k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_20k", use_fast=False)
        elif token_type == '50k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_50k", use_fast=False)
        elif token_type  == '100k':
            self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_100k", use_fast=False)
        elif token_type == 'char':
            self.tokenizer = CharTokenizer()

        self.EOS_TOKEN  = self.tokenizer.eos_token_id
        self.SOS_TOKEN  = self.tokenizer.bos_token_id
        self.PAD_TOKEN  = self.tokenizer.convert_tokens_to_ids('<|padding|>') if self.token_type != "char" else self.tokenizer.pad_token_id
        self.UNK_TOKEN  = self.tokenizer.unk_token_id
        self.VOCAB_SIZE = self.tokenizer.vocab_size

        # Test tokenization methods to ensure everything is working correctly
        test_text = "HI DEEP LEARNERS"
        test_tok  = self.tokenize(test_text)
        test_enc  = self.encode(test_text)
        test_dec  = self.decode(test_enc)

        print(f"[Tokenizer Loaded]: {token_type}")
        print(f"\tEOS_TOKEN:  {self.EOS_TOKEN}")
        print(f"\tSOS_TOKEN:  {self.SOS_TOKEN}")
        print(f"\tPAD_TOKEN:  {self.PAD_TOKEN}")
        print(f"\tUNK_TOKEN:  {self.UNK_TOKEN}")
        print(f"\tVOCAB_SIZE: {self.VOCAB_SIZE}")
        print("Examples:")
        print(f"\t[DECODE EOS, SOS, PAD, UNK] : {self.decode([self.EOS_TOKEN, self.SOS_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN])}")
        print(f"\t[Tokenize HI DEEP LEARNERS] : {test_tok}")
        print(f"\t[Encode   HI DEEP LEARNERS] : {test_enc}")
        print(f"\t[Decode   HI DEEP LEARNERS] : {test_dec}")



    def tokenize(self, data:str) -> List[str]:
        return self.tokenizer.tokenize(data)

    def encode(self, data:str, return_tensors=False) -> List[int]:
        if return_tensors:
            return self.tokenizer.encode(data, return_tensors='pt')
        return self.tokenizer.encode(data)

    def decode(self, data:List[int]) -> str:
        return self.tokenizer.decode(data)



#load dataset

class SpeechDataset(Dataset):

    def __init__(self,
                 partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
                 config:dict,
                 tokenizer:GTokenizer,
                 isTrainPartition:bool
                 ):
        """
        Initialize the SpeechDataset.

        Args:
            partition (str): Partition name
            config (dict): Configuration dictionary for dataset settings.
            tokenizer (GTokenizer): Tokenizer class for encoding and decoding text data.
            isTrainPartition (bool): Flag indicating if this partition is for training.
        """

        # general: Get config values
        self.config           = config
        self.root             = self.config['root']
        self.partition        = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN
        self.sos_token = tokenizer.SOS_TOKEN
        self.pad_token = tokenizer.PAD_TOKEN
        self.subset    = self.config['subset']
        self.feat_type = self.config['feat_type']
        self.num_feats = self.config['num_feats']
        self.norm      = self.config['norm'] 

        # paths | files
        self.fbank_dir   = os.path.join(self.root, self.partition, "fbank")
        self.fbank_files = sorted(os.listdir(self.fbank_dir))
        subset           = max(config['batch_size'], int(self.subset * len(self.fbank_files)))
        self.fbank_files = sorted(os.listdir(self.fbank_dir))[:subset]

        if self.partition != 'test-clean':
          self.text_dir    = os.path.join(self.root, self.partition, "text")
          self.text_files  = sorted(os.listdir(self.text_dir))
          self.text_files  = sorted(os.listdir(self.text_dir))[:subset]
          assert len(self.fbank_files) == len(self.text_files), "Number of fbank files and text files must be the same"

        self.length = len(self.fbank_files)
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []


        for i in tqdm(range(len(self.fbank_files)), desc=f"Loading fbank and transcript data for {self.partition}"):
            # load features
            feats = np.load(os.path.join(self.fbank_dir, self.fbank_files[i])).T
            if self.feat_type == 'mfcc':
                feats = self.fbank_to_mfcc(feats)

            if self.config['norm'] == 'cepstral':
                feats = (feats - np.mean(feats, axis=0)) / (np.std(feats, axis=0) + 1E-8)

            self.feats.append(feats[:, :self.num_feats])

            # load and encode transcripts
            # Why do we have two different types of targets?
            # How do we want our decoder to know the start of sequence <SOS> and end of sequence <EOS>?

            if self.partition != 'test-clean':
              # Note: You dont have access to transcripts in dev_clean
              transcript = np.load(os.path.join(self.text_dir, self.text_files[i])).tolist()
              transcript = "".join(transcript)
              #Invoke our tokenizer to tokenize the string
              tokenized  = self.tokenizer.encode(transcript)
              ## TODO: How will you use tokenized?
              self.transcripts_shifted.append([self.sos_token] + tokenized)
              self.transcripts_golden.append(tokenized + [self.eos_token])

        if self.partition != 'test-clean':
          assert len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)

        # precompute global stats for global mean and variance normalization
        self.global_mean, self.global_std = None, None
        if self.config['norm'] == 'global_mvn':
            self.global_mean, self.global_std = self.compute_global_stats()
          
        # Torch Audio Transforms
        # time masking
        self.time_mask = torch.nn.Sequential(
            *[tat.TimeMasking(time_mask_param=self.config['specaug_conf']['time_mask_width_range']) for _ in range(self.config['specaug_conf']['num_time_mask'])]
        ) if self.config['specaug_conf']['apply_time_mask'] else None
        
        # frequency masking
        self.freq_mask = torch.nn.Sequential(
            *[tat.FrequencyMasking(freq_mask_param=self.config['specaug_conf']['freq_mask_width_range']) for _ in range(self.config['specaug_conf']['num_freq_mask'])]
        ) if self.config['specaug_conf']['apply_freq_mask'] else None


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feat = torch.FloatTensor(self.feats[idx])
        shifted_transcript, golden_transcript = None, None
        if self.partition != 'test-clean':
          shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
          golden_transcript = torch.LongTensor(self.transcripts_golden[idx])
        # Apply global mean and variance normalization if enabled
        if self.global_mean is not None and self.global_std is not None:
            feat = (feat - self.global_mean.unsqueeze(0)) / (self.global_std.unsqueeze(0) + 1e-8)
        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch):
        # @NOTE: batch corresponds to output from __getitem__ for a minibatch

        '''
        1.  Extract the features and labels from 'batch'.
        2.  We will additionally need to pad both features and labels,
            look at PyTorch's documentation for pad_sequence.
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lengths of features, and lengths of labels.

        '''
        # Prepare batch (features)
        batch_feats = [item[0] for item in batch] # TODO: position of feats do you return from get_item + transpose B x T x F
        lengths_feats = [len(feat) for feat in batch_feats] # Lengths of each T x F sequence
        batch_feats_pad = pad_sequence(batch_feats, batch_first=True) # Pad sequence
        # batch_feats_pad = batch_feats_pad.transpose(1, 2)  # Transpose to B x T x F
        
        if self.partition != 'test-clean':
            batch_transcript   = [item[1] for item in batch] # B x T
            batch_golden       = [item[2] for item in batch] # B x T
            lengths_transcript = [len(transcript) for transcript in batch_transcript] # Lengths of each T
            batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value = self.tokenizer.PAD_TOKEN) # Pad sequence
            batch_golden_pad     = pad_sequence(batch_golden, batch_first=True, padding_value = self.tokenizer.PAD_TOKEN) # Pad sequence

        # TODO: do specaugment transforms
        if self.config["specaug"] and self.isTrainPartition:
            # transpose back to F x T to apply transforms
            batch_feats_pad = batch_feats_pad.transpose(1, 2)

            # shape should be B x num_feats x T
            assert batch_feats_pad.shape[1] == self.num_feats
            
            # Apply frequency mask
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for i in range(self.config["specaug_conf"]["num_freq_mask"]):
                    batch_feats_pad = self.freq_mask[i](batch_feats_pad)
            
            # time mask
            if self.config["specaug_conf"]["apply_time_mask"]:
                for i in range(self.config["specaug_conf"]["num_time_mask"]):
                    batch_feats_pad = self.time_mask[i](batch_feats_pad)

            # transpose back to T x F
            batch_feats_pad = batch_feats_pad.transpose(1, 2)
            # shape should be B x T x num_feats
            assert batch_feats_pad.shape[2] == self.num_feats

        # Return the following values:
        # padded features, padded shifted labels, padded golden labels, actual length of features, actual length of the shifted label
        if self.partition != 'test-clean':
            return batch_feats_pad, batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_feats), torch.tensor(lengths_transcript)
        else:
            return batch_feats_pad, None, None, torch.tensor(lengths_feats), None

    def fbank_to_mfcc(self, fbank):
        # Helper function that applies the dct to the filterbank features to concert them to mfccs
        mfcc = dct(fbank, type=2, axis=1, norm='ortho')
        return mfcc

    #Will be discussed in bootcamp
    def compute_global_stats(self):
        # Compute global mean and variance of the dataset
        all_feats = []
        for file in tqdm(self.text_files, desc="Computing global stats"):
            feats = np.load(os.path.join(self.text_dir, file))
            all_feats.append(feats)
        
        all_feats = np.concatenate(all_feats, axis=0)
        global_mean = np.mean(all_feats, axis=0)
        global_var = np.var(all_feats, axis=0)
        
        # Save the computed stats
        np.save(os.path.join(self.root, 'global_mean.npy'), global_mean)
        np.save(os.path.join(self.root, 'global_var.npy'), global_var)
        
        return global_mean, global_var





class TextDataset(Dataset):
    def __init__(self, partition: str, config:dict, tokenizer: GTokenizer):
        """
        Initializes the TextDataset class, which loads and tokenizes transcript files.

        Args:
            partition (str): Subdirectory under root that specifies the data partition (e.g., 'train', 'test').
            config (dict): Configuration dictionary for dataset settings.
            tokenizer (GTokenizer): Tokenizer instance for encoding transcripts into token sequences.
        """

        # General attributes
        self.root      = config['root']
        self.subset    = config['subset']
        self.partition = partition
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.EOS_TOKEN  # End of sequence token
        self.sos_token = tokenizer.SOS_TOKEN  # Start of sequence token
        self.pad_token = tokenizer.PAD_TOKEN  # Padding token

        # Paths and files
        self.text_dir = os.path.join(self.root, self.partition)  # Directory containing transcript files
        self.text_files = sorted(os.listdir(self.text_dir))  # Sorted list of transcript files

        # Limit to subset of files if specified
        subset = max(config['batch_size'], int(self.subset * len(self.text_files)))
        self.text_files = self.text_files[:subset]
        self.length = len(self.text_files)

        # Storage for encoded transcripts
        self.transcripts_shifted, self.transcripts_golden = [], []

        # Load and encode transcripts
        for file in tqdm(self.text_files, desc=f"Loading transcript data for {partition}"):
            transcript = np.load(os.path.join(self.text_dir, file)).tolist()
            transcript = " ".join(transcript.split())  # Process text
            tokenized = self.tokenizer.encode(transcript)  # Tokenize transcript
            # Store shifted and golden versions of transcripts
            self.transcripts_shifted.append(np.array([self.eos_token] + tokenized))
            self.transcripts_golden.append(np.array(tokenized + [self.eos_token]))

    def __len__(self) -> int:
        """Returns the total number of transcripts in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Retrieves the shifted and golden version of the transcript at the specified index.

        Args:
            idx (int): Index of the transcript to retrieve.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: Shifted and golden version of the transcript.
        """
        shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
        golden_transcript = torch.LongTensor(self.transcripts_golden[idx])
        return shifted_transcript, golden_transcript

    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a batch of transcripts for model input, applying padding as needed.

        Args:
            batch (List[Tuple[torch.LongTensor, torch.LongTensor]]): Batch of (shifted, golden) transcripts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Padded shifted transcripts (batch_transcript_pad).
                - Padded golden transcripts (batch_golden_pad).
                - Lengths of shifted transcripts.
        """

        # Separate shifted and golden transcripts from batch
        batch_transcript = [i[0] for i in batch]  # B x T
        batch_golden = [i[1] for i in batch]  # B x T
        lengths_transcript = [len(i) for i in batch_transcript]

        # Pad sequences
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=self.pad_token)
        batch_golden_pad = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        # Return padded sequences and lengths
        return batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_transcript)

#MASKs

def PadMask(padded_input, input_lengths=None, pad_idx=None):
    """ Create a mask to identify non-padding positions.

    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: Optional, the actual lengths of each sequence before padding, shape (N,).
        pad_idx: Optional, the index used for padding tokens.

    Returns:
        A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
    """

    # If input is a 2D tensor (N, T), add an extra dimension
    if padded_input.dim() == 2:
        padded_input = padded_input.unsqueeze(-1)

    # TODO: Initialize the mask variable here. What type should it be and how should it be initialized?

    if input_lengths is not None:
        # Use the provided input_lengths to create the mask.
        N, T, _ = padded_input.shape
        mask = torch.ones((N, T), dtype=torch.bool)
        for i in range(N):
            # TODO: Set non-padding positions to False based on input_lengths
            mask[i, :input_lengths[i]] = False
    else:
        # TODO: Infer the mask from the padding index.
        mask = (padded_input.squeeze(-1) == pad_idx)  # Shape (N, T)
    mask = mask.to(padded_input.device)
    return mask


def CausalMask(input_tensor):
    """
    Create an attention mask for causal self-attention based on input lengths.

    Args:
        input_tensor (torch.Tensor): The input tensor of shape (N, T, *).

    Returns:
        attn_mask (torch.Tensor): The causal self-attention mask of shape (T, T)
    """
    T = input_tensor.shape[1]  # Sequence length
    attn_mask = ~torch.tril(torch.ones((T, T), dtype = torch.bool)).to(input_tensor.device)
    """
    # TODO: Initialize attn_mask as a tensor of zeros with the right shape.
    attn_mask = NotImplemented # Shape (T, T)

    # TODO: Create a lower triangular matrix to form the causal mask.
    causal_mask = NotImplemented  # Lower triangular matrix

    # TODO: Combine the initial mask with the causal mask.
    attn_mask = attn_mask | causal_mask
    """
    return attn_mask



# positional encoding

class PositionalEncoding(torch.nn.Module):
    ''' Position Encoding from Attention Is All You Need Paper '''

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Initialize a tensor to hold the positional encodings
        pe          = torch.zeros(max_len, d_model)

        # Create a tensor representing the positions (0 to max_len-1)
        position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions
        # This term creates a series of values that decrease geometrically, used to generate varying frequencies for positional encodings
        div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
      return x + self.pe[:, :x.size(1)]


#speech embedding


# 2-Layer BiLSTM
class BiLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(BiLSTMEmbedding, self).__init__()
        self.bilstm = nn.LSTM(
                input_dim, output_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
        )

    def forward(self, x,  x_len):
        """
        Args:
            x.    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # BiLSTM expects (batch_size, seq_len, input_dim)
        # Pack the padded sequence to avoid computing over padded tokens
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        # Pass through the BiLSTM
        packed_output, _ = self.bilstm(packed_input)
        # Unpack the sequence to restore the original padded shape
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output

### DO NOT MODIFY

class Conv2DSubsampling(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, time_stride=2, feature_stride=2):
        """
        Conv2dSubsampling module that can selectively apply downsampling
        for time and feature dimensions, and calculate cumulative downsampling factor.
        Args:
            time_stride (int): Stride along the time dimension for downsampling.
            feature_stride (int): Stride along the feature dimension for downsampling.
        """
        super(Conv2DSubsampling, self).__init__()

        # decompose to get effective stride across two layers
        tstride1, tstride2 = self.closest_factors(time_stride)
        fstride1, fstride2 = self.closest_factors(feature_stride)

        self.feature_stride = feature_stride
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, kernel_size=3, stride=(tstride1, fstride1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=(tstride2, fstride2)),
            torch.nn.ReLU(),
        )
        self.time_downsampling_factor = tstride1 * tstride2
        # Calculate output dimension for the linear layer
        conv_out_dim = (input_dim - (3 - 1) - 1) // fstride1 + 1
        conv_out_dim = (conv_out_dim - (3 - 1) - 1) // fstride2 + 1
        conv_out_dim = output_dim * conv_out_dim
        self.out = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, output_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            x_mask (torch.Tensor): Optional mask for the input tensor.

        Returns:
            torch.Tensor: Downsampled output of shape (batch_size, new_seq_len, output_dim).
        """
        x = x.unsqueeze(1)  # Add a channel dimension for Conv2D
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x

    def closest_factors(self, n):
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        # Return the factor pair
        return max(factor, n // factor), min(factor, n // factor)



class SpeechEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, time_stride, feature_stride, dropout):
        super(SpeechEmbedding, self).__init__()

        self.cnn = Conv2DSubsampling(input_dim, output_dim, dropout=dropout, time_stride=time_stride, feature_stride=feature_stride)
        self.blstm = BiLSTMEmbedding(output_dim, output_dim, dropout)
        self.time_downsampling_factor = self.cnn.time_downsampling_factor

    def forward(self, x, x_len, use_blstm: bool = False):
        """
        Args:
            x    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len // stride, output_dim)
        """
        # First, apply Conv2D subsampling
        x = self.cnn(x)
        # Adjust sequence length based on downsampling factor
        x_len = torch.ceil(x_len.float() / self.time_downsampling_factor).int()
        x_len = x_len.clamp(max=x.size(1))

        # Apply BiLSTM if requested
        if use_blstm:
            x = self.blstm(x, x_len)

        return x, x_len



# transformer encoder

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):

        super(EncoderLayer, self).__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout = dropout, batch_first=True)
        self.ffn1 = nn.Sequential(nn.Linear(d_model, d_ff), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))# Hint: Linear layer - GELU - dropout - Linear layer
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask):
        # Step 1: Apply pre-normalization
        ''' TODO '''
        x = self.pre_norm(x)

        # Step 2: Self-attention with with dropout, and with residual connection
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=pad_mask)
        x = x + self.dropout_1(attn_output)
        # Step 3: Apply normalization
        ''' TODO '''
        x = self.norm1(x)
        # Step 4: Apply Feed-Forward Network (FFN) with dropout, and residual connection
        ''' TODO '''
        ffn_output = self.ffn1(x)
        x =  x + self.dropout_2(ffn_output)
        # Step 5: Apply normalization after FFN
        ''' TODO '''
        x = self.norm2(x)

        return x, pad_mask


class Encoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 max_len,
                 target_vocab_size,
                 dropout=0.1):

        super(Encoder, self).__init__()

        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.after_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, target_vocab_size)


    def forward(self, x, x_len):

        # Step 1: Create padding mask for inputs
        ''' TODO '''
        pad_mask = PadMask(x, input_lengths = x_len)
        # Step 2: Apply positional encoding
        ''' TODO '''
        x = self.pos_encoding(x)
        # Step 3: Apply dropout
        ''' TODO '''
        x = self.dropout(x)
        # Step 4: Add the residual connection (before passing through layers)
        x_residual = x
        ''' TODO '''
        # Step 5: Pass through all encoder layers
        for enc_layer in self.enc_layers:
            x_new, _ = enc_layer(x, pad_mask)
            x = x_new + x_residual
            x_residual = x
        # Step 6: Apply final normalization
        ''' TODO '''
        x = self.after_norm(x)

        # Step 7: Pass a branch through the CTC head
        ''' TODO '''
        x_ctc = self.ctc_head(x)

        return x, x_len, x_ctc.log_softmax(2).permute(1, 0, 2)


# transformer decoder

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # @TODO: fill in the blanks appropriately (given the modules above)
        self.mha1       = nn.MultiheadAttention(d_model, num_heads, dropout = dropout, batch_first=True) #self attention
        self.mha2       = nn.MultiheadAttention(d_model, num_heads, dropout = dropout, batch_first=True) # cross attention
        self.ffn        = nn.Sequential(nn.Linear(d_model, d_ff), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.identity   = nn.Identity()
        self.pre_norm   = nn.LayerNorm(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
       

    def forward(self, padded_targets, enc_output, pad_mask_enc, pad_mask_dec, slf_attn_mask):
        # pad_mask_dec can be used for absolute positional embedding if its elements are float?

        padded_targets = self.pre_norm(padded_targets)

        # Step 1: Self Attention
        # (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
        mha1_output, mha1_attn_weights = self.mha1(padded_targets, padded_targets, padded_targets, attn_mask=slf_attn_mask, key_padding_mask=pad_mask_dec)
        # (2) add dropout
        mha1_output = self.dropout_1(mha1_output)
        # (3) residual connections
        padded_targets = padded_targets + mha1_output
        # (4) layer normalization
        padded_targets = self.layernorm1(padded_targets)

        # Step 2: Cross Attention
        if enc_output is not None:
            # (1) pass through the Multi-Head Attention (Hint: you need to store weights here as part of the return value)
            # think about if key, value, query here are the same as the previous one?
            mha2_output, mha2_attn_weights = self.mha2(padded_targets, enc_output, enc_output, key_padding_mask=pad_mask_enc)
            # (2) add dropout
            mha2_output = self.dropout_2(mha2_output)
            # (3) residual connections
            padded_targets = padded_targets + mha2_output
            # (4) layer normalization
            padded_targets = self.layernorm2(padded_targets)
        else:
            mha2_output = self.identity(padded_targets)
            mha2_attn_weights = torch.zeros_like(mha1_attn_weights)

        # Step 3: Feed Forward Network
        # (1) pass through the FFN
        ffn_output = self.ffn(padded_targets)
        # (2) add dropout
        ffn_output = self.dropout_3(ffn_output)
        # (3) residual connections
        padded_targets = padded_targets + ffn_output
        # (4) layer normalization
        padded_targets = self.layernorm3(padded_targets)

        return padded_targets, mha1_attn_weights, mha2_attn_weights

class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff, dropout,
                 max_len,
                 target_vocab_size):

        super().__init__()

        self.max_len        = max_len
        self.num_layers     = num_layers
        self.num_heads      = num_heads

        # use torch.nn.ModuleList() with list comprehension looping through num_layers
        # @NOTE: think about what stays constant per each DecoderLayer (how to call DecoderLayer)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.target_embedding       = nn.Embedding(target_vocab_size, d_model) # use torch.nn.Embedding
        self.positional_encoding    = PositionalEncoding(d_model, max_len)
        self.final_linear           = nn.Linear(d_model, target_vocab_size)
        self.dropout                = nn.Dropout(dropout)


    def forward(self, padded_targets, target_lengths, enc_output, enc_input_lengths):

        # Processing targets
        # create a padding mask for the padded_targets with <PAD_TOKEN>
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_input=padded_targets, input_lengths=target_lengths).to(padded_targets.device)
        # creating an attention mask for the future subsequences (look-ahead mask)
        causal_mask = CausalMask(input_tensor=padded_targets).to(padded_targets.device)
        # computing embeddings for the target sequence
        # Step1:  Apply the embedding
        ''' TODO '''
        embedded_targets = self.target_embedding(padded_targets)

        # Step2:  Apply positional encoding
        ''' TODO '''
        embedded_targets = self.positional_encoding(embedded_targets)
        
        # Step3:  Create attention mask to ignore padding positions in the input sequence during attention calculation
        ''' TODO '''
        pad_mask_enc = None
        if enc_output is not None:
            pad_mask_enc = PadMask(padded_input=enc_output, input_lengths=enc_input_lengths).to(enc_output.device)

        # Step4: Pass through decoder layers
        # @NOTE: store your mha1 and mha2 attention weights inside a dictionary
        # @NOTE: you will want to retrieve these later so store them with a useful name
        ''' TODO '''
        runnint_att = {}
        for i in range(self.num_layers):
            embedded_targets, runnint_att['layer{}_dec_self'.format(i + 1)], runnint_att['layer{}_dec_cross'.format(i + 1)] = self.dec_layers[i](
                embedded_targets, enc_output, pad_mask_enc, pad_mask_dec, causal_mask
            )
    

        # Step5: linear layer (Final Projection) for next character prediction
        ''' TODO '''
        seq_out = self.final_linear(embedded_targets)

        return seq_out, runnint_att


    def recognize_greedy_search(self, enc_output, enc_input_lengths, tokenizer):
        ''' passes the encoder outputs and its corresponding lengths through autoregressive network
            @NOTE: You do not need to make changes to this method.
        '''
        # start with the <SOS> token for each sequence in the batch
        batch_size = enc_output.size(0)
        target_seq = torch.full((batch_size, 1), tokenizer.SOS_TOKEN, dtype=torch.long).to(enc_output.device)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_output.device)

        for _ in range(self.max_len):

            seq_out, runnint_att = self.forward(target_seq, None, enc_output, enc_input_lengths)
            logits = torch.nn.functional.log_softmax(seq_out[:, -1], dim=1)

            # selecting the token with the highest probability
            # @NOTE: this is the autoregressive nature of the network!
            # appending the token to the sequence
            # checking if <EOS> token is generated
            # or opration, if both or one of them is true store the value of the finished sequence in finished variable
            # end if all sequences have generated the EOS token
            next_token = logits.argmax(dim=-1).unsqueeze(1) # Why to unsqueeze?
            target_seq = torch.cat([target_seq, next_token], dim=-1)
            eos_mask = next_token.squeeze(-1) == tokenizer.EOS_TOKEN
            finished |= eos_mask
            if finished.all(): break

        # remove the initial <SOS> token and pad sequences to the same length
        target_seq = target_seq[:, 1:]
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(target_seq,
            (0, self.max_len - max_length), value=tokenizer.PAD_TOKEN)

        return target_seq



class Transformer(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        d_model,
        d_ff,
        initialization,
        std,
        # Embedding
        input_dim,
        time_stride,
        feature_stride,
        embed_dropout,
        # Encoder
        enc_num_layers,
        enc_num_heads,
        speech_max_len,
        enc_dropout,
        # Decoder
        dec_num_layers,
        dec_num_heads,
        dec_dropout,
        trans_max_len
        ):

        super(Transformer, self).__init__()

        self.embedding = SpeechEmbedding(input_dim, d_model, time_stride, feature_stride, embed_dropout)
        speech_max_len = int(np.ceil(speech_max_len/self.embedding.time_downsampling_factor))

        self.encoder  = Encoder(enc_num_layers, d_model, enc_num_heads, d_ff, speech_max_len, target_vocab_size, enc_dropout)

        self.decoder = Decoder(dec_num_layers, d_model, dec_num_heads, d_ff, dec_dropout, trans_max_len, target_vocab_size)

        # initialize the weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # print("Linear")
                if initialization == "uniform":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                # print("Conv2d")
                if initialization == "uniform":
                    nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LSTM):
                # print("LSTM")
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        if initialization == "uniform":
                            nn.init.xavier_uniform_(param)
                        else:
                            nn.init.xavier_normal_(param)
            elif isinstance(module, nn.Embedding):
                # print("Embedding")
                nn.init.normal_(module.weight, std = std)
            elif isinstance(module, nn.MultiheadAttention):
                # print("MultiheadAttention")
                if initialization == "uniform":
                    nn.init.xavier_uniform_(module.in_proj_weight)
                    nn.init.xavier_uniform_(module.out_proj.weight)
                else:
                    nn.init.xavier_normal_(module.in_proj_weight)
                    nn.init.xavier_normal_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)


    def forward(self, padded_input, input_lengths, padded_target, target_lengths, mode:Literal['full', 'dec_cond_lm', 'dec_lm']='full'):
        '''DO NOT MODIFY'''
        if mode == 'full': # Full transformer training
            encoder_output, encoder_lengths          = self.embedding(padded_input, input_lengths, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)
        if mode == 'dec_cond_lm': # Training Decoder as a conditional LM
            encoder_output, encoder_lengths   = self.embedding(padded_input, input_lengths, use_blstm=True)
            ctc_out = None
        if mode == 'dec_lm': # Training Decoder as an LM
            encoder_output, encoder_lengths, ctc_out = None, None, None

        # passing Encoder output through Decoder
        output, attention_weights = self.decoder(padded_target, target_lengths, encoder_output, encoder_lengths)
        return output, attention_weights, ctc_out


    def recognize(self, inp, inp_len, tokenizer, mode:Literal['full', 'dec_cond_lm', 'dec_lm'], strategy:str='greedy'):
        """ sequence-to-sequence greedy search -- decoding one utterance at a time """
        '''DO NOT MODIFY'''
        if mode == 'full':
            encoder_output, encoder_lengths          = self.embedding(inp, inp_len, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)
        
        if mode == 'dec_cond_lm':
            encoder_output, encoder_lengths,  = self.embedding(inp, inp_len, use_blstm=True)
            ctc_out = None
      
        if mode == 'dec_lm':
            encoder_output, encoder_lengths, ctc_out = None, None, None
        
        if strategy =='greedy':
          out = self.decoder.recognize_greedy_search(encoder_output, encoder_lengths, tokenizer=tokenizer)
        elif strategy == 'beam':
          out = self.decoder.recognize_beam_search(encoder_output, encoder_lengths, tokenizer=tokenizer, beam_width=5)
        return out


# Utilities -metrics

def calculateMetrics(reference, hypothesis):
    # sentence-level edit distance
    dist = aF.edit_distance(reference, hypothesis) # What is this for?
    # split sentences into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # compute edit distance
    dist = aF.edit_distance(ref_words, hyp_words)
    # calculate WER
    wer = dist / len(ref_words)
    # convert sentences into character sequences
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    # compute edit distance
    dist = aF.edit_distance(ref_chars, hyp_chars)
    # calculate CER
    cer = dist / len(ref_chars)
    return dist, wer * 100, cer * 100


def calculateBatchMetrics(predictions, y, y_len, tokenizer):
    '''
    Calculate levenshtein distance, WER, CER for a batch
    predictions (Tensor) : the model predictions
    y (Tensor) : the target transcript
    y_len (Tensor) : Length of the target transcript (non-padded positions)
    '''
    batch_size, _  = predictions.shape
    dist, wer, cer = 0., 0., 0.
    for batch_idx in range(batch_size):

        # trim predictons upto the EOS_TOKEN
        pad_indices = torch.where(predictions[batch_idx] == tokenizer.EOS_TOKEN)[0]
        lowest_pad_idx = pad_indices.min().item() if pad_indices.numel() > 0 else len(predictions[batch_idx])
        pred_trimmed = predictions[batch_idx, :lowest_pad_idx]

        # trim target upto EOS_TOKEN
        y_trimmed   = y[batch_idx, 0 : y_len[batch_idx]-1]

        # decodes
        pred_string  = tokenizer.decode(pred_trimmed)
        y_string     = tokenizer.decode(y_trimmed)

        # calculate metrics and update
        curr_dist, curr_wer, curr_cer = calculateMetrics(y_string, pred_string)
        dist += curr_dist
        wer  += curr_wer
        cer  += curr_cer

    # average by batch sizr
    dist /= batch_size
    wer  /= batch_size
    cer  /= batch_size
    return dist, wer, cer, y_string, pred_string



#misc

def save_attention_plot(plot_path, attention_weights, epoch=0, mode: Literal['full', 'dec_cond_lm', 'dec_lm'] = 'full'):
    """
    Saves attention weights plot to a specified path.

    Args:
        plot_path (str): Directory path where the plot will be saved.
        attention_weights (Tensor): Attention weights to plot.
        epoch (int): Current training epoch (default is 0).
        mode (str): Mode of attention - 'full', 'dec_cond_lm', or 'dec_lm'.
    """
    if not isinstance(attention_weights, (np.ndarray, torch.Tensor)):
        raise ValueError("attention_weights must be a numpy array or torch Tensor")

    plt.clf()  # Clear the current figure
    sns.heatmap(attention_weights, cmap="viridis", cbar=True)  # Create heatmap
    plt.title(f"{mode} Attention Weights - Epoch {epoch}")
    plt.xlabel("Target Sequence")
    plt.ylabel("Source Sequence")

    # Save the plot with clearer filename distinction
    attention_type = "cross" if epoch < 100 else "self"
    epoch_label = epoch if epoch < 100 else epoch - 100
    plt.savefig(f"{plot_path}/{mode}_{attention_type}_attention_epoch{epoch_label}.png")




def save_model(model, optimizer, scheduler, metric, epoch, path):
    """
    Saves the model, optimizer, and scheduler states along with a metric to a specified path.

    Args:
        model (nn.Module): Model to be saved.
        optimizer (Optimizer): Optimizer state to save.
        scheduler (Scheduler or None): Scheduler state to save.
        metric (dict): Metric dictionary to be saved.
        epoch (int): Current epoch number.
        path (str): File path for saving.
    """
    # Ensure metric is provided as a tuple with correct structure
    # if not (isinstance(metric, tuple) and len(metric) == 2):
    #     raise ValueError("metric must be a tuple in the form (name, value)")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else {},
            **metric,  # Unpacks the metric name and value
            "epoch": epoch
        },
        path
    )


def load_checkpoint(
    checkpoint_path,
    model,
    embedding_load: bool,
    encoder_load: bool,
    decoder_load: bool,
    optimizer=None,
    scheduler=None
):
    """
    Loads weights from a checkpoint into the model and optionally returns updated model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (Transformer): Transformer model to load weights into.
        embedding_load (bool): Load embedding weights if True.
        encoder_load (bool): Load encoder weights if True.
        decoder_load (bool): Load decoder weights if True.
        optimizer (Optimizer, optional): Optimizer to load state into (if provided).
        scheduler (Scheduler, optional): Scheduler to load state into (if provided).

    Returns:
        model (Transformer): Model with loaded weights.
        optimizer (Optimizer or None): Optimizer with loaded state if provided.
        scheduler (Scheduler or None): Scheduler with loaded state if provided.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()

    # Define the components to be loaded
    load_map = {
        "embedding": embedding_load,
        "encoder": encoder_load,
        "decoder": decoder_load
    }

    # Filter and load the specified components
    for key, should_load in load_map.items():
        if should_load:
            component_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith(key)}
            if component_state_dict:
                model_state_dict.update(component_state_dict)
            else:
                print(f"Warning: No weights found for {key} in checkpoint.")

    # Load the updated state_dict into the model
    model.load_state_dict(model_state_dict, strict = False)
    loaded_components = ", ".join([k.capitalize() for k, v in load_map.items() if v])
    print(f"Loaded components: {loaded_components}")

    # Load optimizer and scheduler states if available and provided
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler


# Train | Validate

def train_step(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    ctc_loss: nn.CTCLoss,
    ctc_weight: float,
    optimizer,
    scaler,
    device: str,
    train_loader: DataLoader,
    tokenizer: Any,
    mode: Literal['full', 'dec_cond_lm', 'dec_lm'],
    config: dict
) -> Tuple[float, float, torch.Tensor]:
    """
    Trains a model for one epoch based on the specified training mode.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.CrossEntropyLoss): The loss function for cross-entropy.
        ctc_loss (nn.CTCLoss): The loss function for CTC.
        ctc_weight (float): Weight of the CTC loss in the total loss calculation.
        optimizer (Optimizer): The optimizer to update model parameters.
        scaler (GradScaler): For mixed-precision training.
        device (str): The device to run training on, e.g., 'cuda' or 'cpu'.
        train_loader (DataLoader): The training data loader.
        tokenizer (Any): Tokenizer with PAD_TOKEN attribute.
        mode (Literal): Specifies the training objective.

    Returns:
        Tuple[float, float, torch.Tensor]: The average training loss, perplexity, and attention weights.
    """
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc=f"[Train mode: {mode}]")
    
    running_loss = 0.0
    running_perplexity = 0.0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Separate inputs and targets based on the mode
        if mode != 'dec_lm':
            inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths = batch
            inputs = inputs.to(device)
        else:
            inputs, inputs_lengths = None, None
            targets_shifted, targets_golden, targets_lengths = batch

        targets_shifted = targets_shifted.to(device)
        targets_golden = targets_golden.to(device)

        # Forward pass with mixed-precision
        with torch.autocast(device_type=device, dtype=torch.float16):
        # with torch.xpu.amp.autocast(enabled = True, dtype = torch.bfloat16):
            raw_predictions, attention_weights, ctc_out = model(inputs, inputs_lengths, targets_shifted, targets_lengths, mode=mode)
            padding_mask = torch.logical_not(torch.eq(targets_shifted, tokenizer.PAD_TOKEN)) # This mask is the opposite to the attention mask

            # Calculate cross-entropy loss
            ce_loss = criterion(raw_predictions.transpose(1, 2), targets_golden) * padding_mask
            loss = ce_loss.sum() / padding_mask.sum()

            # Optionally optimize a weighted sum of ce and ctc_loss from the encoder outputs
            # Only available during full transformer training, a ctc_loss must be passed in
            if mode == 'full' and ctc_loss and ctc_out is not None:
                inputs_lengths = torch.ceil(inputs_lengths.float() / model.embedding.time_downsampling_factor).int()
                inputs_lengths = inputs_lengths.clamp(max=ctc_out.size(0))
                loss = loss + ctc_weight * ctc_loss(ctc_out, targets_golden, inputs_lengths, targets_lengths)

        if isinstance(config["max_gradient_norm"], (float, int)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_gradient_norm"])
        
        # Backward pass and optimization with mixed-precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      
        # Accumulate loss and perplexity for monitoring
        running_loss += float(loss.item())
        perplexity = torch.exp(loss)
        running_perplexity += perplexity.item()

        # Update the progress bar
        batch_bar.set_postfix(
            loss=f"{running_loss / (i + 1):.4f}",
            perplexity=f"{running_perplexity / (i + 1):.4f}"
        )
        batch_bar.update()

        # Clean up to save memory
        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths, ce_loss, loss
        gc.collect()
        torch.cuda.empty_cache()

    # Compute average loss and perplexity
    avg_loss = running_loss / len(train_loader)
    avg_perplexity = running_perplexity / len(train_loader)
    batch_bar.close()

    return avg_loss, avg_perplexity, attention_weights


def validate_step(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: str,
    mode: Literal['full', 'dec_cond_lm', 'dec_lm'],
    threshold: int = 5
) -> Tuple[float, Dict[int, Dict[str, str]], float, float]:
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        tokenizer (Any): Tokenizer with a method to handle special tokens.
        device (str): The device to run validation on, e.g., 'cuda' or 'cpu'.
        mode (Literal): Specifies the validation objective.
        threshold (int, optional): Max number of batches to validate on (for early stopping).

    Returns:
        Tuple[float, Dict[int, Dict[str, str]], float, float]: The average distance, JSON output with inputs/outputs,
                                                               average WER, and average CER.
    """
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc="Val")

    running_distance = 0.0
    running_wer = 0.0
    running_cer = 0.0
    json_output = {}

    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            # Separate inputs and targets based on the mode
            if mode != 'dec_lm':
                inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths = batch
                inputs = inputs.to(device)
                inputs_lengths = inputs_lengths.to(device)
            else:
                inputs, inputs_lengths = None, None
                _, targets_shifted, targets_golden, _, targets_lengths = batch
                
            targets_shifted = targets_shifted.to(device)
            targets_golden = targets_golden.to(device)

            # Perform recognition and calculate metrics
            with torch.autocast(device_type=device, dtype=torch.float16):
            # with torch.xpu.amp.autocast(enabled = True, dtype = torch.bfloat16):
                greedy_predictions = model.recognize(inputs, inputs_lengths, tokenizer=tokenizer, mode=mode)
            dist, wer, cer, y_string, pred_string = calculateBatchMetrics(greedy_predictions, targets_golden, targets_lengths, tokenizer) # Can't infer target_lengths from targets_golden?

            # Accumulate metrics
            running_distance += dist
            running_wer += wer
            running_cer += cer
            json_output[i] = {"Input": y_string, "Output": pred_string}

            # Update progress bar
            batch_bar.set_postfix(
                running_distance=f"{running_distance / (i + 1):.4f}",
                WER=f"{running_wer / (i + 1):.4f}",
                CER=f"{running_cer / (i + 1):.4f}"
            )
            batch_bar.update()

            # Early stopping for thresholded validation
            if threshold and i == threshold:
                break

            del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
            gc.collect()
            torch.cuda.empty_cache()

    # Compute averages
    num_batches = threshold + 1 if threshold else len(val_loader)
    avg_distance = running_distance / num_batches
    avg_wer = running_wer / num_batches
    avg_cer = running_cer / num_batches
    batch_bar.close()

    return avg_distance, json_output, avg_wer, avg_cer


def get_optimizer(config, model):
    optimizer = None
    if config["optimizer"] == "SGD":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=float(config["learning_rate"]),
                                    momentum=config["momentum"],
                                    weight_decay=1E-4,
                                    nesterov=config["nesterov"])

    elif config["optimizer"] == "Adam":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=float(config["learning_rate"]),weight_decay=0.01 )

    elif config["optimizer"] == "AdamW":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=float(config["learning_rate"]),
                                    weight_decay=0.01)
    return optimizer


def get_scheduler(config, optimizer):
    scheduler  =  None
    if config["scheduler"] == "ReduceLR":
        #Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        factor=config["factor"], patience=config["patience"], min_lr=1E-8, threshold=1E-1)

    elif config["scheduler"] == "CosineAnnealing":
        #Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        T_max = config["train_epochs"] if config["mode"] == "full" else config["pretrain_epochs"], eta_min=1E-8)
    return scheduler




def test_step(model, test_loader, tokenizer, device,config):
        model.eval()
        # progress bar
        batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc="Test", ncols=5)


        predictions = []

        ## Iterate through batches
        for i, batch in enumerate(test_loader):

            inputs, _, _, inputs_lengths, _ = batch
            inputs          = inputs.to(device)

            with torch.inference_mode(): #TODO():
                greedy_predictions = model.recognize(inputs, inputs_lengths, tokenizer = tokenizer, mode = "full")#TODO call model recognize function
            
            # @NOTE: modify the print_example to print more or less validation examples
            batch_size, _  = greedy_predictions.shape
            batch_pred = []

            ## TODO decode each sequence in the batch
            for batch_idx in range(batch_size):
                # trim predictons upto the EOS_TOKEN
                pad_indices = torch.where(greedy_predictions[batch_idx] == tokenizer.EOS_TOKEN)[0]
                lowest_pad_idx = pad_indices.min().item() if pad_indices.numel() > 0 else len(greedy_predictions[batch_idx]) + 1
                pred_tensor = greedy_predictions[batch_idx,:lowest_pad_idx]
                pred_string  = tokenizer.decode(pred_tensor)
                print(pred_string)
                batch_pred.append(pred_string)
    
            predictions.extend(batch_pred)

            batch_bar.update()

            del inputs, inputs_lengths
            torch.cuda.empty_cache()

        return predictions

