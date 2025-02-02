import torch
from typing import List, Literal

class CharTokenizer:
    """
    Simple character-level tokenizer.
    """
    def __init__(self):
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|padding|>"
        self.unk_token = "<|unknown|>"
        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")
        self.vocab = {self.eos_token: 0, self.pad_token: 1, self.unk_token: 2}
        for idx, char in enumerate(characters, start=3):
            self.vocab[char] = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = self.vocab[self.eos_token]
        self.bos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.vocab_size = len(self.vocab)

    def tokenize(self, data: str) -> List[str]:
        return [char for char in data]

    def encode(self, data: str, return_tensors: Literal['pt'] = None) -> List[int]:
        encoded = [self.vocab.get(char.upper(), self.unk_token) for char in data]
        if return_tensors == 'pt':
            return torch.tensor(encoded).unsqueeze(0)
        return encoded

    def decode(self, data: List[int]) -> str:
        if isinstance(data, torch.Tensor):
            data = data.cpu().tolist()
        return ''.join([self.inv_vocab.get(x, '') for x in data])

class GTokenizer:
    """
    Wrapper for pretrained tokenizers or the character-level tokenizer.
    """
    def __init__(self, token_type: Literal['1k', '10k', '50k', 'char'] = 'char', logger=None):
        self.token_type = token_type
        if token_type == 'char':
            self.tokenizer = CharTokenizer()
        else:
            from transformers import AutoTokenizer
            if token_type == '1k':
                self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_1k", use_fast=False)
            elif token_type == '10k':
                self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_10k", use_fast=False)
            elif token_type == '20k':
                self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_20k", use_fast=False)
            elif token_type == '50k':
                self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_50k", use_fast=False)
            elif token_type == '100k':
                self.tokenizer = AutoTokenizer.from_pretrained("alexgichamba/hw4_tokenizer_100k", use_fast=False)
        self.EOS_TOKEN = self.tokenizer.eos_token_id
        self.SOS_TOKEN = self.tokenizer.bos_token_id
        if token_type != "char":
            self.PAD_TOKEN = self.tokenizer.convert_tokens_to_ids('<|padding|>')
        else:
            self.PAD_TOKEN = self.tokenizer.pad_token_id
        self.UNK_TOKEN = self.tokenizer.unk_token_id
        self.VOCAB_SIZE = self.tokenizer.vocab_size

    def tokenize(self, data: str) -> List[str]:
        return self.tokenizer.tokenize(data)

    def encode(self, data: str, return_tensors=False) -> List[int]:
        if return_tensors:
            return self.tokenizer.encode(data, return_tensors='pt')
        return self.tokenizer.encode(data)

    def decode(self, data: List[int]) -> str:
        return self.tokenizer.decode(data)
