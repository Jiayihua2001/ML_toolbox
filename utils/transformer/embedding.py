import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.fftpack import dct

class BiLSTMEmbedding(nn.Module):
    """
    A simple 2-layer bidirectional LSTM embedding.
    """
    def __init__(self, input_dim, output_dim, dropout):
        super(BiLSTMEmbedding, self).__init__()
        self.bilstm = nn.LSTM(input_dim, output_dim // 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True,
                              dropout=dropout)
    def forward(self, x, x_len):
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output

class Conv2DSubsampling(nn.Module):
    """
    Convolutional subsampling to reduce the time dimension.
    """
    def __init__(self, input_dim, output_dim, dropout=0.0, time_stride=2, feature_stride=2):
        super(Conv2DSubsampling, self).__init__()
        tstride1, tstride2 = self.closest_factors(time_stride)
        fstride1, fstride2 = self.closest_factors(feature_stride)
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dim, kernel_size=3, stride=(tstride1, fstride1)),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=(tstride2, fstride2)),
            nn.ReLU()
        )
        self.time_downsampling_factor = tstride1 * tstride2
        conv_out_dim = (input_dim - 2) // fstride1 + 1
        conv_out_dim = (conv_out_dim - 2) // fstride2 + 1
        conv_out_dim = output_dim * conv_out_dim
        self.out = nn.Sequential(
            nn.Linear(conv_out_dim, output_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1,2).contiguous().view(b, t, c * f))
        return x
    def closest_factors(self, n):
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        return max(factor, n // factor), min(factor, n // factor)

class SpeechEmbedding(nn.Module):
    """
    Combines convolutional subsampling and an optional BiLSTM.
    """
    def __init__(self, input_dim, output_dim, time_stride, feature_stride, dropout):
        super(SpeechEmbedding, self).__init__()
        self.cnn = Conv2DSubsampling(input_dim, output_dim, dropout=dropout, time_stride=time_stride, feature_stride=feature_stride)
        self.blstm = BiLSTMEmbedding(output_dim, output_dim, dropout)
        self.time_downsampling_factor = self.cnn.time_downsampling_factor
    def forward(self, x, x_len, use_blstm=False):
        x = self.cnn(x)
        x_len = torch.ceil(x_len.float() / self.time_downsampling_factor).int().clamp(max=x.size(1))
        if use_blstm:
            x = self.blstm(x, x_len)
        return x, x_len
