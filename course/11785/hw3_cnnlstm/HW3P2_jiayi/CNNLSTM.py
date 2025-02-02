import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class CNN_BiLSTM(nn.Module):

    def __init__(self, hidden_dim, num_layers, dropout, out_size, in_channels,config):
        super(CNN_BiLSTM, self).__init__()
        channels=config["channels"]
        self.hidden_dim = hidden_dim
        # Define the CNN embedding layers
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, channels[0], kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm1d(channels[0]),
            nn.Conv1d(channels[0], channels[1], kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm1d(channels[1]),
            nn.Conv1d(channels[1], channels[2], kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm1d(channels[2]),
            nn.Conv1d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        # Define the BiLSTM layer
        self.bilstm = nn.LSTM(input_size=channels[3], hidden_size=hidden_dim, num_layers=num_layers,
                              dropout=dropout, bidirectional=True)

        # Define the classification layers
        self.classification = nn.Sequential(
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(4 * hidden_dim, out_size)
        )

        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, lx):
        # Permute input for CNN
        x = x.permute(0, 2, 1)
        embeddings = self.embedding(x)
        embeddings = embeddings.permute(0, 2, 1)

        # Pack the sequence for LSTM
        packed_input = pack_padded_sequence(embeddings, lx, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed_input)
        lstm_out, out_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        # Classification and log softmax
        out = self.classification(lstm_out)
        tag_scores = self.logSoftmax(out)

        return tag_scores, out_lengths
