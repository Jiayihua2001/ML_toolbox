import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .embedding import SpeechEmbedding
from .masks import PadMask

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, pad_mask):
        x = self.pre_norm(x)
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=pad_mask)
        x = x + self.dropout_1(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout_2(ffn_output)
        x = self.norm2(x)
        return x, pad_mask

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_len, target_vocab_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.after_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, target_vocab_size)
    def forward(self, x, x_len):
        pad_mask = PadMask(x, input_lengths=x_len)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x_residual = x
        for layer in self.enc_layers:
            x_new, _ = layer(x, pad_mask)
            x = x_new + x_residual
            x_residual = x
        x = self.after_norm(x)
        x_ctc = self.ctc_head(x)
        return x, x_len, x_ctc.log_softmax(2).permute(1, 0, 2)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mha2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
    def forward(self, padded_targets, enc_output, pad_mask_enc, pad_mask_dec, slf_attn_mask):
        # Self-Attention block
        self_attn_out, _ = self.mha1(padded_targets, padded_targets, padded_targets, attn_mask=slf_attn_mask, key_padding_mask=pad_mask_dec)
        padded_targets = self.layernorm1(padded_targets + self.dropout_1(self_attn_out))
        # Cross-Attention block (if encoder output is provided)
        if enc_output is not None:
            cross_attn_out, _ = self.mha2(padded_targets, enc_output, enc_output, key_padding_mask=pad_mask_enc)
            padded_targets = self.layernorm2(padded_targets + self.dropout_2(cross_attn_out))
        # Feed-Forward block
        ffn_out = self.ffn(padded_targets)
        padded_targets = self.layernorm3(padded_targets + self.dropout_3(ffn_out))
        return padded_targets, None, None

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, max_len, target_vocab_size):
        super().__init__()
        self.max_len = max_len
        self.num_layers = num_layers
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.final_linear = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, padded_targets, target_lengths, enc_output, enc_input_lengths):
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, input_lengths=target_lengths)
        from ..models.masks import CausalMask
        causal_mask = CausalMask(padded_targets)
        embedded_targets = self.target_embedding(padded_targets)
        embedded_targets = self.positional_encoding(embedded_targets)
        pad_mask_enc = None
        if enc_output is not None:
            pad_mask_enc = PadMask(enc_output, input_lengths=enc_input_lengths)
        att_weights = {}
        for i, layer in enumerate(self.dec_layers):
            embedded_targets, att1, att2 = layer(embedded_targets, enc_output, pad_mask_enc, pad_mask_dec, causal_mask)
            att_weights[f"layer{i+1}_self"] = att1
            att_weights[f"layer{i+1}_cross"] = att2
        seq_out = self.final_linear(embedded_targets)
        return seq_out, att_weights

class Transformer(nn.Module):
    def __init__(self, target_vocab_size, d_model, d_ff, initialization, std, 
                 input_dim, time_stride, feature_stride, embed_dropout,
                 enc_num_layers, enc_num_heads, speech_max_len, enc_dropout,
                 dec_num_layers, dec_num_heads, dec_dropout, trans_max_len):
        super(Transformer, self).__init__()
        self.embedding = SpeechEmbedding(input_dim, d_model, time_stride, feature_stride, embed_dropout)
        speech_max_len = int((speech_max_len + self.embedding.time_downsampling_factor - 1) // self.embedding.time_downsampling_factor)
        self.encoder = Encoder(enc_num_layers, d_model, enc_num_heads, d_ff, speech_max_len, target_vocab_size, enc_dropout)
        self.decoder = Decoder(dec_num_layers, d_model, dec_num_heads, d_ff, dec_dropout, trans_max_len, target_vocab_size)
        self._init_weights(initialization, std)
    def _init_weights(self, initialization, std):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if initialization == "uniform":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                if initialization == "uniform":
                    nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        if initialization == "uniform":
                            nn.init.xavier_uniform_(param)
                        else:
                            nn.init.xavier_normal_(param)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)
            elif isinstance(module, nn.MultiheadAttention):
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
    def forward(self, padded_input, input_lengths, padded_target, target_lengths, mode='full'):
        if mode == 'full':
            encoder_output, encoder_lengths = self.embedding(padded_input, input_lengths, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)
        elif mode == 'dec_cond_lm':
            encoder_output, encoder_lengths = self.embedding(padded_input, input_lengths, use_blstm=True)
            ctc_out = None
        elif mode == 'dec_lm':
            encoder_output, encoder_lengths, ctc_out = None, None, None
        output, attention_weights = self.decoder(padded_target, target_lengths, encoder_output, encoder_lengths)
        return output, attention_weights, ctc_out

    def recognize(self, inp, inp_len, tokenizer, mode, strategy='greedy'):
        if mode == 'full':
            encoder_output, encoder_lengths = self.embedding(inp, inp_len, use_blstm=False)
            encoder_output, encoder_lengths, ctc_out = self.encoder(encoder_output, encoder_lengths)
        elif mode == 'dec_cond_lm':
            encoder_output, encoder_lengths = self.embedding(inp, inp_len, use_blstm=True)
            ctc_out = None
        elif mode == 'dec_lm':
            encoder_output, encoder_lengths, ctc_out = None, None, None
        if strategy == 'greedy':
            out = self.decoder.recognize_greedy_search(encoder_output, encoder_lengths, tokenizer)
        return out
