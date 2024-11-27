"dcoder.py: Define the customized XML decoder"

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import math

class XMLDecoderConfig(PretrainedConfig):
    """The config class for XML decoder. """

    model_type='xml_decoder'
    
    def __init__(self, vocab_size=30522, num_hiddens=768, ffn_hidden_size=3072, num_heads=12, max_seq_len=512, num_blks=12, dropout=0.1):
        super().__init__(vocab_size=vocab_size, num_hiddens=num_hiddens, ffn_hidden_size=ffn_hidden_size, max_seq_len=max_seq_len,
                         num_blks=num_blks, num_heads=num_heads, dropout=dropout)

        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.ffn_hidden_size = ffn_hidden_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_blks = num_blks
        self.dropout = dropout
        self.model_type = 'xml_decoder'


class XMLDecoder(PreTrainedModel):
    """The main class of XMLDecoder, utilizing the XMLDecoderBlocks to build a
        transformer decoder for strcutured XML output. """
    
    config_class = XMLDecoderConfig

    def __init__(self, config: XMLDecoderConfig):
        super().__init__(config)
        self.num_hiddens = config.num_hiddens
        self.num_blks = config.num_blks

        self.blks = nn.Sequential()
        self.pos_enc = PositionalEncoding(config.num_hiddens, config.dropout, max_len=1000)
        self.embedding = nn.Embedding(config.vocab_size, config.num_hiddens)
        for i in range(self.num_blks):
            self.blks.add_module(f"block{i}", XMLDecoderBlock(config.num_hiddens, config.ffn_hidden_size, config.num_heads, config.dropout, i))
        self.dense = nn.Linear(self.num_hiddens, config.vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, input_ids, encoder_hidden_states, attention_mask=None, encoder_attention_mask=None):
        X = self.pos_enc(self.embedding(input_ids) * math.sqrt(self.num_hiddens))

        for i, blks in enumerate(self.blks):
            X, state = blks(X, state)
        
        return self.dense(X), state


class XMLDecoderBlock(nn.Module):
    """The XMLDecoderBlock used in XMLDecoder. """
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        # num_hiddens denotes the feature dimension of X throughout
        super().__init__()
        self.i = i
        self.attention1 = nn.MultiheadAttention(num_hiddens, num_heads, dropout)
        self.residual_norm1 = ResidualNorm(num_hiddens, dropout)

        self.attention2 = nn.MultiheadAttention(num_hiddens, num_heads, dropout)
        self.residual_norm2 = ResidualNorm(num_hiddens, dropout)

        self.ffn = PointwiseFFN(ffn_num_hiddens, num_hiddens, num_hiddens)
        self.residual_norm3 = ResidualNorm(num_hiddens, dropout)
    
    def forward(self, X, state):
        enc_output, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.residual_norm1(X, X2)

        Y2 = self.attention2(Y, enc_output, enc_output, enc_valid_lens)
        Z = self.residual_norm2(Y, Y2)
        return self.residual_norm3(Z, self.ffn(Z)), state


class ResidualNorm(nn.Module):
    def __init__(self, shape, dropout):
        super().__init__()
        self.shape = shape
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(shape)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, X, Y):
        return self.layer_norm(self.dropout_layer(Y) + X)
    

class PointwiseFFN(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_hiddens):
        super().__init__()
        self.layer1 = nn.Linear(num_hiddens, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        return self.layer2(self.relu(self.layer1(X)))


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    