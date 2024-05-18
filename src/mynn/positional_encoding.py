# --------------------------------------------------------------------------------
# Copyright (c) 2024 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
# 
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
# 
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------
import torch
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding module injects some information about the relative or absolute position of the tokens in the
    sequence. The positional encodings have the same dimension as the embeddings so that the two can be summed. Here,
    we use sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 80):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionalEncodingV2(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 80):
        """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
        # inherit from Module
        super().__init__()
        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)
        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)
        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)
        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)
        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)
        # add dimension
        pe = pe.unsqueeze(0)
        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # perform dropout
        return self.dropout(x)
