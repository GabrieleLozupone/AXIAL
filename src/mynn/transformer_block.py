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
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attention_output, _ = self.attention(query, key, value)
        # Apply normalization and the residual connection
        out1 = self.norm1(query + self.dropout(attention_output))
        # Pass through the feed-forward network, apply dropout, then normalization and add the residual connection
        forward_output = self.feed_forward(out1)
        out2 = self.norm2(out1 + self.dropout(forward_output))
        return out2
