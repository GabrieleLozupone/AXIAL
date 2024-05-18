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
import torch.nn.functional as func


class AttentionLayer(nn.Module):
    """
    Attention module. This module is used to compute attention weights and apply them to the input.
    Args:
        input_size (int): The input size of the module.
    Inputs:
        inputs (torch.Tensor): The input of the module with shape `(batch_size, seq_len, input_size)`.
    Outputs:
        weighted_mean (torch.Tensor): The weighted mean of the input. The shape is `(batch_size, input_size)`.
        attention_weights (torch.Tensor): The attention weights. The shape is `(batch_size, seq_len)`.
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, 1)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        # Calculate attention scores
        scores = self.linear(inputs).view(batch_size, seq_len)
        # Apply softmax to get attention weights
        attention_weights = func.softmax(scores, dim=1)
        # Compute the weighted mean
        weighted_mean = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return weighted_mean, attention_weights
