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
"""
Functions for creating neural network classifiers.
"""
import torch
from torch import nn


def get_classifier(embedding_dim, num_classes, dropout, hidden_dim=0, num_hidden_layers=0):
    """
    Get a classifier based on the specified parameters.
    @param embedding_dim: The embedding dimension of the backbone
    @param num_classes: The number of classes
    @param dropout: The dropout probability
    @param hidden_dim: The hidden dimension of the hidden layers
    @param num_hidden_layers: The number of hidden layers
    @return: A classifier module based on the specified parameters.
    """
    if num_hidden_layers == 0:
        # Return a fully connected layer
        return nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, num_classes)
        )
    else:
        return nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout, inplace=True)
            ) for _ in range(num_hidden_layers)]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
