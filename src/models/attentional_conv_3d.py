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
import torch.nn as nn
from src import mynn


class AttentionalConv3D(nn.Module):
    """
    Attentional Convolutional 3D Network. This module is used to classify 3D images. 
    It uses a backbone to extract features from the images and then applies an attentional layer to the features. 
    Finally, it applies a classifier to the output of the attentional layer.
    Args:
        backbone (torch.nn.Module): The backbone network used to extract features from the images.
        embedding_dim (int): The dimension of the embedding vector.
        num_slices (int): The number of slices to extract from the MRI.
        num_classes (int): The number of classes to classify.
        dropout (float): The dropout probability to apply to the classifier.
        return_attention_weights (bool): Whether to return the attention weights.
    Inputs:
        x (torch.Tensor): The input of the module with shape `(batch_size, num_slices, 3, 224, 224)`.
    Outputs:
        logits (torch.Tensor): The output of the module with shape `(batch_size, num_classes)`.
    """

    def __init__(self, backbone, embedding_dim, num_slices, num_classes, dropout=0.4, return_attention_weights=False):
        super().__init__()
        self.num_slices = num_slices
        self.feat_map_dim = embedding_dim
        self.backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.attention = mynn.AttentionLayer(input_size=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
        self.return_attention_weights = return_attention_weights

    def forward(self, x):
        # ---------------------- x.shape = (batch_size, num_slices, 3, 224, 224) ----------------------------------
        # 1. Reshape input to combine batch and images dimensions to create a single batch of images.
        x = x.view(-1, *x.size()[2:])  # e.g. (32, 80, 3, 224, 224) -> (32 * 80, 3, 224, 224)
        # 2. Pass the input through the backbone
        x = self.backbone(x)  # e.g. (32 * 80, 3, 224, 224) -> (32 * 80, 1280, 7, 7)
        # 3. Apply the AdaptiveAvgPool2d
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)  # e.g. (32 * 80, 1280, 1, 1) -> (32 * 80, 1280)
        # 4. Turn back the batch dimension to separate the 3D images
        x = x.view(-1, self.num_slices, *x.size()[1:])  # e.g. (32 * 80, 1280) -> (32, 80, 1280)
        # 5. Compute the attention
        x, attention_weights = self.attention(x)  # e.g. (32, 80, 1280) -> (32, 1280)
        # 6. Apply the classifier
        x = self.classifier(x)  # e.g. (32, 1280) -> (32, num_classes)
        if self.return_attention_weights:
            return x, attention_weights
        return x
