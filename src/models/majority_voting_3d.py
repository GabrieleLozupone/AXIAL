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


class MajorityVoting3D(nn.Module):
    """
    Majority Voting on 3D images. This module is used to classify 3D images. 
    It uses a backbone to extract features from the images and then applies a mean layer. 
    Finally, it applies a classifier to the output of the mean layer.
    Args:
        backbone (torch.nn.Module): The backbone network used to extract features from the images.
        embedding_dim (int): The dimension of the embedding vector.
        num_slices (int): The number of slices to extract from the MRI.
        num_classes (int): The number of classes to classify.
        dropout (float): The dropout probability to apply to the classifier.
    Inputs:
        x (torch.Tensor): The input of the module with shape `(batch_size, num_slices, 3, 224, 224)`.
    Outputs:
        logits (torch.Tensor): The output of the module with shape `(batch_size, num_classes)`.
    """

    def __init__(self, backbone, embedding_dim, num_slices, num_classes, dropout=0.4):
        super().__init__()
        self.num_slices = num_slices
        self.feat_map_dim = embedding_dim
        self.backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        # ---------------------- x.shape = (batch_size, num_slices, 3, 224, 224) ----------------------------------
        # 1. Reshape input to combine batch and images dimensions to create a single batch of images.
        x = x.view(-1, *x.size()[2:])  # e.g. (32, 80, 3, 224, 224) -> (32 * 80, 3, 224, 224)
        # 2. Pass the input through the backbone
        x = self.backbone(x)  # e.g. (32 * 80, 3, 224, 224) -> (32 * 80, 1280, 7, 7)
        # 3. Apply the AdaptiveAvgPool2d
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)  # e.g. (32 * 80, 1280, 1, 1) -> (32 * 80, 1280)
        # 4. Compute the prediction
        x = self.classifier(x)  # e.g. (32 * 80, 1280) -> (32 * 80, num_classes)
        # 5. Turn back the batch dimension to separate the 3D images
        x = x.view(-1, self.num_slices, *x.size()[1:])  # e.g. (32 * 80, num_classes) -> (32, 80, num_classes)
        # 6. Compute majority voting
        x = x.mean(dim=1)  # e.g. (32, 80, num_classes) -> (32, num_classes)
        return x
