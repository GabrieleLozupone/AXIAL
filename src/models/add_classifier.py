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
from torch import nn


class BackboneWithClassifier(nn.Module):
    """
    This module is used to attach a classifier to a backbone network.
    @param backbone: The backbone network used to extract features from the images.
    @param embedding_dim: The dimension of the embedding vector.
    @param num_classes: The number of classes to classify.
    @param dropout: The dropout probability to apply to the classifier.
    """

    def __init__(self, backbone, embedding_dim, num_classes, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
