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
import torch


class TransformerConv3D(nn.Module):
    """
    Transformer Convolutional 3D Network. This module is used to classify 3D images.
    It uses a backbone to extract features from the images, then starting from the center slice of the MRI, it encodes
    the slice using transformer heads when Q are the feature vector of the center slice, K and V are the feature map
    of the others MRI slices. Finally, it applies a classifier to the output of the attention fusion.
    Args:
        backbone (torch.nn.Module): The backbone network used to extract features from the images.
        embedding_dim (int): The dimension of the embedding vector.
        num_heads (int): The number of transformer heads to use.
        num_slices (int): The number of slices to extract from the MRI.
        num_classes (int): The number of classes to classify.
        dropout (float): The dropout probability to apply to the classifier.
    Inputs:
        x (torch.Tensor): The input of the module with shape `(batch_size, num_slices, 3, 224, 224)`.
    Outputs:
        logits (torch.Tensor): The output of the module with shape `(batch_size, num_classes)`.
    """

    def __init__(self, backbone, embedding_dim, num_slices, num_heads, num_classes, dropout=0.4, attn_dropout=0.1):
        super().__init__()
        self.num_slices = num_slices
        self.feat_map_dim = embedding_dim
        self.backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pos_enc = mynn.PositionalEncodingV2(d_model=embedding_dim, max_length=num_slices)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout,
                                                     batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        # ---------------------- x.shape = (32, 80, 3, 224, 224) ----------------------------------
        # 1. Reshape input to combine batch and images dimensions to create a single batch of images.
        x = x.view(-1, *x.size()[2:])  # e.g. (32, 80, 3, 224, 224) -> (32 * 80, 3, 224, 224)
        # 2. Pass the input through the backbone
        x = self.backbone(x)  # e.g. (32 * 80, 3, 224, 224) -> (32 * 80, 512, 7, 7)
        # 3. Apply the AdaptiveAvgPool2d
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)  # e.g. (32 * 80, 512, 7, 7) -> (32 * 80, 512)
        # 4. Turn back the batch dimension to separate the 3D images
        x = x.view(-1, self.num_slices, *x.size()[1:])  # e.g. (32 * 80, 512) -> (32, 80, 512)
        # 5. Apply positional encoding
        x = self.pos_enc(x)
        # 6. Compute attention with transformer heads
        center = self.num_slices // 2
        q = x[:, center, :].unsqueeze(1)
        # get k that is all the slices except the center one
        k = torch.cat((x[:, :center, :], x[:, center + 1:, :]), dim=1)
        v = k
        x, _ = self.cross_attention(q, k, v)
        # 7. Squeeze the second dimension
        x = x.squeeze(1)  # e.g. (32, 1, 512) -> (32, 512)
        # 8. Apply the classifier
        x = self.classifier(x)  # e.g. (32, 512) -> (32, num_classes)
        return x
