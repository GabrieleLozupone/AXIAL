# -----------------------------------------------------------------------------
# This file contains part of the project "Joint Learning Framework of Cross-Modal Synthesis
# and Diagnosis for Alzheimer's Disease by Mining Underlying Shared Modality Information".
# Original repository: https://github.com/thibault-wch/Joint-Learning-for-Alzheimer-disease.git
# -----------------------------------------------------------------------------

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
from .aware_dataset import AwareDataset
import torch


def get_aware_loaders(
        train_df,
        val_df,
        test_df,
        config
):
    classes = train_df.diagnosis.unique().sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    train_set = AwareDataset(dataframe=train_df,
                             load_size=256,
                             crop_size=256,
                             classes=classes,
                             class_to_idx=class_to_idx,
                             mode='train',
                             slicing_plane=config['slicing_plane'])
    val_set = AwareDataset(dataframe=val_df,
                           load_size=256,
                           crop_size=256,
                           classes=classes,
                           class_to_idx=class_to_idx,
                           mode='val',
                           slicing_plane=config['slicing_plane'])
    test_set = AwareDataset(dataframe=test_df,
                            load_size=256,
                            crop_size=256,
                            classes=classes,
                            class_to_idx=class_to_idx,
                            mode='test',
                            slicing_plane=config['slicing_plane'])
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    num_classes = len(classes)
    print(f"\nTrain images shape: {next(iter(train_dataloader))[0].shape}")
    return train_dataloader, val_dataloader, test_dataloader, num_classes
