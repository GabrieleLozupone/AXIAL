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
from torch.utils.data import DataLoader
from src.data.dataset_2d import ADNIDataset2D
from src.data.dataset import ADNIDataset
from torchvision import transforms
from typing import Tuple, Any
import pandas as pd
from src.data.transforms import RandomTransformations


def get_dataloaders(train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    batch_size: int,
                    num_workers: int,
                    num_slices: int,
                    train_transform: transforms.Compose,
                    test_transform: transforms.Compose,
                    data_augmentation: RandomTransformations = None,
                    data_augmentation_slice: transforms.Compose = None,
                    revert_slices_order: bool = False,
                    output_type: str = "3D",
                    slicing_plane: str = "axial") -> \
        Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], int]:
    """
    Create dataloaders for train, validation and test sets.
    The dataloaders iterate over the dataset and return a batch of images and labels.
    The images shape are:
        - (batch_size, 3, width, height) in the 2D case
        - (batch_size, num_slices, 3, width, height) in the 3D case
    Args:
        slicing_plane: the plane to extract the slices from
        revert_slices_order: whether to revert the order of the slices
        train_df: training dataframe with columns (subject, mri_path, diagnosis)
        val_df: validation dataframe with columns (subject, mri_path, diagnosis)
        test_df: testing dataframe with columns (subject, mri_path, diagnosis)
        batch_size: batch size
        num_workers: number of workers for the dataloaders
        num_slices: number of slices to extract from the MRI
        train_transform: torchvision.transforms.Compose
        test_transform: torchvision.transforms.Compose
        data_augmentation: src.data.transforms.RandomTransformations
        data_augmentation_slice: torchvision.transforms.Compose
        output_type: 2D or 3D
    Returns:
        train_dataloader: DataLoader for the training set
        val_dataloader: DataLoader for the validation set
        test_dataloader: DataLoader for the test set
        num_classes: number of classes
    """
    # Create classes and class_to_idx attributes
    classes = train_df.diagnosis.unique().sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # Show the index of each class
    print(f"Class to index: {class_to_idx}")
    if output_type not in ["2D", "3D"]:
        raise ValueError("output_type must be 2D or 3D")
    # Create the datasets
    train_dataset = None
    val_dataset = None
    if output_type == "3D":
        train_dataset = ADNIDataset(dataframe=train_df,
                                    transform=train_transform,
                                    data_augmentation=data_augmentation,
                                    data_augmentation_slice=data_augmentation_slice,
                                    num_slices=num_slices,
                                    classes=classes,
                                    class_to_idx=class_to_idx,
                                    revert_slices_order=revert_slices_order,
                                    slicing_plane=slicing_plane)
        val_dataset = ADNIDataset(dataframe=val_df,
                                  transform=test_transform,
                                  num_slices=num_slices,
                                  classes=classes,
                                  class_to_idx=class_to_idx,
                                  revert_slices_order=revert_slices_order,
                                  slicing_plane=slicing_plane)
    elif output_type == "2D":
        train_dataset = ADNIDataset2D(dataframe=train_df,
                                      transform=train_transform,
                                      data_augmentation=data_augmentation_slice,
                                      num_slices=num_slices,
                                      classes=classes,
                                      class_to_idx=class_to_idx,
                                      slicing_plane=slicing_plane)
        val_dataset = ADNIDataset2D(dataframe=val_df,
                                    transform=test_transform,
                                    num_slices=num_slices,
                                    classes=classes,
                                    class_to_idx=class_to_idx,
                                    slicing_plane=slicing_plane)
    # The test dataset is the same for both 2D and 3D to use the entire MRI in the test phase
    test_dataset = ADNIDataset(dataframe=test_df,
                               transform=test_transform,
                               num_slices=num_slices,
                               classes=classes,
                               class_to_idx=class_to_idx,
                               revert_slices_order=revert_slices_order,
                               slicing_plane=slicing_plane)
    # Create train validation and test dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    # The batch size for the test dataloader is 1 to use the entire MRI in the test phase
    if output_type == "2D":
        batch_size = 1
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    # Print some information about the datasets
    print(f"\nClasses: {classes}\n")
    # Number of classes in train and test sets
    print(f"\nTraining classes distribution: {train_df['diagnosis'].value_counts()}")
    print(f"\nValidation samples: {val_df['diagnosis'].value_counts()}")
    print(f"\nTest samples: {test_df['diagnosis'].value_counts()}")
    # Number of samples in train and test sets
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"\nValidation samples: {len(val_dataset)}")
    print(f"\nTest samples: {len(test_dataset)}")
    # Number of batches in train and test sets
    print(f"\nTrain batches: {len(train_dataloader)}")
    print(f"\nValidation batches: {len(val_dataloader)}")
    print(f"\nTest batches: {len(test_dataloader)}\n")
    # Print the shape of the images
    print(f"\nTrain images shape: {next(iter(train_dataloader))[0].shape}")
    return train_dataloader, val_dataloader, test_dataloader, len(classes)
