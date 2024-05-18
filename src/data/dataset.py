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
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple
import pandas as pd
import torch
import nibabel as nib
import random
from src.data.transforms import RandomTransformations


class ADNIDataset(Dataset):
    """
        This variant of the ADNI dataset extracts 2D slices from the MRI and returns them as RGB images.
        The transformation is applied to all 2D slices, so it's not a 3D transformation.
        The slices are extracted from the middle of the MRI in the axial plane in both directions.
        Args:
            dataframe: with columns (mri_path, diagnosis)
            transform: torchvision.transforms.Compose
            data_augmentation: torchvision.transforms.Compose
            num_slices: number of slices to extract from the MRI
            slicing_offset: offset to apply to the middle slice to extract the slices
            slicing_from_left: whether to extract the slices from the right or left of the middle slice if offset is not 0
            classes: list of classes
            class_to_idx: dictionary with class names as keys and class indices as values
            revert_slices_order: whether to revert the order of the slices
            slicing_plane: the plane to extract the slices from
        Returns:
            X: torch.Tensor of shape (num_slices, 3, width, height)
            y: int (class index) label
    """

    # Overwrite __init__() method
    def __init__(self,
                 dataframe: pd.DataFrame,
                 transform: transforms.Compose,
                 data_augmentation: RandomTransformations = None,
                 data_augmentation_slice: transforms.Compose = None,
                 num_slices: int = 80,
                 slicing_offset: int = 0,
                 slicing_from_left: bool = True,
                 classes: list = None,
                 class_to_idx: dict = None,
                 revert_slices_order: bool = False,
                 slicing_plane: str = 'axial'):
        # Get the dataframe with the image paths and labels
        self.dataframe = dataframe
        # Get the number of slices to extract from the MRI
        self.num_slices = num_slices
        # Setup transform
        self.transform = transform
        # Setup data augmentation
        self.data_augmentation = data_augmentation
        # Setup data augmentation for the slices
        self.data_augmentation_slice = data_augmentation_slice
        # Create classes and class_to_idx attributes
        self.classes = classes
        self.class_to_idx = class_to_idx
        # Get the offset to apply to the middle slice to extract the slices
        self.slicing_offset = slicing_offset
        # Get the flag to revert the order of the slices
        self.revert_slices_order = revert_slices_order
        # Get the slicing plane
        self.slicing_plane = slicing_plane
        # Get the flag to extract the slices from the right or left of the middle slice
        self.slicing_from_left = slicing_from_left

    # Create a function to load images
    def load_image(self, index: int):
        """Opens an sMRI via a path and returns it."""
        image_path = self.dataframe.mri_path.values[index]
        # Load the image with nibabel  
        return nib.load(image_path)

    # Override __len__()
    def __len__(self) -> int:
        """Returns the total number of samples."""
        return self.dataframe.__len__()

    # Override __getitem__() method to return a particular sample
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns one sample of data, data and label (X, y)."""
        # Load the image
        img = self.load_image(index)
        # Convert the image to a numpy array 
        img = img.get_fdata(dtype='float32')
        # Get the class name and index
        class_name = self.dataframe.diagnosis.values[index]  # expects the dataframe in format (filename, label)
        class_idx = self.class_to_idx[class_name]
        # Create a mapping between the slicing plane and the dimension of the image
        slicing_plane_dim = {'sagittal': 0, 'coronal': 1, 'axial': 2}
        # Get the dimension of the slicing plane
        dim = slicing_plane_dim[self.slicing_plane]
        # Slicing 3D image only if the number of slices is less than the number of slices in the image in that plane
        if self.num_slices < img.shape[dim]:
            # Get the initial and final slice to extract considering the offset
            if self.slicing_from_left:
                initial_slice = max(0, int(img.shape[dim] / 2 - self.num_slices / 2 - self.slicing_offset))
                final_slice = min(img.shape[dim], int(img.shape[dim] / 2 - self.slicing_offset + self.num_slices / 2))
            else:
                initial_slice = max(0, int(img.shape[dim] / 2 + self.slicing_offset - self.num_slices / 2))
                final_slice = min(img.shape[dim], int(img.shape[dim] / 2 + self.slicing_offset + self.num_slices / 2))
            # Extract the slices according to the slicing plane
            if dim == 0:
                img = img[initial_slice:final_slice, :, :]
            elif dim == 1:
                img = img[:, initial_slice:final_slice, :]
            else:
                img = img[:, :, initial_slice:final_slice]
        else:
            self.num_slices = img.shape[dim]
        # Get the resize shape from the transformations if it exists
        resize_shape = None
        for transformation in self.transform.transforms:
            if isinstance(transformation, transforms.Resize):
                resize_shape = transformation.size
                break
        # Create a tensor to store the slices with the correct shape
        if resize_shape is not None:
            transformed_mri = torch.empty((self.num_slices, 3, resize_shape[0], resize_shape[1]))
        else:
            if dim == 0:
                transformed_mri = torch.empty((self.num_slices, 3, img.shape[1], img.shape[2]))
            elif dim == 1:
                transformed_mri = torch.empty((self.num_slices, 3, img.shape[0], img.shape[2]))
            else:
                transformed_mri = torch.empty((self.num_slices, 3, img.shape[0], img.shape[1]))
        # Convert the 3D image in tensor
        img = torch.from_numpy(img).unsqueeze(0)  # e.g. (193, 80, 193) for coronal plane -> (1, 193, 80, 193)
        # Apply data augmentation if it exists
        if self.data_augmentation is not None:
            img = self.data_augmentation(img)
        # Rotate the image to have a standard orientation
        if dim == 0:
            img = torch.rot90(img, 1, [2, 3])
        elif dim == 1:
            img = torch.rot90(img, 1, [1, 3])
        else:
            img = torch.rot90(img, 1, [1, 2])
        # Remove the first dimension
        img = img.squeeze(0)
        # Decide if apply the transformation on the slices with the same probability of apply a transformation to the
        # 3D image
        # if self.data_augmentation is not None:
        #     apply_transformations = random.random() < self.data_augmentation.p
        # Apply the transformation to all slices according to the slicing plane
        for i in range(img.shape[dim]):
            # Convert the slice to PIL image
            if dim == 0:
                single_slice = img[i, :, :]
            elif dim == 1:
                single_slice = img[:, i, :]
            else:
                single_slice = img[:, :, i]
            # Convert the slice tensor image to RGB copying the same tensor 3 times
            single_slice = torch.stack((single_slice, single_slice, single_slice), dim=0)
            # Apply the transformation
            transformed_mri[i, :, :, :] = self.transform(single_slice)
            if self.data_augmentation_slice is not None:
                transformed_mri[i, :, :, :] = self.data_augmentation_slice(transformed_mri[i, :, :, :])
        # Revert the order of the slices if it's specified
        if self.revert_slices_order:
            transformed_mri = torch.flip(transformed_mri, dims=[0])
        return transformed_mri, class_idx  # return data, label (X, y)
