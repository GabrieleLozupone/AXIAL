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
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import cv2


class ADNIDataset2D(Dataset):
    """
        This variant of the ADNI dataset extracts a single 2D slice from all the possible slices obtainable from the
        MRIs in the dataset.
        The transform is applied to the slice before returning it.
        The slices are extracted from the middle of the MRI in the axial plane in both directions.
        Args:
            dataframe with columns (mri_path, diagnosis)
            transform: torchvision.transforms.Compose
            data_augmentation: torchvision.transforms.Compose
            num_slices: number of slices to extract from the MRI
            classes: list of classes
            class_to_idx: dictionary with class names as keys and class indices as values
        Returns:
            X: torch.Tensor of shape (3, width, height)
            y: int (class index) label
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 transform: transforms.Compose,
                 num_slices: int = 80,
                 classes: list = None,
                 class_to_idx: dict = None,
                 data_augmentation: transforms.Compose = None,
                 slicing_plane: str = 'axial'):
        # Get the dataframe with the image paths and labels
        self.dataframe = dataframe
        # Get the number of slices to extract from the MRI
        self.num_slices = num_slices
        # Setup transform
        self.transform = transform
        # Setup data augmentation
        self.data_augmentation = data_augmentation
        # Create classes and class_to_idx attributes
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.slicing_plane = slicing_plane

    def load_image(self, index: int):
        """
            This function loads an MRI and returns a 2D slice. The slice is extracted according to the index that is
            the i-th of the total possible slices in dataset (number_of_3d_image * num_slices).
        """
        # Get the index of the MRI and the slice
        mri_idx = index // self.num_slices
        slice_idx = index % self.num_slices
        # Get the path of the MRI
        image_path = self.dataframe.mri_path.values[mri_idx]
        # Load the image with nibabel
        img = nib.load(image_path)
        img = img.get_fdata(dtype='float32')
        # Create a mapping between the slicing plane and the dimension of the image
        slicing_plane_dim = {'sagittal': 0, 'coronal': 1, 'axial': 2}
        # Get the dimension of the slicing plane
        dim = slicing_plane_dim[self.slicing_plane]
        # Slice the image from the middle to both directions in the axial plane
        if self.num_slices < img.shape[dim]:
            middle_slice = int(img.shape[dim] / 2)
            initial_slice = middle_slice - int(self.num_slices / 2)
            final_slice = middle_slice + int(self.num_slices / 2)
            if dim == 0:
                img = img[initial_slice:final_slice, :, :]
            elif dim == 1:
                img = img[:, initial_slice:final_slice, :]
            else:
                img = img[:, :, initial_slice:final_slice]
        else:
            self.num_slices = img.shape[dim]
        # Rotate the image to have a standard orientation
        if dim == 0:
            img = np.rot90(img, 1, [1, 2])
        elif dim == 1:
            img = np.rot90(img, 1, [0, 2])
        else:
            img = np.rot90(img, 1, [0, 1])
        # Convert the image to RGB and return it
        if dim == 0:
            return cv2.cvtColor(img[slice_idx, :, :], cv2.COLOR_GRAY2RGB)
        elif dim == 1:
            return cv2.cvtColor(img[:, slice_idx, :], cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(img[:, :, slice_idx], cv2.COLOR_GRAY2RGB)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return self.dataframe.__len__() * self.num_slices

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns one sample of data, data and label (X, y)."""
        # Load the image
        img = self.load_image(index)
        # Get the class name and index
        mri_idx = index // self.num_slices
        class_name = self.dataframe.diagnosis.values[mri_idx]  # expects the dataframe in format (filename, label)
        class_idx = self.class_to_idx[class_name]
        # Apply the transformation to the image
        trans = transforms.ToTensor()
        transformed_mri = self.transform(trans(img))
        # Apply data augmentation
        if self.data_augmentation is not None:
            transformed_mri = self.data_augmentation(transformed_mri)
        return transformed_mri, class_idx  # return data, label (X, y)
