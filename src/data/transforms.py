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
import random
import torchio as tio


class RandomTransformations:
    def __init__(self, transformations, p=0.5):
        self.transformations = transformations
        self.p = p

    def __call__(self, img):
        # Decide if we are going to apply the transformations
        apply_transformations = random.random() < self.p
        if not apply_transformations:
            return img
        # Decide on the transformations to apply
        applied_transforms = {trans: random.random() < p for trans, p in self.transformations.items()}
        # Apply the transformations
        for trans, apply in applied_transforms.items():
            if apply:
                # print(f"Applied {trans}")
                img = trans(img)
        return img

    def __str__(self) -> str:
        s = "RandomTransformations{\n\n"
        for trans, p in self.transformations.items():
            s += trans.__str__() + f" with probability {p}:\n"
            s += trans.__dict__.__str__() + "\n\n"
        s += "}"
        return s


def get_transforms_from_config(config) -> RandomTransformations:
    # Check if we are going to apply any transformation
    if config['RandomTransformations']['probability'] == 0:
        print("No 3D data augmentation will be applied")
        return None
    # Parse the YAML content and construct transformations
    transformations = {}
    for item in config['RandomTransformations']['transformations']:
        # Dynamically get the transformation class from the 'tio' module
        transformation_class = getattr(tio, item['transformation'])
        # Create the transformation object with the provided parameters
        transformation = transformation_class(**item['parameters'])
        # Add the transformation to the dictionary with its probability
        transformations[transformation] = item['probability']
    return RandomTransformations(transformations=transformations, p=config['RandomTransformations']['probability'])


def get_transforms(data_augmentation_config=None, slicing_plane="axial", data_augmentation_slice=True):
    """
    Returns the transformations to apply to the 2D slices.
    The two variants of ADNIDataset (2D and 3D) consider data augmentation as different transformations.
    @return: transform, data_augmentation
    """
    # Define the transformations to apply to the 2D slices
    normalize = transforms.Normalize(mean=[175.5078, 175.5078, 175.5078],
                                     std=[216.1257, 216.1257, 216.1257])  # for axial
    if slicing_plane == "coronal":
        normalize = transforms.Normalize(mean=[173.4015, 173.4015, 173.4015],
                                         std=[214.3374, 214.3374, 214.3374])
    elif slicing_plane == "sagittal":
        normalize = transforms.Normalize(mean=[168.9790, 168.9790, 168.9790],
                                         std=[214.2967, 214.2967, 214.2967])
    simple_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        normalize,
    ])
    # Define the transformations to apply to the 2D slices
    slice_transform = None
    if data_augmentation_slice:
        slice_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomApply([
            #     transforms.RandomRotation(degrees=(-10, 10))
            # ], p=0.3),
            # transforms.RandomApply([
            #     transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 1.3))
            # ], p=0.1),
            # transforms.RandomApply([
            #     transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            # ], p=0.3),
            # transforms.RandomApply([
            #     transforms.ElasticTransform(alpha=(5.0, 15.0), sigma=(3.0, 6.0))
            # ], p=0.3),
        ])
    # Define the transformations to apply to the 3D volumes
    if data_augmentation_config is None:
        data_augmentation = None
    else:
        data_augmentation = get_transforms_from_config(data_augmentation_config)
    return simple_transform, data_augmentation, slice_transform
