# -----------------------------------------------------------------------------
# This file is part of the project "Joint Learning Framework of Cross-Modal Synthesis
# and Diagnosis for Alzheimer's Disease by Mining Underlying Shared Modality Information".
# Original repository: https://github.com/thibault-wch/Joint-Learning-for-Alzheimer-disease.git
# -----------------------------------------------------------------------------
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import torch


class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        c, h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)
        self.slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]

    def __call__(self, x, slice_change=True):
        if slice_change:
            self.slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *self.slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


class AwareDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 load_size: int = 256,
                 crop_size: int = 256,
                 slicing_plane: str = 'sagittal',
                 classes: list = None,
                 class_to_idx: dict = None,
                 mode: str = 'train'):
        self.dataframe = dataframe
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.load_size = load_size
        self.crop_size = crop_size
        self.mode = mode
        self.slicing_plane = slicing_plane
        if self.mode == 'train' and self.load_size != self.crop_size:
            self.is_transform = True
            self.transforms = RandomCrop3D((1, int(self.load_size), int(self.load_size), int(self.load_size)),
                                           (int(self.crop_size), int(self.crop_size), int(self.crop_size)))
        else:
            self.is_transform = False

    # Create a function to load images
    def load_image(self, index: int):
        """Opens an sMRI via a path and returns it."""
        image_path = self.dataframe.mri_path.values[index]
        # Load the image with nibabel  
        return nib.load(image_path)

    def __len__(self) -> int:
        return self.dataframe.__len__()

    def __getitem__(self, index: int) -> torch.Tensor:
        img = self.load_image(index)
        img = img.get_fdata(dtype='float32')
        # Get the class name and index
        label = self.dataframe.diagnosis.values[index]
        class_idx = self.class_to_idx[label]
        # Normalize the image with min-max scaling
        img = (img - img.min()) / (img.max() - img.min())
        # Pad the image to 256x256x256
        img = np.pad(img, ((128 - img.shape[0] // 2 - img.shape[0] % 2, 128 - img.shape[0] // 2),
                           (128 - img.shape[1] // 2 - img.shape[0] % 2, 128 - img.shape[1] // 2),
                           (128 - img.shape[2] // 2 - img.shape[0] % 2, 128 - img.shape[2] // 2)),
                     'constant', constant_values=(-1, -1))
        A = torch.from_numpy(img).type(torch.FloatTensor)
        if self.is_transform == True:
            A = self.transforms(A, slice_change=True)
        # Swap dimension according to slicing plane
        if self.slicing_plane == 'coronal':
            A = A.permute(1, 0, 2)
        elif self.slicing_plane == 'axial':
            A = A.permute(2, 0, 1)
        else:
            if self.slicing_plane != 'sagittal':
                print("Invalid slicing plane, using sagittal")
        return A, class_idx
