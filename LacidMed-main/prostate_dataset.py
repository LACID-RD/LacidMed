import os
import nibabel as nib  # Library for loading NIFTI files
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Define a custom dataset class for loading NIFTI images and their corresponding masks
class ProstateNiftiDataset(Dataset):
    """
    Custom dataset class for loading NIFTI image and mask pairs for segmentation tasks.
    """

    def __init__(self, image_dir, mask_dir):
        """
        Initializes the dataset with paths to images and masks directories.

        Parameters:
        - image_dir (str): Directory containing the NIFTI image files.
        - mask_dir (str): Directory containing the NIFTI mask files.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii")])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii")])
        
        # Ensure images and masks are paired correctly
        assert len(self.images) == len(self.masks), "Mismatch in number of images and masks!"


    def resize_tensor(self, tensor, size):
        """
        Resize a tensor to the given size using bilinear interpolation.
        Handles 3D tensors (C, H, W) or 4D tensors (B, C, H, W).
        """
        batch_added = False
        if tensor.ndim == 3:  # (C, H, W)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            batch_added = True

        resized = F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

        if batch_added:
            resized = resized.squeeze(0)  # Remove batch dimension

        return resized

    def __getitem__(self, index):
        """
        Retrieves the image and mask pair at the given index and applies transformations.

        Parameters:
        - index (int): Index of the desired pair.

        Returns:
        - img (Tensor): Transformed image tensor.
        - mask (Tensor): Transformed mask tensor.
        """
        # Load the NIFTI image and mask
        img_path = self.images[index]
        mask_path = self.masks[index]

        img = nib.load(img_path).get_fdata()  # Load image data
        mask = nib.load(mask_path).get_fdata()  # Load mask data

        # Normalize image intensity to [0, 1]
        img = (img - img.min()) / (img.max() - img.min())

        # Add a channel dimension for grayscale
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to tensors
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Resize images and masks
        #img = self.transform(img)
        #mask = self.transform(mask)

        mask[mask != 0] = 1

        # Resize the image and mask
        img = self.resize_tensor(img, (256, 256))  
        mask = self.resize_tensor(mask, (256, 256))

        return img, mask

    def __len__(self):
        """
        Returns the total number of image-mask pairs in the dataset.
        """
        return len(self.images)
