from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class PixelDataset(Dataset):
    def __init__(self, root_a : str, root_b : str, transform : list = None):
        """ Initializes Pixel dataset

        Args:
            root_a (str): Directory for training set A
            root_b (str): Directory for training set B
            transform (list, optional): List of transforms. Defaults to None.
        """
        self.root_a = root_a
        self.root_b = root_b
        self.transform = transform

        self.a_images = os.listdir(root_a)
        self.b_images = os.listdir(root_b)
        self.a_len = len(self.a_images)
        self.b_len = len(self.b_images)
        self.length_dataset = max(self.a_len, self.b_len)

    
    def __len__(self):
        """Returns the length of the dataset
        """
        return self.length_dataset

    def __getitem__(self, index : int):
        """ Returns the image at the index

        Args:
            index (int): Index of the the dataset
        """
        a_img = self.a_images[index]
        b_img = self.b_images[index]

        a_path = os.path.join(self.root_a, a_img)
        b_path = os.path.join(self.root_b, b_img)

        a_img = np.array(Image.open(a_path).convert("RGB"))
        b_img = np.array(Image.open(b_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=a_img, image0=b_img)
            a_img = augmentations["image"]
            b_img = augmentations["image0"]
        
        return a_img, b_img
