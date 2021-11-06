import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import torch.nn as nn

def get_dataloaders_celeba(batch_size, num_workers=0,
                           train_transforms=None,
                           test_transforms=None,
                           download=True):
    """ Generate dataset for training, testing and validation

    Args:
        batch_size (int): batch size
        num_workers (int, optional): number of workers. Defaults to 0.
        train_transforms ([type], optional): training transforms. Defaults to None.
        test_transforms ([type], optional): test set transforms. Defaults to None.
        download (bool, optional): condition to download data. Defaults to True.

    Returns:
        [type]: [description]
    """

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.CelebA(root='.',
                                    split='train',
                                    transform=train_transforms,
                                    download=download)

    valid_dataset = datasets.CelebA(root='.',
                                    split='valid',
                                    transform=test_transforms)

    test_dataset = datasets.CelebA(root='.',
                                   split='test',
                                   transform=test_transforms)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader