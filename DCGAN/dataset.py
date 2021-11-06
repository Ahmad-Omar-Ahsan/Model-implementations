import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import torch.nn as nn
from torch.utils.data import random_split

def get_dataloaders_cifar10(batch_size, num_workers=0,
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

    dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transforms)
    val_size = 5000
    train_size = len(dataset) - val_size 

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader,test_loader


   