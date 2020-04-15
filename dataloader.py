from torchvision import datasets, transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def load_train_data(dataset_name, batch_size, val_split=0.9, dataset_seed=0):
    if dataset_name.lower() == "mnist":
        dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))
    else:
        dataset = None

    indices = np.arange(0, len(dataset))
    np.random.seed(dataset_seed)
    np.random.shuffle(indices)
    train_amnt = int(len(indices) * val_split)
    train_indices = indices[:train_amnt]
    val_indices = indices[train_amnt:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=val_sampler)
    return train_loader, val_loader


def load_test_data(dataset_name, batch_size):
    if dataset_name.lower() == "mnist":
        dataset = datasets.MNIST('./data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))
    else:
        dataset = None
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return test_loader
