import os
import sys
import random

import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler


class ExampleDataset(data.Dataset):
    def __init__(self, cfg, train):
        self.cfg = cfg
        self.train = train

        if self.train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.dataset = torchvision.datasets.CIFAR10(
            root=cfg.DATA.ROOT, train=train, download=True, transform=transform)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def test_dataset():
    from config.defaults import get_cfg
    cfg = get_cfg()
    dataset = ExampleDataset(cfg, train=True)
    x, y = dataset[4]
    print(x.shape)
    print(y)


if __name__ == '__main__':
    test_dataset()
