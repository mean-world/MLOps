#!/usr/bin/env python3
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class SRDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, transform=None):
        self.hr_dir, self.lr_dir = hr_dir, lr_dir
        self.hr_images = sorted(os.listdir(self.hr_dir))
        self.lr_images = sorted(os.listdir(self.lr_dir))
        self.transform = transform
        assert len(self.hr_images) == len(self.lr_images), 'Count of images in hr and lr should be same.'
        for hr, lr in zip(self.hr_images, self.lr_images):
            assert os.path.splitext(hr)[0] == os.path.splitext(lr)[0], f'HR and LR filenames do not match: {hr} vs {lr}'

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])

        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')

        if self.transform:
            hr_tensor = self.transform(hr_image)
            lr_tensor = self.transform(lr_image)
            return lr_tensor, hr_tensor
        return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_image)


def create_sr_dataloaders(
    hr_dir: str,
    lr_dir: str,
    batch_size: int,
    val_split=0.1,
    test_split=0.1,
    augment_train=True,
    augment_val=False,
    augment_test=False,
    num_workers=4,
    random_seed=42,
):
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    full_dataset = SRDataset(hr_dir, lr_dir)
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_transforms = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        if augment_train
        else transforms.ToTensor()
    )
    val_transforms = (
        transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if augment_val
        else transforms.ToTensor()
    )
    test_transforms = (
        transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if augment_test
        else transforms.ToTensor()
    )

    train_dataset.transform = train_transforms
    val_dataset.transform = val_transforms
    test_dataset.transform = test_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
