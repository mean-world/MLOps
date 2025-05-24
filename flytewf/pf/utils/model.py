import os
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# dataloader part
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.transform = transform

        assert len(self.hr_images) == len(self.lr_images), (
            'Number of HR and LR images must be the same.'
        )
        for hr, lr in zip(self.hr_images, self.lr_images):
            assert os.path.splitext(hr)[0] == os.path.splitext(lr)[0], (
                f'HR and LR filenames do not match: {hr} vs {lr}'
            )

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
        else:
            return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_image)


def create_sr_dataloaders(
    hr_dir,
    lr_dir,
    batch_size,
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# model part
class SimpleUpscaleCNN(nn.Module):
    def __init__(self):
        super(SimpleUpscaleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.upsample(x)
        x = self.conv_out(x)
        return x


class SRLightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        hr_dir='data/hr',
        lr_dir='data/lr',
        batch_size=16,
        num_workers=4,
        random_seed=42,
    ):
        super().__init__()
        self.model = SimpleUpscaleCNN()
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        outputs = self(lr_images)
        loss = self.criterion(outputs, hr_images)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        outputs = self(lr_images)
        loss = self.criterion(outputs, hr_images)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        outputs = self(lr_images)
        loss = self.criterion(outputs, hr_images)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_loader = create_sr_dataloaders(
            self.hr_dir,
            self.lr_dir,
            self.batch_size,
            val_split=0.1,
            test_split=0.1,
            augment_train=True,
            augment_val=False,
            augment_test=False,
            num_workers=self.num_workers,
            random_seed=self.random_seed,
        )[0]
        return train_loader

    def val_dataloader(self):
        val_loader = create_sr_dataloaders(
            self.hr_dir,
            self.lr_dir,
            self.batch_size,
            val_split=0.1,
            test_split=0.1,
            augment_train=False,
            augment_val=False,
            augment_test=False,
            num_workers=self.num_workers,
            random_seed=self.random_seed,
        )[1]
        return val_loader

    def test_dataloader(self):
        test_loader = create_sr_dataloaders(
            self.hr_dir,
            self.lr_dir,
            self.batch_size,
            val_split=0.1,
            test_split=0.1,
            augment_train=False,
            augment_val=False,
            augment_test=False,
            num_workers=self.num_workers,
            random_seed=self.random_seed,
        )[2]
        return test_loader
