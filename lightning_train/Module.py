import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from dataloader import create_sr_dataloaders
from model import SRModel


class SRLightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        hr_dir="data/hr",
        lr_dir="data/lr",
        batch_size=16,
        num_workers=4,
        random_seed=42,
    ):
        super().__init__()
        self.model = SRModel()
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
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        outputs = self(lr_images)
        loss = self.criterion(outputs, hr_images)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        outputs = self(lr_images)
        loss = self.criterion(outputs, hr_images)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

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
