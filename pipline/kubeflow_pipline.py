from typing import List

from kfp import client
from kfp import dsl
from kfp.dsl import Dataset
from kfp.dsl import Input
from kfp.dsl import Model
from kfp.dsl import Output

@dsl.component(base_image="python:3.10.17-bullseye", packages_to_install=[])
def load_dataset(Portrait_dataset: Output[Dataset]):
    import os
    import zipfile

    os.system("git clone https://github.com/mean-world/MLOps.git")

    def compress_directory_to_zip(input_dir, output_zip_path):
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, input_dir)
                    zipf.write(file_path, relative_path)


    input_directory = "pipline/test_dataset/"  
    Portrait_dataset.uri = Portrait_dataset.uri + '.zip'
    output_zip_file = Portrait_dataset.path
    compress_directory_to_zip(input_directory, output_zip_file)


@dsl.component(base_image="pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel", packages_to_install=['torch', 'torchvision', 'PIL', 'pytorch-lightning'])
def train_model(
    Portrait_dataset: Input[Dataset],
    model: Output[Model],
):
    import os
    import zipfile
    
    def extract_zip(zip_file_path, extract_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zipf:
            zipf.extractall(extract_path)
    
    zip_file = Portrait_dataset.path
    destination_path = "dataset"  

    os.makedirs(destination_path, exist_ok=True)  # 創建目標資料夾，如果不存在
    extract_zip(zip_file, destination_path)

    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from PIL import Image
    import os
    from torchvision import transforms
    import random
    
    #dataloader part
    class SRDataset(Dataset):
        def __init__(self, hr_dir, lr_dir, transform=None):
            self.hr_dir = hr_dir
            self.lr_dir = lr_dir
            self.hr_images = sorted(os.listdir(hr_dir))
            self.lr_images = sorted(os.listdir(lr_dir))
            self.transform = transform

            assert len(self.hr_images) == len(self.lr_images), "Number of HR and LR images must be the same."
            for hr, lr in zip(self.hr_images, self.lr_images):
                assert os.path.splitext(hr)[0] == os.path.splitext(lr)[0], f"HR and LR filenames do not match: {hr} vs {lr}"

        def __len__(self):
            return len(self.hr_images)

        def __getitem__(self, idx):
            hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
            lr_path = os.path.join(self.lr_dir, self.lr_images[idx])

            hr_image = Image.open(hr_path).convert("RGB")
            lr_image = Image.open(lr_path).convert("RGB")

            if self.transform:
                hr_tensor = self.transform(hr_image)
                lr_tensor = self.transform(lr_image)
                return lr_tensor, hr_tensor
            else:
                return transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_image)

    def create_sr_dataloaders(hr_dir, lr_dir, batch_size, val_split=0.1, test_split=0.1, augment_train=True, augment_val=False, augment_test=False, num_workers=4, random_seed=42):
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        full_dataset = SRDataset(hr_dir, lr_dir)
        dataset_size = len(full_dataset)
        val_size = int(val_split * dataset_size)
        test_size = int(test_split * dataset_size)
        train_size = dataset_size - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))

        # 定義不同的 transform 用於訓練、驗證和測試集
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]) if augment_train else transforms.ToTensor()

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ]) if augment_val else transforms.ToTensor()

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
        ]) if augment_test else transforms.ToTensor()

        train_dataset.transform = train_transforms
        val_dataset.transform = val_transforms
        test_dataset.transform = test_transforms

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader
    
    import torch.nn as nn
    import torch.nn.functional as F

    #model part 
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
        
    import torch.optim as optim
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    class SRLightningModule(pl.LightningModule):
        def __init__(self, learning_rate, hr_dir='data/hr', lr_dir='data/lr', batch_size=16, num_workers=4, random_seed=42):
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
                self.hr_dir, self.lr_dir, self.batch_size, val_split=0.1, test_split=0.1,
                augment_train=True, augment_val=False, augment_test=False,
                num_workers=self.num_workers, random_seed=self.random_seed
            )[0]
            return train_loader

        def val_dataloader(self):
            val_loader = create_sr_dataloaders(
                self.hr_dir, self.lr_dir, self.batch_size, val_split=0.1, test_split=0.1,
                augment_train=False, augment_val=False, augment_test=False,
                num_workers=self.num_workers, random_seed=self.random_seed
            )[1]
            return val_loader

        def test_dataloader(self):
            test_loader = create_sr_dataloaders(
                self.hr_dir, self.lr_dir, self.batch_size, val_split=0.1, test_split=0.1,
                augment_train=False, augment_val=False, augment_test=False,
                num_workers=self.num_workers, random_seed=self.random_seed
            )[2]
            return test_loader
        
    #main part
    learning_rate = 0.001
    batch_size = 2
    max_epochs = 10
    hr_data_dir = 'dataset/target' # 你的高解析度圖片目錄
    lr_data_dir = 'dataset/input' # 你的低解析度圖片目錄
    num_workers = 4
    random_seed = 42


    model = SRLightningModule(
        learning_rate=learning_rate,
        hr_dir=hr_data_dir,
        lr_dir=lr_data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed
    )

    # --- 創建 Trainer ---
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='auto', devices=1)

    # --- 訓練模型 ---
    trainer.fit(model) # 現在不需要傳遞 datamodule 如果你在 LightningModule 中定義了 dataloaders

    # --- 驗證模型 ---
    trainer.validate(model)

    # --- 測試模型 ---
    trainer.test(model)

    # --- 保存模型 ---
    model.uri = model.uri + '.ckpt'
    with open(model.path, 'wb') as f:
         trainer.save_checkpoint("sr_lightning_model.ckpt")
    

@dsl.pipeline(name='kubeflow-pipeline')
def my_pipeline():
    load_dataset_task = load_dataset()
    train_model(load_dataset_task)


endpoint = 'http://localhost:8888'
kfp_client = client.Client(host=endpoint)
run = kfp_client.create_run_from_pipeline_func(
    my_pipeline,)
url = f'{endpoint}/#/runs/details/{run.run_id}'
print(url)