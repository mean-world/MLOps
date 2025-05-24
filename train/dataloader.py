import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
from torchvision import transforms
import random


class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.transform = transform

        assert len(self.hr_images) == len(self.lr_images), (
            "Number of HR and LR images must be the same."
        )
        for hr, lr in zip(self.hr_images, self.lr_images):
            assert os.path.splitext(hr)[0] == os.path.splitext(lr)[0], (
                f"HR and LR filenames do not match: {hr} vs {lr}"
            )

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
    """
    創建超解析度任務的訓練集、驗證集和測試集資料載入器。

    Args:
        hr_dir (str): 高解析度圖片所在的目錄。
        lr_dir (str): 低解析度圖片所在的目錄。
        batch_size (int): 每個批次的圖片對數量。
        val_split (float, optional): 驗證集佔總資料集的比例。默認為 0.1。
        test_split (float, optional): 測試集佔總資料集的比例。默認為 0.1。
        augment_train (bool, optional): 是否對訓練集啟用資料增強。默認為 True。
        augment_val (bool, optional): 是否對驗證集啟用資料增強。默認為 False。
        augment_test (bool, optional): 是否對測試集啟用資料增強。默認為 False。
        num_workers (int, optional): 資料載入的 worker 數量。默認為 4。
        random_seed (int, optional): 隨機數種子，用於確保資料集切分的可重複性。默認為 42。

    Returns:
        tuple: 包含訓練資料載入器 (train_loader)、驗證資料載入器 (val_loader) 和測試資料載入器 (test_loader)。
    """
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

    # 定義不同的 transform 用於訓練、驗證和測試集
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


# if __name__ == '__main__':
#     # 示例使用
#     hr_data_dir = 'dataset/image' # 你的高解析度圖片目錄
#     lr_data_dir = 'dataset/downsample_0.25' # 你的低解析度圖片目錄
#     batch_size = 16
#     val_split = 0.1
#     test_split = 0.1
#     num_workers = 4
#     random_seed = 42

#     train_loader, val_loader, test_loader = create_sr_dataloaders(
#         hr_data_dir, lr_data_dir, batch_size, val_split, test_split,
#         augment_train=True, augment_val=False, augment_test=False,
#         num_workers=num_workers, random_seed=random_seed
#     )

#     print(f"Train dataset size: {len(train_loader.dataset)}")
#     print(f"Validation dataset size: {len(val_loader.dataset)}")
#     print(f"Test dataset size: {len(test_loader.dataset)}")

#     # 測試資料載入器 (只看一個批次)
#     for i, (lr_batch, hr_batch) in enumerate(train_loader):
#         print("Train LR Batch Shape:", lr_batch.shape)
#         print("Train HR Batch Shape:", hr_batch.shape)
#         break

#     for i, (lr_batch, hr_batch) in enumerate(val_loader):
#         print("Validation LR Batch Shape:", lr_batch.shape)
#         print("Validation HR Batch Shape:", hr_batch.shape)
#         break

#     for i, (lr_batch, hr_batch) in enumerate(test_loader):
#         print("Test LR Batch Shape:", lr_batch.shape)
#         print("Test HR Batch Shape:", hr_batch.shape)
#         break
