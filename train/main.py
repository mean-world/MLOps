import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 用於顯示訓練進度條

import dataloader
import model

# 檢查是否有可用的 GPU
device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 設定資料目錄和訓練參數 ---
hr_data_dir = "dataset/image"
lr_data_dir = "dataset/downsample_0.25"
learning_rate = 0.001
num_epochs = 10
batch_size = 2
val_split = 0.1
test_split = 0.1
num_workers = 0
random_seed = 42

# --- 2. 創建資料載入器 ---
train_loader, val_loader, test_loader = dataloader.create_sr_dataloaders(
    hr_data_dir,
    lr_data_dir,
    batch_size,
    val_split,
    test_split,
    augment_train=True,
    augment_val=False,
    augment_test=False,
    num_workers=num_workers,
    random_seed=random_seed,
)

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(val_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")

# --- 3. 實例化模型、損失函數和優化器 ---
model = model.SRModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 4. 訓練循環 (需要加入驗證) ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (lr_images, hr_images) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    ):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

    # 驗證階段
    model.eval()  # 將模型設置為評估模式
    val_loss = 0.0
    with torch.no_grad():  # 在評估階段禁用梯度計算
        for i, (lr_val_images, hr_val_images) in enumerate(val_loader):
            lr_val_images = lr_val_images.to(device)
            hr_val_images = hr_val_images.to(device)

            val_outputs = model(lr_val_images)
            val_loss += criterion(val_outputs, hr_val_images).item()

    epoch_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}")

# --- 5. 在測試集上評估模型 (在訓練完成後) ---
model.eval()
test_loss = 0.0
with torch.no_grad():
    for i, (lr_test_images, hr_test_images) in enumerate(test_loader):
        lr_test_images = lr_test_images.to(device)
        hr_test_images = hr_test_images.to(device)

        test_outputs = model(lr_test_images)
        test_loss += criterion(test_outputs, hr_test_images).item()

epoch_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {epoch_test_loss:.4f}")

# --- 6. 保存訓練好的模型 ---
torch.save(model.state_dict(), "sr_model_split.pth")
print("Training finished. Model saved as sr_model_split.pth")
