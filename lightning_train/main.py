from Module import SRLightningModule
import pytorch_lightning as pl

if __name__ == "__main__":
    # --- 設定超參數 ---
    learning_rate = 0.001
    batch_size = 16
    max_epochs = 10
    hr_data_dir = "../dataset/image"  # 你的高解析度圖片目錄
    lr_data_dir = "../dataset/downsample_0.25"  # 你的低解析度圖片目錄
    num_workers = 4
    random_seed = 42

    # --- 創建 LightningModule ---
    model = SRLightningModule(
        learning_rate=learning_rate,
        hr_dir=hr_data_dir,
        lr_dir=lr_data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed,
    )

    # --- 創建 Trainer ---
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1)

    # --- 訓練模型 ---
    trainer.fit(
        model
    )  # 現在不需要傳遞 datamodule 如果你在 LightningModule 中定義了 dataloaders

    # --- 驗證模型 ---
    trainer.validate(model)

    # --- 測試模型 ---
    trainer.test(model)

    # --- 保存模型 ---
    trainer.save_checkpoint("sr_lightning_model.ckpt")
    print("Lightning model saved!")
