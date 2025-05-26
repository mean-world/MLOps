def deploy(model_path: str, ray_address: str):
    import base64
    import io

    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    from PIL import Image
    from ray import serve
    from torchvision import transforms

    def pil_to_base64(pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format='PNG')
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

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
        def __init__(self, learning_rate=0.001, hr_dir='data/hr', lr_dir='data/lr', batch_size=16, num_workers=4, random_seed=42):
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

    @serve.deployment
    class ImageUpscaler:
        def __init__(self, checkpoint_path):
            self.model = SRLightningModule.load_from_checkpoint(checkpoint_path)
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.transform = transforms.Compose([transforms.ToTensor()])

        async def __call__(self, request):
            image_bytes = await request.body()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            lr_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                sr_tensor = self.model(lr_tensor)
            sr_image = transforms.ToPILImage()(sr_tensor.squeeze(0).cpu())
            base64_image = pil_to_base64(sr_image)
            return {'super_resolution_image': base64_image}

    # ray.init(address=f'ray://{ray_address}:10001')
    deployment = ImageUpscaler.bind(checkpoint_path=str(model_path))
    serve.run(deployment)


deploy('yolo.ckpt', 'localhost')
