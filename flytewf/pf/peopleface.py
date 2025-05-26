import io
import os

import flytekit as fl
import lightning as pl
import ray
import torch
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig
from PIL import Image
from ray import serve
from torchvision import transforms

from .common import compress_directory_to_zip, extract_zip
from .model import SRLightningModule

image_spec = fl.ImageSpec(
    name='download-people-zip',
    registry='localhost:30000',
    apt_packages=['git'],
    python_version='3.10',
)

train_spec = fl.ImageSpec(
    base_image='pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel',
    packages=['torch', 'torchvision', 'pillow', 'lightning', 'ray', 'flytekit'],
    registry='localhost:30000',
)


serve_spec = fl.ImageSpec(
    name='ray-serve',
    requirements='uv.lock',
    registry='localhost:30000',
)

ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={'log-color': 'True'}),
    worker_node_config=[WorkerNodeConfig(group_name='ray-group', replicas=1)],
    enable_autoscaling=False,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=3600,
)


@fl.task(container_image=image_spec)  # , cache=fl.Cache(version='0.1'))
def download_dataset(
    git_src_url: str = 'https://github.com/mean-world/MLOps.git',
    zip_file_name: str = 'pf',
) -> fl.FlyteFile:
    output_dir = str(fl.current_context().working_directory)
    os.system(f'git clone {git_src_url} && mv MLOps {output_dir}')
    out_path = os.path.join(output_dir, f'{zip_file_name}.zip')
    compress_directory_to_zip(os.path.join(output_dir, 'MLOps/pipeline/test_dataset'), out_path)
    return fl.FlyteFile(path=str(out_path))


@fl.task(enable_deck=True, container_image=train_spec)
def train_model(zip_file: fl.FlyteFile) -> fl.FlyteFile:
    destination_path = os.path.join(str(fl.current_context().working_directory), 'dataset')
    os.makedirs(destination_path, exist_ok=True)
    print(zip_file)
    extract_zip(str(zip_file), destination_path)
    learning_rate = 0.001
    batch_size = 2
    max_epochs = 10
    hr_data_dir = os.path.join(destination_path, 'target')
    lr_data_dir = os.path.join(destination_path, 'input')
    num_workers = 0
    random_seed = 42

    sample_model = SRLightningModule(
        learning_rate=learning_rate,
        hr_dir=hr_data_dir,
        lr_dir=lr_data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed,
    )

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='auto', devices=1)

    trainer.fit(sample_model)

    trainer.validate(sample_model)

    trainer.test(sample_model)

    model_file = f'{fl.current_context().working_directory}/yolo.ckpt'
    trainer.save_checkpoint(model_file)
    return fl.FlyteFile(path=model_file)


@fl.task(
    ray_config=ray_config,
    container_image=serve_spec,
)
def deploy_service(model_path: fl.FlyteFile, ray_address: str):
    import base64

    import pytorch_lightning as pl
    import torch.nn as nn

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
        def __init__(
            self,
        ):
            super().__init__()
            self.model = SimpleUpscaleCNN()

        def forward(self, x):
            return self.model(x)

    @serve.deployment
    class ImageUpscaler:
        def __init__(self, state_dict):
            self.model = SRLightningModule()
            self.model.load_state_dict(state_dict)
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

    model_path.download()
    state_dict = torch.load(model_path)
    ray.init(address='ray://localhost:10001')
    deployment = ImageUpscaler.bind(state_dict["state_dict"])
    serve.run(deployment)


@fl.workflow
def main_wf():
    zip_file = download_dataset()
    model_path = train_model(zip_file)
    deploy_service(model_path, 'localhost')
