import os
from pathlib import Path

import flytekit as fl
import lightning as L
import torch
import torch.nn.functional as F
from flytekit.types.directory import FlyteDirectory
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

acr_uri = 'localhost:30000'


class CustomMNIST(Dataset):
    def __init__(self, file_path, transform=None):
        data_dict = torch.load(file_path)
        self.data = data_dict['data']
        self.targets = data_dict['targets']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform is not None:
            image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


train_spec = fl.ImageSpec(
    name='mnist-model',
    requirements='uv.lock',
    registry=acr_uri,
)


@fl.task(
    container_image=train_spec,
    requests=fl.Resources(mem='1Gi', cpu='1'),
    limits=fl.Resources(mem='1Gi', cpu='1'),
    cache=fl.Cache(version='2.0', serialize=True),
)
def download_mnist_dataset(dataset_path: str) -> FlyteDirectory:
    working_dir = fl.current_context().working_directory
    local_dir = Path(working_dir) / dataset_path
    local_dir.mkdir(exist_ok=True)
    MNIST(local_dir, download=True)
    return FlyteDirectory(path=str(local_dir))


@fl.task(
    container_image=train_spec,
    requests=fl.Resources(mem='2Gi', cpu='1', gpu='1'),
    limits=fl.Resources(mem='2Gi', cpu='1', gpu='1'),
    cache=fl.Cache(version='2.0', serialize=True),
)
def filter(dataset_dir: fl.FlyteDirectory) -> fl.FlyteFile:
    dataset_path = dataset_dir.path
    dataset_dir.download()
    dataset = MNIST(dataset_path, download=False, transform=transforms.ToTensor())
    indices = [i for i in range(len(dataset)) if dataset.targets[i] in [0, 1]]
    filtered_dataset = Subset(dataset, indices)
    output = os.path.join(dataset_path, 'mnist.pt')
    torch.save(
        {
            'data': filtered_dataset.dataset.data[indices],
            'targets': filtered_dataset.dataset.targets[indices],
        },
        output,
    )
    return fl.FlyteFile(path=output)


@fl.task(
    container_image=train_spec,
    requests=fl.Resources(mem='2Gi', cpu='1', gpu='1'),
    limits=fl.Resources(mem='2Gi', cpu='1', gpu='1'),
)
def train(dataset_file: fl.FlyteFile) -> fl.FlyteFile:
    dataset = CustomMNIST(
        file_path=dataset_file,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(dataset)
    autoencoder = LitAutoEncoder(Encoder(), Decoder())

    L.Trainer(max_epochs=5).fit(model=autoencoder, train_dataloaders=train_loader)
    working_dir = fl.current_context().working_directory
    output = Path(working_dir) / 'mnist_decoder.pt'
    return fl.FlyteFile(path=str(output))


@fl.workflow
def mnist_model_wf(dataset_path: str):
    dataset = download_mnist_dataset(dataset_path)
    filter_dataset = filter(dataset)
    train(filter_dataset)
