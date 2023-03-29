import pytorch_lightning as pl
from ae import AE
from vae import VAE
import torch
import torchvision
from torchvision import transforms as T
import wandb
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import clustering_assessment as ca
from datetime import datetime
from pytorch_lightning.strategies.ddp import DDPStrategy
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=256)
args = parser.parse_args()
latent_dim = args.latent_dim
models_path = f"/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/models/ae/best_models_dim_{latent_dim}"
if not os.path.exists(models_path):
    os.makedirs(models_path,exist_ok = True)
#ds_list = ["FMNIST","MNIST","CIFAR10","pitbull_tensor", "expert_tensor", "dog_breeds_tensor","tiny_imagenet_tensor"]
ds_list = ["FMNIST","MNIST","CIFAR10"]
for ds_name in ds_list:
    if ds_name == "FMNIST" or "MNIST":
        transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize(32) 
        ]
        )
    else:
        transforms = T.Compose(
        [
            T.ToTensor()
        ]
        )
        
    early_stopping = EarlyStopping("val_loss", patience=5, mode="min")
    wandb_logger = WandbLogger(project="data_assessment_ae", name=f"trial_ae_latent_dim_{latent_dim}_{ds_name}")
    api_key = open(
        "/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/wandb_api_key.txt",
        "r",
    )
    key = api_key.read()
    api_key.close()
    os.environ["WANDB_API_KEY"] = key
    if 'tensor' in ds_name:
        images = torch.load(
            f"/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data/{ds_name}/images.pt"
        )
        labels = torch.load(
            f"/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data/{ds_name}/labels.pt"
        )

        tensor_dataset = torch.utils.data.TensorDataset(images, labels)
        train_ds, test_ds = torch.utils.data.random_split(tensor_dataset, [0.8, 0.2])
        batch_size = 512 if "imagenet" in ds_name else 32
    else:
        train_ds, test_ds, _ = ca.load_dataset(
            ds_name,
            merged=False,
            root="/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data",
            transform=transforms
        )
        batch_size = 1024
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )
    ds_best_model_path = os.path.join(models_path,ds_name)
    # if not os.path.exists(ds_best_model_path):
    #     os.makedirs(ds_best_model_path)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=ds_best_model_path,
        filename=f"{datetime.now()}_" + "{epoch}-{val_loss:.2f}",
    )
    img_width = train_ds[0][0].shape[1]

    model = AE(img_width,latent_dim=latent_dim)
    
    if ds_name == 'FMNIST' or ds_name == 'MNIST':
        model.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.decoder.conv1 = torch.nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        

    trainer = pl.Trainer(
        devices="auto",
        num_nodes=1,
        accelerator="auto",
        max_epochs=5000,
        enable_checkpointing=True,
        precision=16,
        log_every_n_steps=1,
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[early_stopping, checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()
    print(checkpoint_callback.best_model_path)
