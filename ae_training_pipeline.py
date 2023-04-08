import os
import torch
import wandb
import pandas as pd
import clustering_assessment as ca
from ae import AE
from argparse import ArgumentParser
from torchvision import transforms as T
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy


parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
parser.add_argument("--csv_path", type=str, default="csv_results/")
parser.add_argument("--grayscale", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--ds_name", type=str, default="MNIST")
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument(
    "--ae_size",
    type=str,
    default="medium",
    help="small, medium, large, it changes the channels in AE layers",
)
parser.add_argument("--batch_size", type=int, default=128)

args = parser.parse_args()

DATA_DIR = args.data_dir
CHECKPOINT_PATH = args.checkpoint_path
CSV_PATH = args.csv_path
TRAIN = args.train
DS_NAME = args.ds_name
LATENT_DIM = args.latent_dim
AE_SIZE = args.ae_size
BATCH_SIZE = args.batch_size
GRAYSCALE = args.grayscale
IMAGE_SIZE = args.image_size
transforms = T.Compose([T.ToTensor(), T.Resize(IMAGE_SIZE, IMAGE_SIZE)])


model = AE(IMAGE_SIZE, latent_dim=LATENT_DIM, ae_size=AE_SIZE, grayscale=GRAYSCALE)
