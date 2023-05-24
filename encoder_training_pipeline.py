import os
import torch
import wandb
import logging
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pytorch_lightning as pl
from argparse import ArgumentParser
from torchvision import transforms as T
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from ae import AE
from vae import VAE
import clustering_assessment as ca
import datetime
import time
from torch.utils.data.dataset import ConcatDataset

logging.basicConfig(filename="logname.log", level=logging.INFO, filemode="a")
logging.info("Starting")

logger = logging.getLogger()

DEVICE_GPU = torch.device("cuda:0")
DEVICE_CPU = torch.device("cpu")
# pl.seed_everything(42)
parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("--checkpoint_folder_path", type=str, default="checkpoints/")
parser.add_argument("--csv_folder_path", type=str, default="csv_results_unbound/")
parser.add_argument("--embedding_folder_path", type=str, default="embeddings_unbound/")
parser.add_argument("--inference_raw", action="store_true")
parser.add_argument("--grayscale", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--inference", action="store_true")
parser.add_argument("--embed", action="store_true")
parser.add_argument("--ds_name", type=str, default="MNIST")
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--image_size", type=int, default=128)
parser.add_argument("--vae", action="store_true", help="Use VAE instead of AE")
parser.add_argument(
    "--eval_samples_per_class",
    type=int,
    default=75,
    help="Number of samples per class used for binary tree evaluation",
)
parser.add_argument(
    "--alternative_samples_per_class",
    type=int,
    default=20,
    help="Number of samples per class used for alternative binary tree evaluation when \
    the number of classes is too large (bigger than 20 currently)",
)
parser.add_argument(
    "--max_class_count",
    type=int,
    default=20,
    help="Max class count when \
    sampling for binary tree evaluation. If the number of classes is bigger than this, \
    alternative_samples_per_class is extracted instead.",
)
parser.add_argument(
    "--scaling_factor",
    type=float,
    default=1.0,
    help="Rescale number of parameters in AE by scaling the channels in conv layers",
)

parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

DATA_DIR = args.data_dir
TRAIN = args.train
EMBED = args.embed
ALTERNATIVE_SAMPLES_PER_CLASS = args.alternative_samples_per_class
MAX_CLASS_COUNT = args.max_class_count
INFERENCE = args.inference
INFERENCE_RAW = args.inference_raw
GRAYSCALE = args.grayscale
BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size
DS_NAME = args.ds_name
LATENT_DIM = args.latent_dim
SCALING_FACTOR = args.scaling_factor
MODEL_TYPE = "VAE" if args.vae else "AE"
EXPERIMENT_NAME = f"{MODEL_TYPE}_{SCALING_FACTOR}_{DS_NAME}_{LATENT_DIM}"
CHECKPOINT_FOLDER = args.checkpoint_folder_path
CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, f"{EXPERIMENT_NAME}.ckpt")
CSV_PATH = os.path.join(args.csv_folder_path, f"{EXPERIMENT_NAME}.csv")
EMBEDDINGS_FOLDER = args.embedding_folder_path
N_SAMPLES_PER_CLASS = args.eval_samples_per_class


def train(
    train_dataset, val_dataset, batch_size, epochs, model, checkpoint_path, exp_name
) -> None:
    # wandb_logger =
    print(f"Evaluating {exp_name}")
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=3,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        filename=exp_name,
        save_top_k=1,
        mode="min",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )
    trainer = pl.Trainer(
        devices="auto",
        num_nodes=1,
        accelerator="auto",
        enable_checkpointing=True,
        precision=16,
        log_every_n_steps=1,
        max_epochs=epochs,
        logger=WandbLogger(name=exp_name, project="dataset_assessment", log_model=True),
        callbacks=[early_stop_callback, checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(model, train_loader, val_loader)
    print("Training done")
    wandb.finish()


def perform_embedding(model, dataset, device=DEVICE_GPU):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )
    for batch in loader:
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            if isinstance(model, AE):
                embedding = model.fc(model.encoder(x))
            elif isinstance(model, VAE):
                enc_out = model.encoder(x)
                mu = model.fc_mu(enc_out)
                log_var = model.fc_var(enc_out)
                _, _, embedding = model.sample(mu, log_var)

        try:
            embedded_images = torch.cat([embedded_images, embedding])
            target = torch.cat([target, y])
        except NameError:
            embedded_images = embedding
            target = y
    print("Embedding done")
    return embedded_images.cpu(), target.cpu()


def extract_balanced_classes_dataset(
    dataset,
    n_samples_per_class,
    max_class_count=MAX_CLASS_COUNT,
    alternative_n_samples=ALTERNATIVE_SAMPLES_PER_CLASS,
) -> np.ndarray:
    """Extracts a balanced subset of the dataset, with n_samples_per_class samples per class.
    If the dataset has more than max_class_count classes, the number of samples per class is
    reduced to alternative_n_samples.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to extract from
        n_samples_per_class (int): Number of samples per class to extract
        max_class_count (int, optional): Maximum number of classes to consider. Defaults to 20.
        alternative_n_samples (int, optional): Number of samples per class to extract if the
    dataset has more than max_class_count classes. Defaults to 10.

    Returns:
        chosen_indexes (np.ndarray): Array of indexes of the extracted samples
    """
    print(f"Extracting balanced classes, {n_samples_per_class}")
    logger.log(20, "Starting extraction")
    _, ds_labels = zip(*dataset)
    ds_labels = np.array(ds_labels)
    unique_classes = np.unique(ds_labels)
    logger.log(20, unique_classes)
    for unique_label in unique_classes:
        label_indexes = np.where(ds_labels == unique_label)[0]
        random_subsample = np.random.choice(
            label_indexes,
            n_samples_per_class
            if len(unique_classes) < max_class_count
            else alternative_n_samples,
            replace=False,
        )
        try:
            chosen_indexes = np.concatenate((chosen_indexes, random_subsample))
        except:
            chosen_indexes = random_subsample

    return chosen_indexes


def compute_tree_metrics_raw(dataset, n_samples_extracted, csv_save_path=None) -> None:
    indexes = extract_balanced_classes_dataset(dataset, n_samples_extracted)
    logger.log(20, "Starting binary tree metrics computation")
    if isinstance(dataset, ConcatDataset):
        subset = torch.utils.data.Subset(dataset, indexes)
        x_balanced, y_balanced = zip(*subset)
        x_balanced = torch.stack(x_balanced)
    else:
        subset = dataset[indexes]
        x_balanced, y_balanced = subset[:][0], subset[:][1]

    x_balanced = x_balanced.flatten(start_dim=1).numpy()
    y_balanced = np.array(y_balanced)
    logger.log(20, f"Balanced classes flattened, shape: {x_balanced.shape} total")
    tree_classifier = DecisionTreeClassifier(random_state=0)
    tree_classifier.fit(x_balanced, y_balanced)
    result_dict = {
        "mean_impurity": float(tree_classifier.tree_.impurity.mean()),
        "std_impurity": float(tree_classifier.tree_.impurity.std()),
        "num_leaves": float(tree_classifier.tree_.n_leaves),
        "max_depth": float(tree_classifier.tree_.max_depth),
        "capacity": float(tree_classifier.tree_.capacity),
    }
    result_dataframe = pd.DataFrame(result_dict, index=[0])
    result_dataframe.to_csv(csv_save_path, index=False)
    print(f"Metrics computed and saved to {csv_save_path}")


def extract_balanced_classes(features, target, n_samples_per_class):
    print(f"Extracting balanced classes, {n_samples_per_class}")
    unique_classes = np.unique(target)
    for unique_label in unique_classes:
        label_indexes = np.where(target == unique_label)[0]
        random_subsample = np.random.choice(
            label_indexes,
            n_samples_per_class,
            replace=False,
        )
        try:
            chosen_indexes = np.concatenate((chosen_indexes, random_subsample))
        except:
            chosen_indexes = random_subsample
    return features[chosen_indexes], target[chosen_indexes]


def compute_tree_metrics_embeddings(
    features,
    target,
    n_samples_extracted,
    csv_save_path=None,
) -> None:
    x_balanced, y_balanced = extract_balanced_classes(
        features, target, n_samples_extracted
    )
    tree_classifier = DecisionTreeClassifier(random_state=0)
    tree_classifier.fit(x_balanced, y_balanced)
    result_dict = {
        "mean_impurity": float(tree_classifier.tree_.impurity.mean()),
        "std_impurity": float(tree_classifier.tree_.impurity.std()),
        "num_leaves": float(tree_classifier.tree_.n_leaves),
        "max_depth": float(tree_classifier.tree_.max_depth),
        "capacity": float(tree_classifier.tree_.capacity),
    }
    result_dataframe = pd.DataFrame(result_dict, index=[0])
    result_dataframe.to_csv(csv_save_path, index=False)
    print(f"Metrics computed and saved to {csv_save_path}")


class ImageTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dirpath, transform=None) -> None:
        self.images = torch.load(f"{dirpath}/images.pt")
        self.labels = torch.load(f"{dirpath}/labels.pt")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple:
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    if not os.path.exists(args.checkpoint_folder_path):
        os.makedirs(args.checkpoint_folder_path)
    if not os.path.exists(args.csv_folder_path):
        os.makedirs(args.csv_folder_path)
    if not os.path.exists(args.embedding_folder_path):
        os.makedirs(args.embedding_folder_path)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    transforms = T.Compose([T.ToTensor(), T.Resize([IMAGE_SIZE, IMAGE_SIZE])])
    if args.vae:
        encoding_model = VAE(
            IMAGE_SIZE,
            latent_dim=LATENT_DIM,
            ae_size=SCALING_FACTOR,
            grayscale=GRAYSCALE,
        )
    else:
        encoding_model = AE(
            IMAGE_SIZE,
            latent_dim=LATENT_DIM,
            ae_size=SCALING_FACTOR,
            grayscale=GRAYSCALE,
        )
    api_key = open("/net/tscratch/people/plgmazurekagh/cyfrovet/wandb_api_key.txt", "r")
    key = api_key.read()
    api_key.close()
    os.environ["WANDB_API_KEY"] = key
    if "tensor" in DS_NAME:
        ## deleting the to tensor transform - disguiting, but still, works
        transforms.transforms.pop(0)
        full_dataset = ImageTensorDataset(
            f"{DATA_DIR}/{DS_NAME}/", transform=transforms
        )
        if TRAIN:
            train_ds, test_ds = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
    else:
        if TRAIN:
            train_ds, test_ds, _ = ca.load_dataset(
                DS_NAME, merged=False, root=DATA_DIR, transform=transforms
            )
            ## full ds for further inference
            if EMBED:
                full_dataset = torch.utils.data.ConcatDataset([train_ds, test_ds])
        elif not TRAIN and (EMBED or INFERENCE_RAW):
            full_dataset, _ = ca.load_dataset(
                DS_NAME, merged=True, root=DATA_DIR, transform=transforms
            )

    if TRAIN:
        train(
            train_ds,
            test_ds,
            BATCH_SIZE,
            1000,
            encoding_model,
            CHECKPOINT_FOLDER,
            EXPERIMENT_NAME,
        )

    if EMBED:
        try:
            if MODEL_TYPE == "AE":
                trained_model = AE.load_from_checkpoint(CHECKPOINT_PATH).to(DEVICE_GPU)
            elif MODEL_TYPE == "VAE":
                trained_model = VAE.load_from_checkpoint(CHECKPOINT_PATH).to(DEVICE_GPU)
        except FileNotFoundError:
            print("No checkpoint found, check filepath or train the model first.")

        images_embedded, labels = perform_embedding(
            trained_model, full_dataset, DEVICE_GPU
        )
        torch.save(
            images_embedded, f"{EMBEDDINGS_FOLDER}/{EXPERIMENT_NAME}_images_embedded.pt"
        )
        torch.save(labels, f"{EMBEDDINGS_FOLDER}/{EXPERIMENT_NAME}_labels.pt")

    if INFERENCE:
        try:
            images_embedded, labels
        except NameError:
            print(
                "No embeddings or labels found found, trying to load them from disk..."
            )
            print(f"{EMBEDDINGS_FOLDER}/{EXPERIMENT_NAME}_images_embedded.pt")
            try:
                images_embedded = torch.load(
                    f"{EMBEDDINGS_FOLDER}/{EXPERIMENT_NAME}_images_embedded.pt"
                )
                labels = torch.load(f"{EMBEDDINGS_FOLDER}/{EXPERIMENT_NAME}_labels.pt")
            except FileNotFoundError:
                print("No files found, check filepath or embed the dataset first.")
        images_embedded = images_embedded.cpu().numpy()
        labels = labels.cpu().numpy()

        compute_tree_metrics_embeddings(
            images_embedded, labels, N_SAMPLES_PER_CLASS, CSV_PATH
        )

    if INFERENCE_RAW:
        time_start = time.time()
        print("Computing metrics for raw images...")
        compute_tree_metrics_raw(
            full_dataset,
            N_SAMPLES_PER_CLASS,
            os.path.join(args.csv_folder_path, f"{DS_NAME}_raw.csv"),
        )
        print(f"Done in {time.time() - time_start} seconds.")
