import os
import torch
import wandb
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
import clustering_assessment as ca

DEVICE_GPU = torch.device("cuda:0")
DEVICE_CPU = torch.device("cpu")
pl.seed_everything(42)


def train(
    train_dataset, val_dataset, batch_size, epochs, model, checkpoint_path, exp_name
) -> None:
    wandb_logger = WandbLogger(
        name=exp_name, project="dataset_assessment", log_model=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
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
        logger=wandb_logger,
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
            embedding = model.fc(model.encoder(x))
        try:
            embedded_images = torch.cat([embedded_images, embedding])
            target = torch.cat([target, y])
        except NameError:
            embedded_images = embedding
            target = y
    print("Embedding done")
    return embedded_images.cpu(), target.cpu()


def extract_balanced_classes(features, target, n_samples_per_class):
    unique_classes = np.unique(target)
    indexes = []
    for n, sample_label in enumerate(target):
        if sum(target[indexes] == sample_label) < n_samples_per_class:
            indexes.append(n)
        if sum(
            np.unique(target[indexes], return_counts=True)[1]
        ) == n_samples_per_class * len(unique_classes):
            break
    return features[indexes], target[indexes]


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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--checkpoint_folder_path", type=str, default="checkpoints/")
    parser.add_argument("--csv_folder_path", type=str, default="csv_results/")
    parser.add_argument("--embedding_folder_path", type=str, default="embeddings/ae/")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--ds_name", type=str, default="MNIST")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument(
        "--ae_scaling_factor",
        type=float,
        default=1.0,
        help="Rescale number of parameters in AE by scaling the channels in conv layers",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    TRAIN = args.train
    EMBED = args.embed
    INFERENCE = args.inference
    DS_NAME = args.ds_name
    LATENT_DIM = args.latent_dim
    AE_SCALING_FACTOR = args.ae_scaling_factor
    BATCH_SIZE = args.batch_size
    GRAYSCALE = args.grayscale
    IMAGE_SIZE = args.image_size
    EXPERIMENT_NAME = f"ae_{AE_SCALING_FACTOR}_{DS_NAME}_{LATENT_DIM}"
    CHECKPOINT_FOLDER = args.checkpoint_folder_path
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_FOLDER, f"{EXPERIMENT_NAME}.ckpt")
    CSV_PATH = os.path.join(args.csv_folder_path, f"{EXPERIMENT_NAME}.csv")
    EMBEDDINGS_FOLDER = args.embedding_folder_path
    if not os.path.exists(args.checkpoint_folder_path):
        os.makedirs(args.checkpoint_folder_path)
    if not os.path.exists(args.csv_folder_path):
        os.makedirs(args.csv_folder_path)
    if not os.path.exists(args.embedding_folder_path):
        os.makedirs(args.embedding_folder_path)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    transforms = T.Compose([T.ToTensor(), T.Resize([IMAGE_SIZE, IMAGE_SIZE])])
    autoencoder = AE(
        IMAGE_SIZE,
        latent_dim=LATENT_DIM,
        ae_size=AE_SCALING_FACTOR,
        grayscale=GRAYSCALE,
    )
    api_key = open("/net/tscratch/people/plgmazurekagh/cyfrovet/wandb_api_key.txt", "r")
    key = api_key.read()
    api_key.close()
    os.environ["WANDB_API_KEY"] = key

    if "tensor" in DS_NAME:
        images = torch.load(f"{DATA_DIR}/{DS_NAME}/images.pt")
        targets = torch.load(f"{DATA_DIR}/{DS_NAME}/labels.pt")
        if EMBED:
            full_dataset = torch.utils.data.TensorDataset(images, targets)
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
        elif not TRAIN and EMBED:
            full_dataset, _ = ca.load_dataset(
                DS_NAME, merged=True, root=DATA_DIR, transform=transforms
            )

    if TRAIN:
        train(
            train_ds,
            test_ds,
            BATCH_SIZE,
            1000,
            autoencoder,
            CHECKPOINT_FOLDER,
            EXPERIMENT_NAME,
        )

    if EMBED:
        try:
            trained_model = AE.load_from_checkpoint(CHECKPOINT_PATH).to(DEVICE_GPU)
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
            try:
                images_embedded = torch.load(
                    f"{EMBEDDINGS_FOLDER}/{EXPERIMENT_NAME}_images_embedded.pt"
                )
                labels = torch.load(f"{EMBEDDINGS_FOLDER}/{EXPERIMENT_NAME}_labels.pt")
            except FileNotFoundError:
                print("No files found, check filepath or embed the dataset first.")
        images_embedded = images_embedded.cpu().numpy()
        labels = labels.cpu().numpy()
        print(f"Images embedded shape: {images_embedded.shape}")
        print(f"Images embedded type: {type(images_embedded)}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels type: {type(labels)}")
        compute_tree_metrics_embeddings(images_embedded, labels, 200, CSV_PATH)
