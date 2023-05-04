import os
import torch
import numpy as np
import torchvision
from torchvision import transforms as T
import pytorch_lightning as pl
import torchvision.transforms.functional as TF

pl.seed_everything(42)

### Generate random dataset as lower level benchmark ###
n_samples = 10000

random_images = torch.randn([n_samples, 3, 128, 128], dtype=torch.float32)
random_labels = torch.randint(0, 10, [n_samples], dtype=torch.int64)
save_dir_random = "data/random_tensor_ds"
if not os.path.exists(save_dir_random):
    os.makedirs(save_dir_random)
print(f"Saving random dataset with {random_images.shape}, {random_labels.shape}")
torch.save(random_images, os.path.join(save_dir_random, "images.pt"))
torch.save(random_labels, os.path.join(save_dir_random, "labels.pt"))


### Augment 10 CIFAR10 images (one per class) into a more numerous dataset ###


def transform_images_tensor(batch_in) -> torch.Tensor:
    """Transforms a batch of torch.Tensor images (shape of [B,C,H,W])."""
    random_rot_angle = np.random.randint(-180, 180)
    hue_shift = np.random.uniform(-0.5, 0.5)
    sat_factor = np.random.uniform(1.1, 2)
    sharpness_factor = np.random.uniform(0.1, 3)
    contrast_factor = np.random.uniform(0.1, 3)
    x = TF.adjust_saturation(batch_in, sat_factor)
    x = TF.adjust_sharpness(x, sharpness_factor)
    x = TF.rotate(x, random_rot_angle)
    x = TF.adjust_hue(x, hue_shift)
    x = TF.adjust_contrast(x, contrast_factor)
    return x


def extract_balanced_classes(features, target, n_samples_per_class):
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
    return features[chosen_indexes], target[chosen_indexes], chosen_indexes


transforms = T.Compose([T.ToTensor(), T.Resize([128, 128])])

dataset_test = torchvision.datasets.CIFAR10(
    root="data/", train=False, transform=transforms
)
images, labels = zip(*dataset_test)
images = np.array(images)
labels = np.array(labels)

balanced_features, balanced_labels, indexes = extract_balanced_classes(
    images, labels, 1
)

balanced_features = torch.stack([img for img in balanced_features])
balanced_labels = torch.from_numpy(balanced_labels)

N_REPETITIONS = 1000

repeated_images = torch.cat(
    [transform_images_tensor(balanced_features) for _ in range(N_REPETITIONS)]
)
repeated_labels = torch.cat([balanced_labels for _ in range(N_REPETITIONS)])

save_dir_repeated = "data/repeated_cifar10_ds"
if not os.path.exists(save_dir_repeated):
    os.makedirs(save_dir_repeated)
print(
    f"Saving repeated CIFAR10 dataset with {repeated_images.shape}, {repeated_labels.shape}"
)
torch.save(repeated_images, os.path.join(save_dir_repeated, "images.pt"))
torch.save(repeated_labels, os.path.join(save_dir_repeated, "labels.pt"))
torch.save(indexes, os.path.join(save_dir_repeated, "original_test_ds_indexes.pt"))
