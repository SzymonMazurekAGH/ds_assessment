import clustering_assessment as ca
import os
import json
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
import sklearn
results_dir = "./results"
if not os.path.exists(results_dir):
        os.mkdir(results_dir)
dataset_dir = "/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data"
dataset_names = ["dog_breeds"] #,  "pitbull_balanced","expert_ds",  "FMNIST","MNIST"


for dataset_name in dataset_names:

    transforms = T.Compose(
        [T.ToTensor()])
    images = torch.load(
    f"/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data/dog_breeds_tensor/images.pt"
    )
    labels = torch.load(
        f"/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data/dog_breeds_tensor/labels.pt"
    )
    results_path = os.path.join(results_dir, dataset_name)
    ds_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(ds_path):
        os.mkdir(ds_path)
    dataset, clas_to_idx = ca.load_dataset(dataset_name, merged=True, root=ds_path,transform=transforms)
    dataset = torch.utils.data.TensorDataset(images, labels)
    num_samples = int(round(len(dataset)/2))
  #  num_samples_list = [1000,6000,11000,18000,len(dataset)]
    fake_sample_percentages = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for fake_samples in fake_sample_percentages:
        num_fake_samples = int(round(fake_samples*num_samples))
        fake_ds = torchvision.datasets.FakeData(
            size=num_fake_samples, 
            image_size=(3, 224, 224), 
            num_classes=len(clas_to_idx), 
            transform=transforms
            )
        images_fake, labels_fake = ca.sample_from_dataset(fake_ds, num_samples=num_fake_samples, replacement=False)
        images,labels = ca.sample_from_dataset(dataset, num_samples=num_samples, replacement=False)
        images = np.concatenate((images, images_fake))
        labels = np.concatenate((labels, labels_fake))
        images, labels = sklearn.utils.shuffle(images, labels)
        embedding = ca.compute_umap(images)
        metric_dict = ca.compute_clustering_metrics(embedding, labels)
        json_path = os.path.join(results_path, f"metrics_umap_raw_{num_samples}_samples_{num_fake_samples}_fake.json")
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        with open(json_path, "w") as outfile:
            json.dump(metric_dict, outfile)
        outfile.close()
    
        ca.plot_embedding(
            embedding,
            labels,
            save_path=os.path.join(
                results_path, f"umap_raw_{num_samples}_samples_{num_fake_samples}_fake.png"
            ),
            class_to_idx=clas_to_idx
        )

    
