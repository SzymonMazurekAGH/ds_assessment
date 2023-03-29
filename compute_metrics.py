import clustering_assessment as ca
import os
import json
import torch
from torchvision import transforms as T
from joblib import Parallel, delayed

results_dir = "./results"
if not os.path.exists(results_dir):
        os.mkdir(results_dir)
dataset_dir = "/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data"
dataset_names = ["dog_breeds"] #,  "pitbull_balanced","expert_ds",  "FMNIST","MNIST"

def parallel_func(num_samples):
    images,labels = ca.sample_from_dataset(dataset, num_samples=num_samples, replacement=False)
    embedding = ca.compute_umap(images)
    metric_dict = ca.compute_clustering_metrics(embedding, labels)
    json_path = os.path.join(results_path, f"metrics_umap_raw_{num_samples}_samples.json")
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    with open(json_path, "w") as outfile:
        json.dump(metric_dict, outfile)
    outfile.close()

    ca.plot_embedding(
        embedding,
        labels,
        save_path=os.path.join(
            results_path, f"umap_raw_{num_samples}_samples.png"
        ),
        class_to_idx=clas_to_idx
    )
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
   # num_samples = len(dataset)
    num_samples_list = [1000,6000,11000,18000,len(dataset)]
   # for num_samples in num_samples_list:
    Parallel(n_jobs=5,backend="multiprocessing")(delayed(parallel_func)(i) for i in num_samples_list)



