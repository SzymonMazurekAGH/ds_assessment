
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision.transforms as T
import torchvision
from ae import AE
import pytorch_lightning as pl
from vae import VAE
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import manifold
def reverse_dict(d):
    return {v: k for k, v in d.items()}

def compute_umap(data : np.array, n_neighbors:int =30, min_dist:float=0.1, n_components:int=2, metric:str="euclidean")->np.array:
    """Computes the UMAP embedding of the data. The data has to be in the shape (n_samples, n_features).
    Args:
        data (np.array): The data to be embedded.
        n_neighbors (int): The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100.
        min_dist (float): The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result in a more even dispersal of points. The value should be set relative to the ``spread`` value, which determines the scale at which embedded points will be spread out. default: 0.1
        n_components (int): The dimension of the space to embed into. default: 2
        metric (str): The metric to use to compute distances in high dimensional space. default: 'euclidean'
    Returns:
        embedding (np.array): The embedding of the data.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    embedding = reducer.fit_transform(data)
    return embedding

def compute_tsne(data: np.array, perplexity : int = None, n_iterations :int =  5000, seed :int = 0)->np.array:
    """Computes the t-SNE embedding of the data. The data has to be in the shape (n_samples, n_features).
    Arguments:
        data (np.array): The data to be embedded.
        perplexity (int): The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter. default: None
        n_iterations (int): Maximum number of iterations for the optimization. Should be at least 250. default: 5000
        seed (int): Random seed. default: 0
    Returns:
        embedding (np.array): The embedding of the data.   
    """
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=seed, perplexity=perplexity, n_iter=n_iterations)
    embedding = tsne.fit_transform(data)
    return embedding

def compute_clustering_metrics(embedding: np.array, labels)->dict[str, float]:
    """Computes the clustering metrics of the embedding. The computed metrics are Silhouette Coefficient, Calinski-Harabasz Index and Davies-Bouldin Index.
    Args:
        embedding (np.array): The embedding of the data.
        labels (np.array): The labels of the data.
    Returns:
        metirc_dict (dict): The dictionary containing the computed metrics.
    """
    metric_dict = {}
    metric_dict["Silhouette Coefficient"] = str(silhouette_score(embedding, labels))
    metric_dict["Calinski-Harabasz Index"] = str(calinski_harabasz_score(embedding, labels)/len(labels))
    metric_dict["Davies-Bouldin Index"] = str(davies_bouldin_score(embedding, labels))
    return metric_dict

def plot_embedding(embedding: np.array, labels: np.array, class_to_idx : str, save_path: str=None):
    """Plots the embedding of the data.
    Args:
        embedding (np.array): The embedding of the data.
        labels (np.array): The labels of the data.
        class_to_idx (dict): The dictionary containing the mapping between the class name and the class index.
        save_path (str): The path to save the plot. If None, the plot is not saved. default: None
    """
    fig,ax = plt.subplots(figsize=(15,10))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral',s=10)
    plt.legend(handles=scatter.legend_elements()[0], labels=reverse_dict(class_to_idx).values())
    if save_path is not None:
        plt.savefig(save_path)


def load_dataset(dataset_name : str = 'CIFAR10',merged : bool = True, root : str = './data', download : bool = True, transform : T.Compose = T.ToTensor()):
    """Loads given dataset from Torchvision.
    Args:
        dataset_name (str): The name of the dataset or path to image dataset. 
        Avaliable are: 'CIFAR10', 'MNIST', 'FMNIST'. If dataset_name
        is a path to an image dataset, the ImageFolder dataset will be returned.
    Returns:
        dataset (torch.utils.data.Dataset): The dataset.
        dataset_train, dataset_test (torch.utils.data.Dataset): The train and test dataset when merged is False.
    """
    if dataset_name == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        dataset_test = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        class_to_idx = dataset_train.class_to_idx
        if merged:
            dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
            return dataset,class_to_idx
        else:
            return dataset_train, dataset_test, class_to_idx
    elif dataset_name == 'MNIST':
        dataset_train = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
        dataset_test = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
        class_to_idx = dataset_train.class_to_idx
        if merged:
            dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
            return dataset, class_to_idx
        else:
            return dataset_train, dataset_test, class_to_idx
    elif dataset_name == 'FMNIST':
        dataset_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        dataset_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)
        class_to_idx = dataset_train.class_to_idx
        if merged:
            dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
            return dataset,class_to_idx
        else:
            return dataset_train, dataset_test, class_to_idx
    else:
        try:
            dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
            print(dataset)
            return dataset, dataset.class_to_idx
        except:
            print(f"Cannot find proper image dataset with path {dataset_name}")

def sample_from_dataset(dataset, num_samples : int = 1000, replacement : bool = False):
    """Samples a given number of samples from the dataset.
    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        num_samples (int): The number of samples to sample. default: 1000
        replacement (bool): Whether the samples are with or without replacement. default: False
    Returns:
        images (np.array): The images of the sampled samples.
        labels (np.array): The labels of the sampled samples.
    """
    random_samples = torch.utils.data.RandomSampler(dataset, num_samples=num_samples, replacement=replacement)
    images, labels = zip(*[dataset[i] for i in random_samples])
    images = np.array([x.numpy().flatten() for x in images])

    return np.array(images), np.array(labels)