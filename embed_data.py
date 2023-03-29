import pytorch_lightning as pl
from vae import VAE
from ae import AE
import torch
import torchvision
from torchvision import transforms as T
import clustering_assessment as ca
import os
import logging
from torch.nn import functional as F
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--checkpoint_root_path",type=str)
parser.add_argument("--embedding_path",type=str)
args = parser.parse_args()
latent_dim = args.latent_dim
checkpoint_root_path = args.checkpoint_root_path
embedding_path = args.embedding_path
#dataset_names = ["CIFAR10", "pitbull_tensor", "expert_tensor", "tiny_imagenet_tensor"]
dataset_names = ["FMNIST","MNIST"]
dataset_root_path = '/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data'
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(0)
device = torch.device('cuda:0')
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path,exist_ok=True)
for name in dataset_names:
    
    if name == 'MNIST' or name == 'FMNIST':
        transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize(32)
        ])
       
    else:
        transforms = T.Compose(
        [
            T.ToTensor()
        ]
    )
    chkpt_path = os.path.join(checkpoint_root_path,name)
    chkpt_filename = os.listdir(chkpt_path)[0]
    full_chpkt_path = os.path.join(chkpt_path,chkpt_filename)
    print(full_chpkt_path)
    
    
    if 'tensor' in name:
        images = torch.load(
            os.path.join(dataset_root_path,name,"images.pt")
        )
        labels = torch.load(
            os.path.join(dataset_root_path,name,"labels.pt")
        )
        dataset = torch.utils.data.TensorDataset(images, labels)
    else:
        dataset, _ = ca.load_dataset(
            name,
            merged=True,
            root=dataset_root_path,
            transform=transforms
        )
    

    model = AE.load_from_checkpoint(full_chpkt_path).to(device)
    
    model.eval()
    
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False
    )

    print("Getting here!")
    for batch in loader:
        x,y = batch
        x = x.to(device)
        with torch.no_grad():
            loss = F.mse_loss(model(x), x, reduction='none').squeeze().mean(1).mean(1).mean(1)
            embedding = model.encoder(x)
        try:
            losses =  torch.cat([losses,loss])
            embedded_images = torch.cat([embedded_images,embedding])
            labels = torch.cat([labels,y])
        except:
            losses = loss
            embedded_images = embedding
            labels = y
    embedding_ds_path = os.path.join(embedding_path,name)
    print(f'Embedding shape {embedded_images.shape}')
    print(f'Loss shape {losses.shape}')
    print(f'Labels shape {labels.shape}')
    if not os.path.exists(embedding_ds_path):
        os.mkdir(embedding_ds_path)
    torch.save(embedded_images,os.path.join(embedding_ds_path,'embedded_images.pt'))
    torch.save(labels,os.path.join(embedding_ds_path,'labels.pt'))
    torch.save(losses,os.path.join(embedding_ds_path,'losses.pt'))
    del labels, embedded_images, embedding, losses,loss