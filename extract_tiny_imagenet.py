from datasets import load_dataset
from datasets import Dataset
import os
import torchvision
import numpy as np
import torch
import time
import logging
from joblib import Parallel, delayed
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(0)
def extract_images_labels(dataset, n_th_split, save_path):
    to_tensor = torchvision.transforms.ToTensor()
    starting_indexes = np.linspace(0,len(dataset),10,dtype=int)
    start_index = starting_indexes[n_th_split-1]
    end_index = starting_indexes[n_th_split]

    for i in range(start_index,end_index):
        logger.info(f"Processed {i} th index")
        image = to_tensor(dataset[i]['image']).unsqueeze(0)
        label = torch.tensor(dataset[i]['label']).unsqueeze(0)
        if image.shape[1] != 3:
            continue
        try:
            images = torch.cat((images,image),dim=0)
            labels = torch.cat((labels,label),dim=0)
        except:
            images = image
            labels = label
    torch.save(images,os.path.join(save_path,f"images_part_{n_th_split}.pt"))
    torch.save(labels,os.path.join(save_path,f"labels_part_{n_th_split}.pt"))

def save_unified_ds(temp_path, save_dir, test=False):
    files_list = os.listdir(temp_path)
    labels_list = []
    images_list = []
    for file in files_list:
        if "labels" in file:
            labels_temp = torch.load(os.path.join(temp_path,file))
            labels_list.append(labels_temp)
        elif "images" in file:
            images_temp = torch.load(os.path.join(temp_path,file))
            images_list.append(images_temp)
    images_full = torch.cat(images_list)
    labels_full = torch.cat(labels_list)
    images_filename = "images_test.pt" if test else "images.pt"
    labels_filename = "labels_test.pt" if test else "labels.pt"
    torch.save(labels_full,os.path.join(save_dir,labels_filename))
    torch.save(images_full,os.path.join(save_dir,images_filename))

if __name__ == "__main__":   
    tiny_train = load_dataset('Maysee/tiny-imagenet', split='train',cache_dir = 'hf_datasets/')
    tiny_test = load_dataset('Maysee/tiny-imagenet', split='valid',cache_dir = 'hf_datasets/')
    temp_path = "/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/hf_temp_dir"
    temp_path_test = "/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/hf_temp_dir_test"
    target_path = "/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data/tiny_imagenet_tensor"
    
 
    Parallel(n_jobs=10,require="sharedmem")(delayed(extract_images_labels)(tiny_test,n,temp_path_test) for n in range(1,10))
    Parallel(n_jobs=10,require="sharedmem")(delayed(extract_images_labels)(tiny_train,n,temp_path) for n in range(1,10))
    
    save_unified_ds(temp_path,target_path)
    save_unified_ds(temp_path_test,target_path,test=True)
            
    
    


    