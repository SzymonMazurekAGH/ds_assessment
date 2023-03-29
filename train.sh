#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J dog_age
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=6GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:20:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plglaoisi23-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus-per-node=4
## SBATCH -C memfs
## Plik ze standardowym wyjściem
#SBATCH --output="/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/output/out/metrics_gpu.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/output/err/metrics_gpu.err"
#module add python/3.10
module add CUDA/11.7
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/age_recognition
cd /net/tscratch/people/plgmazurekagh/refactor/dataset_assessment
srun python train_vae.py --latent_dim 128
# --ds_name FMNIST &  python train_vae.py --ds_name MNIST &
# python train_vae.py --ds_name CIFAR10 &  python train_vae.py --ds_name pitbull_tensor &
# python train_vae.py --ds_name expert_tensor &  python train_vae.py --ds_name dog_breeds_tensor
# python extract_tiny_imagenet.py --n_split 0 & python extract_tiny_imagenet.py --n_split 1 &
# python extract_tiny_imagenet.py --n_split 2 & python extract_tiny_imagenet.py --n_split 2 &
# python extract_tiny_imagenet.py --n_split 3 & python extract_tiny_imagenet.py --n_split 4 &
# python extract_tiny_imagenet.py --n_split 5 & python extract_tiny_imagenet.py --n_split 6 &
# python extract_tiny_imagenet.py --n_split 7 & python extract_tiny_imagenet.py --n_split 8 &
# python extract_tiny_imagenet.py --n_split 9 
#python extract_tiny_imagenet.py 
#python compute_metrics.py
