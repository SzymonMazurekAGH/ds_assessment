#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J ds_assessment
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=6GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plglaoisi23-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus=1
## SBATCH -C memfs
## Plik ze standardowym wyjściem
#SBATCH --output="/net/tscratch/people/plgmazurekagh/cyfrovet/dataset_assessment/output/out/training.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgmazurekagh/cyfrovet/dataset_assessment/output/err/training.err"
#module add python/3.10
module add CUDA/11.7
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/age_recognition
cd /net/tscratch/people/plgmazurekagh/cyfrovet/dataset_assessment

srun python ae_training_pipeline.py --embed --inference --ds_name MNIST  --latent_dim 256 --ae_scaling_factor 1.0 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name MNIST  --latent_dim 256 --ae_scaling_factor 0.5 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name MNIST  --latent_dim 256 --ae_scaling_factor 0.75 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name MNIST  --latent_dim 128 --ae_scaling_factor 1.0 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name MNIST  --latent_dim 128 --ae_scaling_factor 0.5 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name MNIST  --latent_dim 128 --ae_scaling_factor 0.75 --grayscale --batch_size 256

srun python ae_training_pipeline.py --embed --inference --ds_name FMNIST  --latent_dim 256 --ae_scaling_factor 1.0 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name FMNIST  --latent_dim 256 --ae_scaling_factor 0.5 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name FMNIST  --latent_dim 256 --ae_scaling_factor 0.75 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name FMNIST  --latent_dim 128 --ae_scaling_factor 1.0 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name FMNIST  --latent_dim 128 --ae_scaling_factor 0.5 --grayscale --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name FMNIST  --latent_dim 128 --ae_scaling_factor 0.75 --grayscale --batch_size 256

srun python ae_training_pipeline.py --embed --inference --ds_name CIFAR10  --latent_dim 256 --ae_scaling_factor 1.0 --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name CIFAR10  --latent_dim 256 --ae_scaling_factor 0.5 --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name CIFAR10  --latent_dim 256 --ae_scaling_factor 0.75 --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name CIFAR10  --latent_dim 128 --ae_scaling_factor 1.0 --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name CIFAR10  --latent_dim 128 --ae_scaling_factor 0.5 --batch_size 256
srun python ae_training_pipeline.py --embed --inference --ds_name CIFAR10  --latent_dim 128 --ae_scaling_factor 0.75 --batch_size 256



