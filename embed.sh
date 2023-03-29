#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J dog_age
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=6GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:10:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plglaoisi23-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus-per-node=1
## Plik ze standardowym wyjściem
#SBATCH --output="/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/output/out/embed.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/output/err/embed.err"
#module add python/3.10
module add CUDA/11.7
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/age_recognition
cd /net/tscratch/people/plgmazurekagh/refactor/dataset_assessment
python embed_data.py --latent_dim 128 --checkpoint_root_path "/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/models/ae/best_models_dim_128" --embedding_path "/net/tscratch/people/plgmazurekagh/refactor/dataset_assessment/embeddings/ae/embedded_data_dim_128"
