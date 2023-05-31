#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J ds_assessment
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 4GB na rdzeń)
#SBATCH --mem-per-cpu=100GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=16:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plglaoisi23-gpu-a100
## Specyfikacja partycji
#SBATCH --partition plgrid-gpu-a100
#SBATCH --gpus=0
## SBATCH -C memfs
## Plik ze standardowym wyjściem
#SBATCH --output="/net/tscratch/people/plgmazurekagh/cyfrovet/dataset_assessment/output/out/training1.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgmazurekagh/cyfrovet/dataset_assessment/output/err/training1.err"
#module add python/3.10
module add CUDA/11.7
conda activate /net/tscratch/people/plgmazurekagh/conda_envs/age_recognition
cd /net/tscratch/people/plgmazurekagh/cyfrovet/dataset_assessment
wandb artifact cache cleanup 1MB
for i in 10 25 50 
do
    echo "Starting for $i"

    # srun python encoder_training_pipeline.py --ds_name repeated_cifar10_tensor --latent_dim 256 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name repeated_cifar10_tensor --latent_dim 256 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name repeated_cifar10_tensor --latent_dim 256 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name repeated_cifar10_tensor --latent_dim 128 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name repeated_cifar10_tensor --latent_dim 128 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name repeated_cifar10_tensor --latent_dim 128 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for repeated_cifar10_tensor"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name dog_breeds_tensor --latent_dim 256 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name dog_breeds_tensor --latent_dim 256 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name dog_breeds_tensor --latent_dim 256 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name dog_breeds_tensor --latent_dim 128 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name dog_breeds_tensor --latent_dim 128 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name dog_breeds_tensor --latent_dim 128 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for dog_breeds_tensor"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name random_tensor_ds --latent_dim 256 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name random_tensor_ds --latent_dim 256 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name random_tensor_ds --latent_dim 256 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name random_tensor_ds --latent_dim 128 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name random_tensor_ds --latent_dim 128 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name random_tensor_ds --latent_dim 128 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for random_tensor_ds"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name MNIST --latent_dim 256 --scaling_factor 1.0 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name MNIST --latent_dim 256 --scaling_factor 0.5 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name MNIST --latent_dim 256 --scaling_factor 0.75 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name MNIST --latent_dim 128 --scaling_factor 1.0 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name MNIST --latent_dim 128 --scaling_factor 0.5 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name MNIST --latent_dim 128 --scaling_factor 0.75 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for MNIST"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name FMNIST --latent_dim 256 --scaling_factor 1.0 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name FMNIST --latent_dim 256 --scaling_factor 0.5 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name FMNIST --latent_dim 256 --scaling_factor 0.75 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name FMNIST --latent_dim 128 --scaling_factor 1.0 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name FMNIST --latent_dim 128 --scaling_factor 0.5 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name FMNIST --latent_dim 128 --scaling_factor 0.75 --grayscale --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for FMNIST"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name CIFAR10 --latent_dim 256 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name CIFAR10 --latent_dim 256 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name CIFAR10 --latent_dim 256 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name CIFAR10 --latent_dim 128 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name CIFAR10 --latent_dim 128 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name CIFAR10 --latent_dim 128 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for CIFAR10"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name expert_tensor --latent_dim 256 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name expert_tensor --latent_dim 256 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name expert_tensor --latent_dim 256 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name expert_tensor --latent_dim 128 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name expert_tensor --latent_dim 128 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name expert_tensor --latent_dim 128 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for expert_tensor"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name pitbull_tensor --latent_dim 256 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name pitbull_tensor --latent_dim 256 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name pitbull_tensor --latent_dim 256 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name pitbull_tensor --latent_dim 128 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name pitbull_tensor --latent_dim 128 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name pitbull_tensor --latent_dim 128 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for pitbull_tensor"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --ds_name tiny_imagenet_tensor --latent_dim 256 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name tiny_imagenet_tensor --latent_dim 256 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name tiny_imagenet_tensor --latent_dim 256 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name tiny_imagenet_tensor --latent_dim 128 --scaling_factor 1.0 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name tiny_imagenet_tensor --latent_dim 128 --scaling_factor 0.5 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # srun python encoder_training_pipeline.py --ds_name tiny_imagenet_tensor --latent_dim 128 --scaling_factor 0.75 --batch_size 256 --inference --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class  --embedding_folder_path embeddings/ae_1/  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for tiny_imagenet_tensor"
    # wandb artifact cache cleanup 1MB
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name pitbull_tensor --latent_dim 256 --scaling_factor 1.0 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for pitbull_tensor raw"
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name expert_tensor --latent_dim 256 --scaling_factor 0.5 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for expert_tensor raw"
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name MNIST --latent_dim 256 --scaling_factor 0.75 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for MNIST raw"
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name CIFAR10 --latent_dim 128 --scaling_factor 1.0 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for CIFAR10 raw"
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name FMNIST --latent_dim 128 --scaling_factor 0.5 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for FMNIST raw"
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name repeated_cifar10_tensor --latent_dim 128 --scaling_factor 1.0 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for repeated_cifar10_tensor raw"
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name random_tensor_ds --latent_dim 128 --scaling_factor 0.5 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for random_tensor_ds raw"
    # srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name dog_breeds_tensor --latent_dim 128 --scaling_factor 0.75 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    # echo "Done for dog_breeds_tensor raw"
    srun python encoder_training_pipeline.py --inference_raw --csv_folder_path csv_results/ae_1/csv_results_${i}_per_class --ds_name tiny_imagenet_tensor --latent_dim 128 --scaling_factor 0.75 --batch_size 256  --eval_samples_per_class $i --alternative_samples_per_class $i  --checkpoint_folder_path checkpoints_v1/
    echo "Done for tiny_imagenet_tensor raw"
done