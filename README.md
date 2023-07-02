# Dataset assessment - part of Cyfrovet

A repository for a project looking for dataset classifiability by measuring its compressive properties via binary tree assessment.

Notes on usage:
  - Datasets contained in torchvision library (MNIST, FMNIST, CIFAR10) are downloaded automatically when requested
  - Remaining datasets were loaded as torch.tensors by custom dataloader - this solution was implemented due to the properties of HPC filesystem used for experiments. They will be made available soon.
  - In case user would like to load the data in classical ImageFolder please implement your own pipeline and modify the encoder_training_pipeline.py script

For more information about the code and data, as well as access to the datasets - please contact via s.mazurek@cyfronet.pl

```
@misc{mazurek2023assessing,
      title={Assessing Dataset Quality Through Decision Tree Characteristics in Autoencoder-Processed Spaces}, 
      author={Szymon Mazurek and Maciej Wielgosz},
      year={2023},
      eprint={2306.15392},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
