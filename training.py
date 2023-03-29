from datetime import datetime
import torch
import torchvision
from torchvision import transforms as T
import torchmetrics
import sklearn
import numpy as np
import pytorch_lightning as pl
from ae import AE
from vae import VAE
import logging
#import wandb
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.callbacks import EarlyStopping
early_stopping = EarlyStopping('val_loss',patience=3,mode='min')
api_key = open("/net/tscratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/wandb_api_key.txt", "r")
key = api_key.read()
api_key.close()
os.environ["WANDB_API_KEY"] = key
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(0)
logger.info(f"Initializing")
# wandb.init(
#         project="classifier_lightining",
#         group="tests",
#     )
#wandb_logger = WandbLogger()
# transforms = T.Compose([T.ToTensor()])

# dataset = torchvision.datasets.ImageFolder(
#     root="/net/ascratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data/expert_ds",
#     transform=transforms,
# )
# dataset = torchvision.datasets.ImageFolder('/net/ascratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/data/expert_ds', transform=torchvision.transforms.ToTensor())
# images = torch.stack([img for img, _ in dataset], dim=0)
# labels = [label for _, label in dataset]
start = datetime.now()
images = torch.load('/net/ascratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/tensor_data/expert_tensor_ds/images.pt')
labels = torch.load('/net/ascratch/people/plgmazurekagh/refactor/Age_recognition_Cyfrovet/tensor_data/expert_tensor_ds/labels.pt')
tensor_dataset = torch.utils.data.TensorDataset(images,labels)
logger.info(f'Loading torch tensors took {datetime.now()-start}')
train_ds, test_ds = torch.utils.data.random_split(tensor_dataset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
)
# hard_example_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
val_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
)

# classes = np.unique(dataset.targets)
# targets = np.array(dataset.targets)
# class_weights = sklearn.utils.class_weight.compute_class_weight(
#     class_weight="balanced", classes=classes, y=targets
# )
# class_weights = torch.tensor(class_weights, dtype=torch.float)
logger.info("Created class weights")
class EfficientNet_training(pl.LightningModule):
    def __init__(self, n_classes, freeze_backbone,class_weights,lr) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Sequential(
                        torch.nn.Dropout(0.5),
                        torch.nn.Linear(2560, 1024),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(1024, 512),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(512, 128),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(128, 3),
                    )

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

            for layer_classifier in model.classifier.children():
                for param in layer_classifier.parameters():
                    param.requires_grad = True
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=n_classes)
        self.lr = lr
    def forward(self, x):
        return self.model(x)
    def training_step(self,batch, batch_idx):
        x,y = batch
        out = self(x)
        loss = self.loss(out,y)
        acc = self.accuracy(out,y)
        self.log("train_loss", loss, on_step=True, prog_bar=True,logger=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True,logger=True)
        return loss
    def validation_step(self,batch, batch_idx):
        x,y = batch
        out = self(x)
        loss = self.loss(out,y)
        acc = self.accuracy(out,y)
        self.log("val_loss", loss,on_step=True, prog_bar=True,logger=True)
        self.log("val_acc", acc,on_step=True, prog_bar=True,logger=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        return optimizer
    
model = EfficientNet_training(n_classes=3, freeze_backbone=True, class_weights=torch.tensor([0.9802, 0.7327, 1.6263],dtype=torch.float), lr=1e-4)
logger.info("Starting training")
trainer = pl.Trainer(max_epochs=40,log_every_n_steps=1,enable_progress_bar=True,num_nodes=1,accelerator='auto',devices="auto",callbacks=[early_stopping])
trainer.fit(model, train_loader,val_loader)
# model = torchvision.models.efficientnet_b7(
#     weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1
# )
# model.classifier = torch.nn.Sequential(
#     torch.nn.Dropout(0.5),
#     torch.nn.Linear(2560, 1024),
#     torch.nn.Dropout(0.3),
#     torch.nn.Linear(1024, 512),
#     torch.nn.Dropout(0.3),
#     torch.nn.Linear(512, 128),
#     torch.nn.Dropout(0.3),
#     torch.nn.Linear(128, 3),
# )

# for param in model.parameters():
#     param.requires_grad = False

# for layer_classifier in model.classifier.children():
#     for param in layer_classifier.parameters():
#         param.requires_grad = True


# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# accuracy = torchmetrics.Accuracy("multiclass", num_classes=3).to(device)
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
# model = model.to(device)

# for epoch in range(10):
#     epooch_loss = 0.0
#     epoch_val_loss = 0.0
#     model.train()
#     for idx, batch in enumerate(train_loader):
#         x, y = batch
#         x = x.to(device)
#         y = y.to(device)
#         y_hat = model(x)
#         try:
#             preds = torch.cat((preds, y_hat), 0)
#             ground_truth = torch.cat((ground_truth, y), 0)
#         except:
#             preds = y_hat
#             ground_truth = y

#         loss = criterion(y_hat, y)
#         epooch_loss += loss.detach().item()
#         loss.backward()
#         optimizer.zero_grad()
#         optimizer.step()
#     acc = accuracy(preds, ground_truth)
#     del preds, ground_truth
#     print(f"Epoch {epoch} train loss: {epooch_loss/len(train_loader)} train acc: {acc}")

#     with torch.no_grad():
#         model.eval()
#         for idx, batch in enumerate(val_loader):
#             x, y = batch
#             x = x.to(device)
#             y = y.to(device)
#             y_hat = model(x)
#             try:
#                 val_preds = torch.cat((val_preds, y_hat), 0)
#                 val_ground_truth = torch.cat((val_ground_truth, y), 0)
#             except:
#                 val_preds = y_hat
#                 val_ground_truth = y
#             loss = criterion(y_hat, y)
#             epoch_val_loss += loss.detach().item()
#         val_acc = accuracy(val_preds, val_ground_truth)
#         del val_preds, val_ground_truth
#         print(
#             f"Epoch {epoch} val loss: {epoch_val_loss/len(val_loader)} val acc: {val_acc}"
#         )
