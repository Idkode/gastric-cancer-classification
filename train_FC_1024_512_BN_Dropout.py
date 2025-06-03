import pandas as pd
import numpy as np
from src.dataset import StomachCancerDataset
from dataloader import build_loader
import torch
from torchvision import transforms
from torchvision.models.resnet import resnet152, ResNet152_Weights
import torch.optim as optim
from torch import nn
import csv
import os
from src.train import train_model
from datetime import datetime

start = datetime.now()

classes = ['ADI',
           'DEB',
           'LYM',
           'MUC',
           'MUS',
           'NOR',
           'STR',
           'TUM']

label_mapper = {label: i for i, label in enumerate(classes)}

train_data = pd.read_csv('train.csv')
X_train = train_data['path'].values
y_train = train_data['label'].map(label_mapper).values

val_data = pd.read_csv('validation.csv')
X_val = val_data['path'].values
y_val = val_data['label'].map(label_mapper).values

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = StomachCancerDataset(X_train, y_train, transform)
val_dataset = StomachCancerDataset(X_val, y_val, transform)

batch_size = 64

trainloader = build_loader(train_dataset, batch_size, 3)
validation_loader = build_loader(val_dataset, batch_size, 3)

model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1024),
                         nn.BatchNorm1d(1024),
                         nn.ReLU(),
                         nn.Dropout(0.4),
                         nn.Linear(1024, 256),
                         nn.BatchNorm1d(256),
                         nn.ReLU(),
                         nn.Dropout(0.4),
                         nn.Linear(256, 8)
)
model_name = model.__class__.__name__

if not os.path.exists(name := 'results/' + model_name):
    os.mkdir(name)

files = os.listdir(name)
nome_arq = "fc_1024_512_bn_dropout"
path = name + '/' + nome_arq + '/'
if not os.path.exists(path):
    os.mkdir(path)

csv_file = (path + 'train_stats.csv')

with open(csv_file, 'a') as f:
    csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss'])

for parameter in model.layer3.parameters():
    parameter.requires_grad = True

for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.avgpool.parameters():
    parameter.requires_grad = True


update = [parameter for x in model.parameters() if x.requires_grad]

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=0.0001)
loss = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                 patience=6)

train_model(model,
            800,
            optimizer,
            trainloader,
            validation_loader,
            loss,
            path,
            csv_file,
            0,
            scheduler)

print(datetime.now())
print(f'Treino finalizado.\nTempo total: {(datetime.now() - start).total_seconds()} segundos')