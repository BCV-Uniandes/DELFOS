from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.efficient import ViT
#from pytorch_pretrained_vit import ViT
# Training settings
batch_size = 20
epochs = 50
lr = 1e-4
gamma = 1.7
seed = 200

def get_metrics(output,label):
    precisions = []
    recalls = []
    neg_vs = []
    specs = []
    accs = []
    FP_rs = []
    FN_rs = []

    FP = sum((output==1) & (label==0))
    TP = sum((output==1) & (label==1))
    FN = sum((output==0) & (label==1))
    TN = sum((output==0) & (label==0))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    neg_v = TN/(TN+FN)
    #acc = metrics.accuracy_score(label, output)
    FP_r = 1-specificity
    FN_r = 1-recall

    recalls.append(recall)
    specs.append(specificity)
    precisions.append(precision)
    neg_vs.append(neg_v)
    #accs.append(acc)
    FP_rs.append(FP_r)
    FN_rs.append(FN_r)

    print('Sensibilidad:{:.3f}'.format(np.mean(recalls)))
    print('Especificidad:{:.3f}'.format(np.mean(specs)))
    print('VPP:{:.3f}'.format(np.mean(precisions)))
    print('VPN:{:.3f}'.format(np.mean(neg_vs)))
    #print('Accuracy:{:.3f}'.format(np.mean(accs)))
    print('FPr:{:.3f}'.format(np.mean(FP_rs)))
    print('FNr:{:.3f}'.format(np.mean(FN_rs)))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'


train_transforms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(384),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

class DelfosDataset(Dataset):
    def __init__(self, file_dir, transform=None):
        self.file_list = glob.glob(os.path.join(file_dir,'*/*/*'))
        random.shuffle(self.file_list)
        self.transform = transform
        

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = 1 if 'CHD' in img_path else 0
        patient = img_path.split('/')[3]


        return img_transformed, label, patient, img_path
train_list = 'newdata/TRAIN/'
test_list = 'newdata/TEST/'
train_data = DelfosDataset(train_list, transform=train_transforms)
test_data = DelfosDataset(test_list, transform=test_transforms)
print(f"Train Data: {len(train_data)}")
print(f"Test Data: {len(test_data)}")
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = DataLoader(dataset = test_data, batch_size=1, shuffle=True)
#breakpoint()
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# model = ViT(
#     image_size = 224,
#     patch_size = 32,
#     num_classes = 2,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# ).to(device)


#model = ViT('B_16_imagenet1k', pretrained=True).to(device)

# loss function
#weights = torch.tensor([1., 1.2]).to(device)

criterion = nn.CrossEntropyLoss()#weight=weights)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label, patient, path in train_loader:
        data = data.to(device)
        label = label.to(device)
        #breakpoint()
        output = model(data)
        loss = criterion(output.softmax(dim=1), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label, patient, path in test_loader:
            data = data.to(device)
            label = label.to(device)
            val_output = model(data)

            val_loss = criterion(val_output.softmax(dim=1), label)
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(test_loader)
            epoch_val_loss += val_loss / len(test_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
df = pd.DataFrame(columns=['path','patient', 'prob_normal', 'label'])

with torch.no_grad():
    for data, label, patient, path in test_loader:
        data = data.to(device)
        label = label.to(device)
        val_output = model(data)
        prob_normal = val_output.softmax(dim=1)[0,0].to('cpu').numpy()
        label_ = label[0].to('cpu').numpy()
        df = df._append({'path':path, 'patient':patient[0],'prob_normal':prob_normal,'label':label_},ignore_index=True)
df.to_csv('predictions_weighted.csv',index=False)


threshold = df['prob_normal'].mean()
df['prob_normal']=(df['prob_normal']<threshold).astype('int')
patients = list(df.patient.unique())
all_patients = list(df.patient.values)
patient_dict = {}

for p in patients:
    idxs = [a for a,e in enumerate(all_patients) if p in e]
    patient_dict[p]=idxs

## ----- Evaluation by patient --------##
new_pred = pd.DataFrame(columns=['path','patient', 'prob_normal', 'label'])

for d in patient_dict:
    indexes = patient_dict[d]
    current = df.iloc[indexes]['prob_normal'].agg('min')
    new_pred = new_pred._append({'path':df.iloc[indexes[0]].path, 'patient':df.iloc[indexes[0]].patient,'prob_normal':current,'label':df.iloc[indexes[0]].label.item()},ignore_index=True)
get_metrics(new_pred.prob_normal.values,new_pred.label.values)