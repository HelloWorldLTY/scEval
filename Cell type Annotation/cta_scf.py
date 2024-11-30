import sys 
import numpy as np
import torch
from torch import nn
sys.path.append("../model/") # path to this folder
from load import *

class LinearProbingClassifier(nn.Module):

    def __init__(self, ckpt_path,frozenmore=True):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.frozenmore = frozenmore

    def build(self, n_class=12):
        model,model_config = load_model_frommmf(self.ckpt_path)
        self.token_emb = model.token_emb
        self.pos_emb = model.pos_emb
        self.encoder = model.encoder
        
        if self.frozenmore:
            for _,p in self.token_emb.named_parameters():
                p.requires_grad = False
            for _,p in self.pos_emb.named_parameters():
                p.requires_grad = False
            print('self.pos_emb and self.token_emb also frozen')
        
        for na, param in self.encoder.named_parameters():
            param.requires_grad = False
        for na, param in self.encoder.transformer_encoder[-2].named_parameters():
            print('self.encoder.transformer_encoder ',na,' have grad')
            param.requires_grad = True


        self.fc1 = nn.Sequential(
        nn.Linear(model_config['encoder']['hidden_dim'], 256),
        nn.ReLU(),
        nn.Linear(256, n_class)  # ['n_class']
        ) 
        self.norm = torch.nn.BatchNorm1d(model_config['encoder']['hidden_dim'], affine=False, eps=1e-6)
        self.model_config = model_config
        
    def forward(self, x, *args, **kwargs):
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.model_config['pad_token_id'])
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.model_config['pad_token_id'])
        
        x = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        x += position_emb

        logits = self.encoder(x,x_padding)

        # mlp
        logits, _ = torch.max(logits, dim=1)  # b,dim

        logits = self.norm(logits)
        logits = self.fc1(logits)

        return logits

# !nvidia-smi





# sample_list['targets'].shape

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

class LitAutoClassifier(L.LightningModule):
    def __init__(self, encoder,weight):
        super().__init__()
        self.class_pred = encoder
        self.loss = nn.CrossEntropyLoss(weight = weight)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        pred = self.class_pred(x)
        loss = self.loss(pred,y)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        pred = self.class_pred(x)
        loss = self.loss(pred,y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

import pandas as pd

df = pd.read_csv("./OS_scRNA_gene_index.19264.tsv", sep="\t")

import scanpy as sc

# data_name = "HumanPBMC_raw"
# data_type = 'inner'
# adata = sc.read(f"/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/{data_name}.h5ad", compress="X")


data_type = 'cross'
data_name = 'aizarani_liver'
adata =  sc.read(f"/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/{data_name}.h5ad", compress="X")
data_name = 'macParland_liver'
adata_test = sc.read(f"/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/{data_name}.h5ad", compress="X")

# adata = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/spaital_mouse_slideseqv2.h5ad")
# adata

df['gene_name'].values.astype('str')
obs_data = adata.obs.copy()

adata_emp = sc.AnnData(np.zeros((1,19264)))
adata_emp.var_names = df['gene_name'].values.astype('str')

adata = sc.concat([adata_emp,adata], keys=[0,1])

adata_new = adata[1:,:]
adata_new.obs = obs_data

adata_new.obs

# adata = sc.read(f"/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/{data_name}.h5ad", compress="X")
# adata_new.obs = adata.obs



if data_type == 'cross':
    adata_emp = sc.AnnData(np.zeros((1,19264)))
    adata_emp.var_names = df['gene_name'].values.astype('str')

    adata = sc.concat([adata_emp,adata_test], keys=[0,1])

    adata_test_new = adata[1:,:]
    adata_test = sc.read(f"/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/{data_name}.h5ad", compress="X")
    adata_test_new.obs = adata_test.obs
    try:
        adata_test_new.obs['celltype'] = list(adata_test_new.obs['Celltype'])
    except:
        adata_test_new.obs['celltype'] = list(adata_test_new.obs['cellTypes'])
    print(adata_test_new.obs)



from sklearn.model_selection import train_test_split

if data_type == 'cross':
    train_label, val_label = train_test_split(adata_new.obs_names, test_size=0.33, shuffle=True, random_state=42
    )

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(adata_new.obs['celltype'])
    adata_train = adata_new[train_label,:]
    adata_val = adata_new[val_label,:]
    adata_test = adata_test_new
else:
    train_label, test_label = train_test_split(adata_new.obs_names, test_size=0.33, shuffle=True, random_state=42
    )

    # ## specific for aorta
    # train_label, test_label = train_test_split(adata.obs_names, test_size=0.20, random_state=2023)
    adata_train = adata_new[train_label,:]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(adata_train.obs['celltype'])

    train_label, val_label = train_test_split(adata_train.obs_names, test_size=0.33, shuffle=True, random_state=42
    )

    adata_train = adata_new[train_label,:]
    adata_val = adata_new[val_label,:]
    adata_test = adata_new[test_label,:]

finetune_model = LinearProbingClassifier(ckpt_path='./models/models.ckpt')
finetune_model.build(n_class = len(le.classes_))
finetune_model = finetune_model.cuda()
# finetune_model(sample_list)

len(le.classes_)

train_label = le.transform(adata_train.obs['celltype'])
val_label = le.transform(adata_val.obs['celltype'])

import collections
label_dict = dict(collections.Counter(train_label))
weight_list = torch.zeros((1,len(le.classes_)))[0]
for i in label_dict.keys():
    try:
        weight_list[i] = 1/ (label_dict[i])
    except:
        weight_list[i] = 0

X_train = torch.FloatTensor(adata_train.X.toarray())

train_label = torch.FloatTensor(train_label).long()

dataset = torch.utils.data.TensorDataset(X_train, train_label)

batch_size = 2

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)



X_test = torch.FloatTensor(adata_val.X.toarray())

val_label = torch.FloatTensor(val_label).long()

dataset = torch.utils.data.TensorDataset(X_test, val_label)

batch_size = 2

val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=2, drop_last=True)

# model
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
autoencoder = LitAutoClassifier(finetune_model,weight=weight_list)

# train model
trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")],max_epochs=10)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)

torch.cuda.empty_cache()

# model
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
autoencoder = LitAutoClassifier(finetune_model,weight=weight_list)
autoencoder = LitAutoClassifier.load_from_checkpoint("/gpfs/gibbs/pi/zhao/tl688/scf_new/scFoundation/model/lightning_logs/version_37682121/checkpoints/epoch=5-step=18414.ckpt", encoder = finetune_model,weight=weight_list)


def inference_model(model, data):
    model = model.cuda()
    pred_list = []
    with torch.no_grad():
        for item in range(0,len(data),2):
            print("finish")
            data_new = torch.FloatTensor(data[item:item+2,:]).cuda()
            logits = model(data_new)
            _,pred =  torch.max(logits, 1)
            pred_list += list(pred.cpu().numpy())
            del logits
            del pred
            del data_new
            torch.cuda.empty_cache()
    return pred_list

model = autoencoder.class_pred.eval()
input_data = adata_test.X.toarray()
pred_out = inference_model(model, input_data)

pred_label = le.inverse_transform(pred_out) 