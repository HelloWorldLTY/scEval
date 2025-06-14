# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

args = argparse.Namespace()

parser = argparse.ArgumentParser()
args.local_rank = -1
args.bin_num = 5
args.gene_num = 1000
args.epoch = 10
args.seed = 2021
args.batch_size = 3
args.learning_rate = 5e-5
args.grad_acc = 60
args.valid_every = 1
args.pos_embed = True
# args.data_path = "/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/Immune_ALL_human.h5ad"
# args.data_path = "/gpfs/gibbs/pi/zhao/tl688/datasets/MHSP_raw.loom"
args.data_path = "/gpfs/gibbs/pi/zhao/tl688/scGPT/spaital_mouse_slideseqv2.h5ad"

args.model_path = './panglao_pretrain.pth' 
args.ckpt_dir = './ckpts/' 
args.model_name = 'finetune' 

args

print(args)
os.environ["RANK"] = '0'
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = True

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name

ckpt_dir = args.ckpt_dir
device = torch.device("cuda")
seed_all(SEED)


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


check_ori = np.load("./gene2vec_16906.npy")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(2)

import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse

def preprocessing(adata_path, species="human"):
    panglao = sc.read_h5ad("./panglao_human.h5ad")
    data = sc.read(adata_path)
    
    if species == 'mouse':
        data.var_names = [i.upper() for i in data.var_names]
    
    counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
    ref = panglao.var_names.tolist()
    obj = data.var_names.tolist()

    for i in range(len(ref)):
        if ref[i] in obj:
            loc = obj.index(ref[i])
            counts[:,i] = data.X[:,loc]

    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = panglao.uns

    sc.pp.filter_cells(new, min_genes=200)
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    sc.pp.highly_variable_genes(new, n_top_genes=1000)
    new = new[:,new.var.highly_variable]
    
    return new
    

import time
t1 = time.time()
data = preprocessing(args.data_path, species="mouse")

data.obs['celltype'] = list(data.obs['cluster'])

# data.obs['batch'] = list(data.obs.sample_ID)
# data.obs['celltype'] = list(data.obs.final_annotation)

label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
#store the label dict and label for prediction
with open('label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('label', 'wb') as fp:
    pkl.dump(label, fp)
class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data = data.X

acc = []
f1 = []
f1w = []
pred_list = pd.Series(['un'] * data.shape[0])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,)

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING
)


path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])

for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None)

trigger_times = 0
max_acc = 0.0

for i in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        running_loss = 0.0
        predictions = []
        truths = []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = loss_fn(logits, labels_v)
                running_loss += loss.item()
                softmax = nn.Softmax(dim=-1)
                final_prob = softmax(logits)
                final = final_prob.argmax(dim=-1)
                final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                predictions.append(final)
                truths.append(labels_v)
            del data_v, labels_v, logits, final_prob, final
            # gather
            predictions = torch.cat(predictions, dim=0)
            truths = torch.cat(truths, dim=0)
            no_drop = predictions != -1
            predictions = np.array((predictions[no_drop]).cpu())
            truths = np.array((truths[no_drop]).cpu())
            cur_acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average='macro')
            val_loss = running_loss / index
            val_loss = val_loss
            if is_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
                print(confusion_matrix(truths, predictions))
                print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))
            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
#                 save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    break
    del predictions, truths
    
t2 = time.time()

