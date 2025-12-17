import scanpy as sc
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from layers.utils import *
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import json

import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

from model import Finetune_Cell_FM

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from functools import partial
from scipy.sparse import csr_matrix as csm

import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description="A program that processes files.")
parser.add_argument("--filename", help="The name of the file to process")
args = parser.parse_args()

class Config_80M:
    def __init__(self):
        self.start_lr = 1e-3
        self.max_lr = 1e-2
        self.min_lr = 5e-4
        self.factor = 5
        self.lora = 0
        self.alpha = 0
        self.lamb = 10
        self.nb_features = 256
        self.nonz_len = 2048
        self.mask_len = 2048
        self.filt_len = 277
        self.dropout = 0.1

        self.enc_dims = 1536
        self.enc_nlayers = 2
        self.enc_num_heads = 48
        self.enc_dropout = 0.1

        self.dec_dims = 512
        self.dec_nlayers = 6
        self.dec_num_heads = 16
        self.dec_dropout = 0.1

        self.temp = 0.2
        self.eps = 1e-2
        self.recompute = True
        self.sim = 0.8
        self.add_zero = True
        self.pad_zero = True
        self.label = False

        self.num_cls = 80
        self.platforms = 27
        self.ttl_step = 1e5

# read data
def read_h5ad(path, test_rate=0.1):
    '''
    load dataset and split train and valid
    '''
    # read anndata
    suffix = path.split('.')[-1]
    if suffix == 'h5ad':
        adata = sc.read_h5ad(path)
    else:
        adata = sc.read_10x_h5(path)
    print('origin shape:', adata.shape)

    data = adata.X.astype(np.float32)
    T = adata.X.sum(1)
    data = csm(np.round(data / np.maximum(1, T / 1e5, dtype=np.float32)))
    data.eliminate_zeros()
    adata.X = data
    return adata


def map_gene_list(gene_list, gene_info):
    """
    将一个基因名列表标准化，映射到目标 geneset 中。

    参数：
        gene_list: list[str]，输入的原始基因名列表（可能包含 alias/旧名）
        geneset: set 或 list[str]，目标标准基因集合（如 HGNC 批准名称）
        map_dict: dict, alias → approved name 的映射字典

    返回：
        mapped_genes: list[str]，成功映射到 geneset 的标准名称
        failed_genes: list[str]，无法映射到 geneset 的原始名称
    """
    geneset = {j: i + 1 for i, j in enumerate(gene_info.index)}
    gene_list = list(map(str, gene_list))  
    genemap = {j: i + 1 for i, j in enumerate(gene_info.index)}
    hgcn = pd.read_csv(r'csv/updated_hgcn.tsv', index_col=1, header=0,
                       sep='\t')
    hgcn = hgcn[hgcn['Status'] == 'Approved']
    map_dict = {}
    alias = hgcn['Alias symbols']
    prev = hgcn['Previous symbols']
    for i in hgcn.index:
        if alias.loc[i] is not np.nan:
            for j in alias.loc[i].split(', '):
                if j not in hgcn.index:
                    map_dict[j] = i
    for i in hgcn.index:
        if prev.loc[i] is not np.nan:
            for j in prev.loc[i].split(', '):
                if j not in hgcn.index:
                    map_dict[j] = i
    mapped_genes = []
    failed_genes = []

    for gene in gene_list:
        if gene in geneset:
            mapped_genes.append(gene)
        elif gene in map_dict and map_dict[gene] in geneset:
            mapped_genes.append(map_dict[gene])
        else:
            failed_genes.append(gene)

    return mapped_genes, failed_genes


class SCrna():
    def __init__(self, adata, mode="train"):
        if mode == "train":
            adata = adata[adata.obs.train == 0]
        elif mode == 'val':
            adata = adata[adata.obs.train == 1]
        else:
            adata = adata[adata.obs.train == 2]
        #adata.var_names = [i.upper() for i in adata.var_names]
        adata.var_names_make_unique()
        self.gene_info = pd.read_csv("../csv/expand_gene_info.csv",
            index_col=0, header=0
        )
        self.geneset = {j: i + 1 for i, j in enumerate(self.gene_info.index)}

        selected_genes_list = adata.var_names.tolist()
        selected_genes, _ = map_gene_list(selected_genes_list, self.gene_info)
        selected_genes = selected_genes[:2048]  # 2048
        pad_len = 2048 - len(selected_genes)
        pad_genes = [f'__pad_{i}__' for i in range(pad_len)]
        self.full_gene_list = selected_genes + pad_genes
        self.selected_gene_len = len(selected_genes)

        used_genes = [g for g in selected_genes if g in adata.var_names]
        pad_num = 2048 - len(used_genes)
        # print(f"[Info] Selected: {len(selected_genes)}, Found: {len(used_genes)}, Padding: {pad_num}")

        X = adata[:, used_genes].X.toarray().astype(np.int32)
        # pad_X = np.zeros((X.shape[0], pad_num), dtype=np.int32)
        # X_padded = np.concatenate([X, pad_X], axis=1)
        try:
            adata.obs['celltype'] = adata.obs['cell_type']
        except:
            adata.obs['celltype'] = 'nan'
        
        ####### add cell type information ######
        if "celltype_id" not in adata.obs.columns:
            celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
            adata.obs["celltype_id"] = celltype_id_labels
        
        celltypes_labels = adata.obs["celltype_id"].values
        self.celltypes = adata.obs["celltype"].unique()
        self.id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        
        celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
        self.celltypes_labels = np.array(celltypes_labels)
        
        ####### add batch information ######
        if "batch_id" not in adata.obs.columns:
            # batch_id_labels = adata.obs["donor_id"].astype("category").cat.codes.values
            batch_id_labels = adata.obs.get("str_batch", "0").astype("category").cat.codes.values
            adata.obs["batch_id"] = batch_id_labels
        
        batch_ids = adata.obs["batch_id"].values
        batch_ids = adata.obs["batch_id"].tolist()
        self.batch_ids = np.array(batch_ids)
        
        
        ####### add feat information (numeric)######
        feat = adata.obs['feat'].tolist()
        self.feat = np.array(feat)

        from anndata import AnnData
        new_adata = AnnData(X)
        # new_adata = AnnData(X_padded)
        new_adata.obs = adata.obs.copy()
        # new_adata.var_names = self.full_gene_list
        new_adata.var_names = used_genes

        self.adata = new_adata
        self.gene = np.array([
            self.geneset.get(g, 0) for g in self.full_gene_list
        ], dtype=np.int32)

        self.T = np.asarray(self.adata.X.sum(1)).ravel()
        self.data = self.adata.X.astype(np.int32)
        # print(f"Use adata shape: {self.adata.shape}")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        data = np.asarray(self.data[idx], dtype=np.float32)
        T = np.asarray(self.T[idx], dtype=np.float32)
        gene = np.asarray(self.gene, dtype=np.int32)
        celltype_label = self.celltypes_labels[idx]  # single cell label
        batch_id = self.batch_ids[idx] # single cell batch id
        feat = self.feat[idx]
        return data, gene, T, celltype_label, batch_id, feat


class TestSCrna():
    def __init__(self, adata, mode="test", mask_rate=0.2, prep=True):
        if mode == "train":
            adata = adata[adata.obs.train == 0]
        elif mode == 'val':
            adata = adata[adata.obs.train == 1]
        else:
            adata = adata[adata.obs.train == 2]

        self.gene_info = pd.read_csv(
            r'csv/expand_gene_info.csv',
            index_col=0, header=0
        )
        self.geneset = {j: i + 1 for i, j in enumerate(self.gene_info.index)}


        selected_genes_list = selected_genes_list = adata.var_names.tolist()
        selected_genes, _ = map_gene_list(selected_genes_list, self.gene_info)
        selected_genes = selected_genes[:2048]  # 2048
        pad_len = 2048 - len(selected_genes)
        pad_genes = [f'__pad_{i}__' for i in range(pad_len)]
        self.full_gene_list = selected_genes + pad_genes
        # padding zero matrix
        used_genes = [g for g in selected_genes if g in adata.var_names]
        pad_num = 2048 - len(used_genes)
        print(f"[Info] Selected: {len(selected_genes)}, Found: {len(used_genes)}, Padding: {pad_num}")

        X = adata[:, used_genes].X.toarray().astype(np.int32)

        from anndata import AnnData
        new_adata = AnnData(X)
        new_adata.obs = adata.obs.copy()
        new_adata.var_names = used_genes


        self.gene = np.array([
            self.geneset.get(g, 0) for g in self.full_gene_list
        ], dtype=np.int32)

        if prep:
            new_adata.layers['x_normed'] = sc.pp.normalize_total(new_adata, target_sum=1e4, inplace=False)['X']
            new_adata.layers['x_log1p'] = new_adata.layers['x_normed']
            sc.pp.log1p(new_adata, layer='x_log1p')

        self.adata = new_adata
        if prep:
            self.data = self.adata.layers['x_log1p'].A.astype(np.float32)
        else:
            self.data = self.adata.X.astype(np.int32)

        self.label = self.data.copy()  # shape [N, G]
        # padding to [N, 2048]
        N, G = self.label.shape
        self.label_padded = np.zeros((N, 2048), dtype=np.float32)
        self.label_padded[:, :G] = self.label

        self.mask_rate = mask_rate
        self.mask = np.zeros((N, G), dtype=bool)

        # Randomly shuffle G positions for each sample
        rand_idx = np.argsort(np.random.rand(N, G), axis=1)
        # For each row, set the first mask_len positions to True
        mask_len = int(G * self.mask_rate)
        self.mask[np.arange(N)[:, None], rand_idx[:, :mask_len]] = True
        # Construct masked_X
        self.masked_X = self.adata.copy()
        self.masked_X[self.mask] = 0.0
        self.mask_padded = np.zeros((N, 2048), dtype=bool)
        self.mask_padded[:, :G] = self.mask

    def __len__(self):
        return len(self.masked_X)

    def __getitem__(self, idx):
        x = self.masked_X[idx].reshape(-1).astype(np.float32)  # masked input
        y = self.label_padded[idx].reshape(-1).astype(np.float32)  # ground truth
        m = self.mask_padded[idx].reshape(-1).astype(bool)  # mask area
        g = self.gene
        return x, g, y, m

class Prepare():
    def __init__(
            self, pad_len, pad=2, zero_len=None,
            mask_ratio=0.3, dw=True,
            uw=False, random=False, cut=None
    ):
        self.dw = dw
        self.uw = uw
        self.zero_len = zero_len
        self.n_genes = 27855
        self.mask_ratio = mask_ratio
        self.pad_len = pad_len
        self.bern = partial(np.random.binomial, p=0.5)
        self.beta = partial(np.random.beta, a=2, b=2)
        self.bino = np.random.binomial
        self.pad = pad
        self.cut = min(pad_len, (cut or pad_len))
        self.random = random
        self.empty_gene = np.zeros(self.n_genes + 1, np.float32)

    def bayes(self, raw_nzdata, T):
        """
        Perform Bayesian sampling on raw non-zero expression data
        to simulate weighted down-sampling
        """
        S = T.copy()
        dw_nzdata = raw_nzdata.copy()
        gamma = self.bern(n=1)
        if self.uw:
            T = T * 5
        elif gamma == 1 and self.dw:
            p = self.beta(size=1)
            n = dw_nzdata.astype(np.int32)
            dw_nzdata = np.maximum(self.bino(n, p.repeat(len(n), 0)), 1)
            S = dw_nzdata.sum()
        return raw_nzdata, dw_nzdata, S, T

    def normalize(self, data, read):
        data = np.log1p(data / read * 1e4).astype(np.float32)
        return data, read

    def zero_idx(self, data):
        """
        Construct zero mask after padding, used to
        indicate positions of zero padding
        """
        seq_len = len(data)
        one = (data != 0).astype(np.float32)
        zero = np.zeros(self.pad_len - seq_len, np.float32)
        zero_mask = np.concatenate([one, zero])
        return data, zero_mask

    def zero_mask(self, seq_len):
        """Construct mask vector corresponding to zero padding positions"""
        zero_len = self.pad_len - seq_len
        unmasked = np.ones(zero_len, np.float32)
        pad = np.zeros(zero_len, np.float32)
        pad = np.stack([unmasked, pad], 1)

        l = int(self.mask_ratio * min(seq_len, zero_len))
        mask = np.random.choice(np.arange(zero_len), l, replace=False)
        zero_mask = np.zeros(zero_len, np.float32)
        if not self.random:
            zero_mask[mask[:int(0.8 * l)]] = 1
        else:
            zero_mask[mask] = 1
        pad[mask] = 0
        return pad, zero_mask

    def mask(self, dw_nzdata):
        """Apply masking on non-zero expression data"""
        seq_len = len(dw_nzdata)
        l = int(self.mask_ratio * seq_len)
        mask = np.arange(seq_len)
        unmasked = np.ones_like(dw_nzdata)
        dw_nzdata = np.stack([unmasked, dw_nzdata], 1)
        if l > 0:
            mask = np.random.permutation(seq_len)[:l]
            if not self.random:
                dw_nzdata[mask[:int(0.8 * l)]] = 0
            else:
                dw_nzdata[mask] = 0
        mask_gene = np.zeros(seq_len, np.float32)
        mask_gene[mask] = 1
        return dw_nzdata, mask_gene

    def pad_gene(self, data, z_data):
        return np.concatenate((data, z_data))

    def pad_zero(self, data):
        shape = (self.pad_len - data.shape[0], *data.shape[1:])
        pad = np.zeros(shape, data.dtype)
        data = np.concatenate((data, pad), 0)
        return data

    def seperate(self, raw_data):
        """Separate non-zero and zero expression positions"""
        nonz = raw_data.nonzero()[0]
        zero = np.where(raw_data == 0)[0]
        return raw_data, nonz, zero

    def compress(self, data, idx):
        return data, data[idx], idx

    def sample(self, data, nonz, zero, freq=None):
        """Sample nonz and zero to construct input and mask candidates"""
        cutted = np.array([])
        if len(nonz) > self.cut:
            w = np.log1p(data[nonz])
            w = w / w.sum()
            order = np.random.choice(np.arange(len(nonz)), len(nonz), replace=False, p=w)
            order = nonz[order]
            nonz = np.sort(order[:self.cut])
            cutted = np.sort(order[self.cut:])

        w = None
        l = self.zero_len or (self.pad_len - len(nonz))
        l = min(len(zero), l)
        if freq is not None:
            w = freq[zero]
            ttl = w.sum()
            if ttl > 0:
                w = w / ttl

        if len(zero) >= l and l > 0:
            z_sample = np.random.choice(zero, l, replace=False, p=w)
        else:
            z_sample = np.array([], dtype=np.int32)

        seq_len = len(nonz)
        return data, nonz, cutted, z_sample, seq_len

    def cat_st(self, S, T):
        ST_feat = np.log1p(np.array([S, T]).astype(np.float32) / 1000)
        return ST_feat

    def attn_mask(self, seq_len):
        mask_row = np.zeros(self.pad_len + self.pad)
        mask_row[:seq_len + self.pad] = 1
        return mask_row.astype(np.float32)

def build_dataset(dataset, prep, batch_size, pad_zero=True, drop=True, shuffle=True):
    def collate_fn(samples):
        raw_nzdata_batch = []
        dw_nzdata_batch = []
        ST_feat_batch = []
        nonz_gene_batch = []
        mask_gene_batch = []
        zero_idx_batch = []
        celltype_label_batch = []
        batch_id_batch = []
        feat_batch = []

        for data, gene, T, celltype_label, batch_id, feat in samples:

            raw_data, nonz, zero = prep.seperate(data)

            data, nonz, cuted, z_sample, seq_len = prep.sample(raw_data, nonz, zero)

            raw_data, raw_nzdata, nonz = prep.compress(raw_data, nonz)
            gene, nonz_gene, _ = prep.compress(gene, nonz)

            raw_nzdata, dw_nzdata, S, T = prep.bayes(raw_nzdata, T)
            dw_nzdata, S = prep.normalize(dw_nzdata, S)
            raw_nzdata, T = prep.normalize(raw_nzdata, T)
            ST_feat = prep.cat_st(S, T)

            if pad_zero:
                zero_idx = prep.attn_mask(seq_len)
                dw_nzdata, mask_gene = prep.mask(dw_nzdata)

                raw_nzdata = prep.pad_zero(raw_nzdata)
                dw_nzdata = prep.pad_zero(dw_nzdata)
                nonz_gene = prep.pad_zero(nonz_gene)
                mask_gene = prep.pad_zero(mask_gene)
            else:
                dw_nzdata, zero_idx = prep.zero_idx(dw_nzdata)
                dw_nzdata, mask_gene = prep.mask(dw_nzdata)
                zero_pad, zero_mask = prep.zero_mask(seq_len)

                gene, z_gene, z_sample = prep.compress(gene, z_sample) 
                nonz_gene = prep.pad_gene(nonz_gene, z_gene)
                raw_nzdata = prep.pad_zero(raw_nzdata)
                dw_nzdata = prep.pad_gene(dw_nzdata, zero_pad)
                mask_gene = prep.pad_gene(mask_gene, zero_mask)

            raw_nzdata_batch.append(torch.tensor(raw_nzdata, dtype=torch.float32))
            dw_nzdata_batch.append(torch.tensor(dw_nzdata, dtype=torch.float32))
            ST_feat_batch.append(torch.tensor(ST_feat, dtype=torch.float32))
            nonz_gene_batch.append(torch.tensor(nonz_gene, dtype=torch.int32))
            mask_gene_batch.append(torch.tensor(mask_gene, dtype=torch.float32))
            zero_idx_batch.append(torch.tensor(zero_idx, dtype=torch.float32))
            celltype_label_batch.append(torch.tensor(celltype_label, dtype=torch.long))
            batch_id_batch.append(torch.tensor(batch_id, dtype=torch.long))
            feat_batch.append(torch.tensor(feat, dtype=torch.float32))

        return {
            'raw_nzdata': torch.stack(raw_nzdata_batch),
            'dw_nzdata': torch.stack(dw_nzdata_batch),
            'ST_feat': torch.stack(ST_feat_batch),
            'nonz_gene': torch.stack(nonz_gene_batch),
            'mask_gene': torch.stack(mask_gene_batch),
            'zero_idx': torch.stack(zero_idx_batch),
            'celltype_label': torch.stack(celltype_label_batch),
            'batch_id': torch.stack(batch_id_batch),
            'feat': torch.stack(feat_batch),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop,
        collate_fn=collate_fn
    )

def build_testdataset(dataset, prep, batch_size, drop=True, shuffle=True):
    def collate_fn(samples):
        dw_nzdata_batch = []
        nonz_gene_batch = []
        zero_idx_batch = []
        label_batch = []
        mask_batch = []

        for data, gene, label, msk in samples:
            
            raw_data, nonz, zero = prep.seperate(data)

            data, nonz, cuted, z_sample, seq_len = prep.sample(raw_data, nonz, zero)

            data, dw_nzdata, nonz = prep.compress(data, nonz)
            gene, nonz_gene, _ = prep.compress(gene, nonz)

            zero_idx = prep.attn_mask(seq_len)

            dw_nzdata = prep.pad_zero(dw_nzdata)
            nonz_gene = prep.pad_zero(nonz_gene)

            dw_nzdata_batch.append(torch.tensor(dw_nzdata, dtype=torch.float32))
            nonz_gene_batch.append(torch.tensor(nonz_gene, dtype=torch.int32))
            zero_idx_batch.append(torch.tensor(zero_idx, dtype=torch.float32))
            label_batch.append(torch.tensor(label, dtype=torch.float32))
            mask_batch.append(torch.tensor(msk, dtype=torch.float32))

        return {
            'dw_nzdata': torch.stack(dw_nzdata_batch),
            'nonz_gene': torch.stack(nonz_gene_batch),
            'zero_idx': torch.stack(zero_idx_batch),
            'label': torch.stack(label_batch),
            'mask': torch.stack(mask_batch),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop,
        collate_fn=collate_fn
    )

def basic(args):
    ### CellFM param ###
    cfg = Config_80M()
    cfg.ecs_threshold = 0.8
    cfg.ecs = True
    cfg.add_zero = True
    cfg.pad_zero = True
    cfg.use_bs = args.batch_size
    cfg.mask_ratio = 0.5
    ### Main param ###
    cfg.dataset = args.dataset
    cfg.feature_col = args.feature_col
    cfg.ckpt_path = args.ckpt_path
    cfg.device = args.device
    cfg.epoch = args.epoch 
    cfg.num_cls = 1
    
    #### A lots of Path ####
    PT_PATH = f"../data_pt/{cfg.dataset}"
    MODEL_PATH = f"../model_checkpoint/{cfg.dataset}" # for 40000 cells
    
    #### Make dir ####
    os.makedirs(PT_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    def load_data(adata_path, mode="train"):
        adata = read_h5ad(adata_path)
        # adata.var_names = adata.var['gene_name']
        adata.obs['celltype'] = adata.obs['cell_type']
        adata.obs['feat'] = adata.obs[cfg.feature_col].cat.codes.values
        cfg.num_cls = len(adata.obs['feat'].unique())
        
        adata.obs['batch_id'] = 0
        if mode == "train":
            adata.obs['train'] = 0
            dataset = SCrna(adata, mode="train")
            prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
            loader = build_dataset(
                dataset,
                prep=prep,
                batch_size=cfg.use_bs,
                pad_zero=cfg.pad_zero,
                drop=False,
                shuffle=True
            )
        if mode== "test":
            adata.obs['train'] = 2
            dataset = SCrna(adata, mode="test")
            prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
            loader = build_dataset(
                dataset,
                prep=prep,
                batch_size=cfg.use_bs,
                drop=False,
                shuffle=False
            )
        return loader
    ################### training ###################
    train_adata_path = f"/data/user/liwb/project/CellFM/datasets/cell_annotion/Inter/{cfg.dataset}/train.h5ad"
    test_adata_path = f"/data/user/liwb/project/CellFM/datasets/cell_annotion/Inter/{cfg.dataset}/test.h5ad"
    
    train_loader = load_data(train_adata_path, mode="train")
    test_loader = load_data(test_adata_path, mode="test")
    

    net = Finetune_Cell_FM(cfg) # 27855

    for name, param in net.named_parameters():
        param.requires_grad = "cls." in name or "encoder" in name
    
    print("Trainable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    net = net.to(cfg.device)
    net.extractor.load_model(weight=True, moment=False)

cfg = Config_80M()

cfg.ckpt_path = "../checkpoints/base_weight.ckpt"
cfg.device = 'cpu'
cfg.ecs_threshold = 0.8
cfg.ecs = True
cfg.add_zero = True
cfg.pad_zero = True
cfg.mask_ratio = 0.5
cfg.num_cls = 1
cfg.feature_col = "celltype"
cfg.use_bs = 32
cfg.mask_ratio = 0.5
cfg.epoch = 30

net = Finetune_Cell_FM(cfg) # 27855

def load_data(adata_path, mode="train"):
    adata = read_h5ad(adata_path)
    adata.obs['feat'] = adata.obs[cfg.feature_col].cat.codes.values
    cfg.num_cls = len(adata.obs['feat'].unique())
    
    adata.obs['batch_id'] = 0
    if mode == "train":
        adata.obs['train'] = 0
        dataset = SCrna(adata, mode="train")
        prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
        loader = build_dataset(
            dataset,
            prep=prep,
            batch_size=cfg.use_bs,
            pad_zero=cfg.pad_zero,
            drop=False,
            shuffle=False
        )
    if mode== "test":
        adata.obs['train'] = 2
        dataset = SCrna(adata, mode="test")
        prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio)
        loader = build_dataset(
            dataset,
            prep=prep,
            batch_size=cfg.use_bs,
            drop=False,
            shuffle=False
        )
    return loader

# train_adata_path = f"/home/tl688/tl688/tl688_data/largedataset/sce_heart.h5ad"
# train_adata_path = f"/home/tl688/tl688/scgpt_dataset/MHSP_raw.h5ad"
train_adata_path = args.filename
train_loader = load_data(train_adata_path, mode="train")
for name, param in net.named_parameters():
    param.requires_grad = "cls." in name or "encoder" in name

# for name, param in net.named_parameters():
#     param.requires_grad = "cls." in name

print("Trainable parameters:")
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name)
cfg.device = 'cuda'
net = net.to(cfg.device)
optimizer = AdamW([p for p in net.parameters() if p.requires_grad], 
                  lr=1e-4,
                  weight_decay=1e-5)

scaler = GradScaler() 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

criterion_cls = nn.CrossEntropyLoss()
for epoch in range(cfg.epoch):
    net.train()
    print("training...")
    running_loss = 0.0
    running_acc = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epoch}")

    for step, batch in enumerate(progress):    

        raw_nzdata = batch['raw_nzdata'].to(cfg.device)
        dw_nzdata = batch['dw_nzdata'].to(cfg.device)
        ST_feat = batch['ST_feat'].to(cfg.device)
        nonz_gene = batch['nonz_gene'].to(cfg.device)
        mask_gene = batch['mask_gene'].to(cfg.device)
        zero_idx = batch['zero_idx'].to(cfg.device)
        celltype_label = batch['celltype_label'].to(cfg.device)
        batch_id = batch['batch_id'].to(cfg.device)
        feat = batch['feat'].long().to(cfg.device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            cls, mask_loss, cls_token = net(
                raw_nzdata=raw_nzdata,
                dw_nzdata=dw_nzdata,
                ST_feat=ST_feat,
                nonz_gene=nonz_gene,
                mask_gene=mask_gene,
                zero_idx=zero_idx
            ) 

            loss = mask_loss

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        accuracy = (cls.argmax(1) == feat).sum().item()
        accuracy = accuracy / len(batch_id)

        running_loss += loss.item()
        running_acc += accuracy

        avg_loss = running_loss / (step + 1)
        avg_acc = running_acc / (step + 1)

        progress.set_postfix(loss=avg_loss, acc=avg_acc)

train_emb  = []

progress = tqdm(train_loader)
for step, batch in enumerate(progress):    
    raw_nzdata = batch['raw_nzdata'].to(cfg.device)
    dw_nzdata = batch['dw_nzdata'].to(cfg.device)
    ST_feat = batch['ST_feat'].to(cfg.device)
    nonz_gene = batch['nonz_gene'].to(cfg.device)
    mask_gene = batch['mask_gene'].to(cfg.device)
    zero_idx = batch['zero_idx'].to(cfg.device)
    celltype_label = batch['celltype_label'].to(cfg.device)
    batch_id = batch['batch_id'].to(cfg.device)
    feat = batch['feat'].long().to(cfg.device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            cls, mask_loss, cls_token = net(
                raw_nzdata=raw_nzdata,
                dw_nzdata=dw_nzdata,
                ST_feat=ST_feat,
                nonz_gene=nonz_gene,
                mask_gene=mask_gene,
                zero_idx=zero_idx
            ) 
    train_emb += cls_token.detach().cpu().numpy().tolist()
outname = train_adata_path.split('/')[-1].split('.')[0]
np.save(f"./cell_embedding_ft/{outname}.pkl", np.array(train_emb))