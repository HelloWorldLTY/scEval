import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

import torch
from anndata import AnnData
import numpy as np
import wandb
from scipy.sparse import issparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("../")
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
    masked_ce_loss
)
import scgpt as scg
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run encoder')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seeds')

    parser.add_argument('--dataset', default='',
                        help='method we intend to perform benchmark') 
    
    parser.add_argument('--dataset_name', type=str, default="PBMC_10K",
                        help='default dataset name.')

    parser.add_argument('--mask_ratio', type=float, default=0.4,
                        help='mask ratio.')
    
    parser.add_argument('--epoches', type=int, default=10,
                        help='epoches.')
    
    parser.add_argument('--n_bins', type=int, default=51,
                        help='n_bins')  
    
    parser.add_argument('--GEPC', type=bool, default=True,
                        help='n_bins')   

    parser.add_argument('--ecs_thres', type=float, default=0.8,
                        help='beta in the ecs loss')
    
    parser.add_argument('--dab_weight', type=float, default=1.0,
                        help='dab weight')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate')    

    parser.add_argument('--n_hvg', type=int, default=2000,
                        help='The number of selected hvgs')    

    parser.add_argument('--explicit_zero_prob', type=bool, default=True,
                        help='whether we need to use zero prob loss or not')    
    
    parser.add_argument('--mask_output_include', type=bool, default=True,
                        help='whether we need to use zero prob loss or not') 

    return parser.parse_args()

def read_pathway(path):
    if "loom" in path:
        adata = sc.read_loom(path)
        if ("MCA" in path) or ("MHSP" in path):
            adata.var_names = [i.upper() for i in adata.var_names]
        return adata 
    
    elif "h5ad" in path:
        adata = sc.read_h5ad(path)
        if "spaital_mouse_slideseqv2.h5ad" in path:
            adata.var_names = [i.upper() for i in adata.var_names]
            adata.obs["celltype"] = list(adata.obs['cluster'])
            adata.obs['batch'] = ['spatial' for i in adata.obs['cluster']]
            return adata 
        elif "Immune_ALL_human.h5ad" in path: 
            adata.obs['batch'] = list(adata.obs.sample_ID)
            adata.obs['celltype'] = list(adata.obs.final_annotation)
            return adata 
        return adata
    

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
# os.environ["WANDB_MODE"] = "offline"


if __name__ == "__main__":
    args = parse_args()

    print("scEval-start")
    hyperparameter_defaults = dict(
        seed=args.seed,
        dataset_name="PBMC_10K",
        do_train=True,
        load_model="save/scGPT_bc",
        mask_ratio=args.mask_ratio,
        epochs=args.epoches,
        n_bins=args.n_bins,
        GEPC=True,  # Masked value prediction for cell embedding
        ecs_thres=args.ecs_thres,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=args.dab_weight,
        lr=args.lr,
        batch_size=64,
        layer_size=128,
        nlayers=4,
        nhead=4,
        # if load model, batch_size, layer_size, nlayers, nhead will be ignored
        dropout=args.dropout,
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        log_interval=100,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
    )
    run = wandb.init(
        config=hyperparameter_defaults,
        project="scGPT",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    config = wandb.config
    print(config)

    set_seed(config.seed)

    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    mask_value = -1
    pad_value = -2
    n_input_bins = config.n_bins

    n_hvg = args.n_hvg  # number of highly variable genes
    max_seq_len = n_hvg + 1
    per_seq_batch_sample = True
    DSBN = True  # Domain-spec batchnorm
    explicit_zero_prob = args.explicit_zero_prob  # whether explicit bernoulli for zeros
    mask_output_include = args.mask_output_include

    dataset_name = config.dataset_name
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    
    # make the batch category column

    adata = read_pathway(args.dataset)
    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    ori_batch_col = "batch"
    adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()
    data_is_raw = True

    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]


    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=3,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)

    adata_train = adata[:, adata.var['dose_cond'] != -1]
    sc.pp.filter_cells(adata_train, min_counts = 1) # choose to only include labeled genes 

    if per_seq_batch_sample:
        # sort the adata by batch_id in advance
        adata_sorted = adata_train[adata_train.obs["batch_id"].argsort()].copy()

  
    # ## Tokenize input

    input_layer_key = "X_binned"
    all_counts = (
        adata_train.layers[input_layer_key].A
        if issparse(adata_train.layers[input_layer_key])
        else adata_train.layers[input_layer_key]
    )
    genes = adata_train.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_gene_labels,
        valid_gene_labels,
        train_gene_name,
        valid_gene_name,
        train_adata_varnames,
        valid_adata_varnames
        
    ) = train_test_split(
        all_counts.T, adata_train.var['dose_cond'], genes, adata_train.var_names, test_size=0.33, shuffle=True, random_state=42
    )

    train_data = train_data.T 
    valid_data = valid_data.T

    if config.load_model is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    gene_ids_train = np.array(vocab(train_gene_name), dtype=int)
    gene_ids_valid = np.array(vocab(valid_gene_name), dtype=int)


    tensor_train_gene_labels = torch.from_numpy(train_gene_labels.values).long().cuda()
    tensor_valid_gene_labels = torch.from_numpy(valid_gene_labels.values).long().cuda()
        
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids_train,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids_valid,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )


    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )
        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )

        tensor_batch_labels_train = torch.from_numpy(batch_ids).long()
        tensor_batch_labels_valid = torch.from_numpy(batch_ids).long()
        
        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
            # "celltype_labels":tensor_celltype_labels_train
        }
        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "batch_labels": tensor_batch_labels_valid,
            # "celltype_labels": tensor_celltype_labels_valid
        }

        return train_data_pt, valid_data_pt


    # dataset
    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}


    # data_loader
    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        dataset = SeqDataset(data_pt)

        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    # # Create and finetune scGPT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=config.GEPC,
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=n_input_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=True,
        pre_norm=config.pre_norm,
    )
    if config.load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    # wandb.watch(model)


    # criterion = masked_mse_loss
    criterion = masked_ce_loss
    criterion_dab = nn.CrossEntropyLoss()


    # for param in model.parameters():
    #     param.requires_grad = False

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.transformer_encoder.layers[-2].parameters():
    #     param.requires_grad = True
        
    model.fc = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.Mish() , nn.Linear(512,256), nn.BatchNorm1d(256), nn.Mish() , nn.Linear(256, 2))
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    def model_forward(
        src,
        values,
        src_key_padding_mask,
        batch_labels = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ):
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        transformer_output = model._encode(
            src, values, src_key_padding_mask, batch_labels
        )
        
        if model.use_batch_labels:
            batch_emb = model.batch_encoder(batch_labels)  # (batch, embsize)
            
        output = {}
        mlm_output = model.decoder(
            transformer_output
            if not model.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        
        gene_emb = model.encoder(torch.tensor(gene_ids_train, dtype=torch.long).to(device))
        # gene_emb = model.value_encoder(values) + gene_emb
        
        output["gene_emb"] = gene_emb
        
        if model.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if model.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = model._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb
        if CLS:
            output["cls_output"] = model.cls_decoder(cell_emb)  # (batch, n_cls)
        if CCE:
            cell1 = cell_emb
            transformer_output2 = model._encode(
                src, values, src_key_padding_mask, batch_labels
            )
            cell2 = model._get_cell_emb_from_layer(transformer_output2)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and model.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = model.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = model.creterion_cce(cos_sim, labels)
        if MVC:
            mvc_output = model.mvc_decoder(
                cell_emb
                if not model.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                model.cur_gene_token_embs,
            )
            if model.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if model.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - model.ecs_threshold) ** 2)

        if model.do_dab:
            output["dab_output"] = model.grad_reverse_discriminator(cell_emb)

        return output


    def train(model: nn.Module, loader: DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = config.log_interval
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            # celltype_labels = batch_data["celltype_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model_forward(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=config.GEPC,
                    ECS=config.ecs_thres > 0,
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict
                
                loss = loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )

                    
                metrics_to_log = {"train/mse": loss_mse.item()}
            
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                    
                if config.GEPC:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc
                    metrics_to_log.update({"train/mvc": loss_gepc.item()})
                    
                if config.GEPC and explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                    
                if config.ecs_thres > 0:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                    
                
                gene_emb = output_dict["gene_emb"]
                output_label_prob = model.fc(gene_emb)
                loss_dab = criterion_dab(output_label_prob, tensor_train_gene_labels)

                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            wandb.log(metrics_to_log)

            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc.item() if config.GEPC else 0.0
            total_error += mre.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                start_time = time.time()


    def define_wandb_metrcis():
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")


    def evaluate(model: nn.Module, loader: DataLoader) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)


                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model_forward(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if DSBN else None,
                    )
                    output_values = output_dict["mlm_output"]

                    gene_emb = output_dict["gene_emb"]
                    output_label_prob = model.fc(gene_emb)
                    loss_dab = criterion_dab(output_label_prob, tensor_train_gene_labels)

                    masked_positions = input_values.eq(mask_value)
                    loss = criterion(output_values, target_values, masked_positions)
                    
                total_loss += loss.item() * len(input_gene_ids)
                total_error += masked_relative_error(
                    output_values, target_values, masked_positions
                ).item() * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids)
                total_num += len(input_gene_ids)

        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/mre": total_error / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
                / total_num,
                "epoch": epoch,
            },
        )

        return total_loss / total_num, total_error / total_num


    def eval_testdata(
        model: nn.Module,
        adata_t: AnnData,
        include_types: List[str] = ["cls"],
    ) -> Optional[Dict]:
        """evaluate the model on test dataset of adata_t"""
        model.eval()

        # copy adata_t to avoid reuse previously computed results stored in adata_t
        adata_t = adata_t.copy()
        
        all_counts = (
            adata_t.layers[input_layer_key].A
            if issparse(adata_t.layers[input_layer_key])
            else adata_t.layers[input_layer_key]
        )

        celltypes_labels = adata_t.obs["celltype"].tolist()
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata_t.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

        # Evaluate cls cell embeddings
        if "cls" in include_types:
            logger.info("Evaluating cls cell embeddings")
            tokenized_all = tokenize_and_pad_batch(
                valid_data,
                gene_ids_valid,
                max_len=max_seq_len,
                vocab=vocab,
                pad_token=pad_token,
                pad_value=pad_value,
                append_cls=True,  # append <cls> token at the beginning
                include_zero_gene=True,
            )
            all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
            src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
                # cell_embeddings = model.encode_batch(
                #     all_gene_ids,
                #     all_values.float(),
                #     src_key_padding_mask=src_key_padding_mask,
                #     batch_size=config.batch_size,
                #     batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                #     time_step=0,
                #     return_np=True,
                # )
                
                # cell_embeddings = cell_embeddings / np.linalg.norm(
                #     cell_embeddings, axis=1, keepdims=True
                # )

                gene_embeddings = model.encoder(torch.tensor(gene_ids_valid, dtype=torch.long).to(device))
                ctp = model.fc(torch.Tensor(gene_embeddings).cuda())
                ctp = nn.Softmax(dim = 1)(ctp)
                label = torch.argmax(ctp, dim=1).cpu().numpy() 
            
            print(label)
            
            print("accuracy")
            
            print(np.sum(label == valid_gene_labels.values)/len(label))
            print(classification_report(label, valid_gene_labels.values, digits=4))
            
            return label

    def return_testdata(
        model: nn.Module,
        adata_t: AnnData,
        include_types: List[str] = ["cls"],
    ) -> Optional[Dict]:
        """evaluate the model on test dataset of adata_t"""
        model.eval()

        # copy adata_t to avoid reuse previously computed results stored in adata_t
        adata_t = adata_t.copy()

        all_counts = (
            adata_t.layers[input_layer_key].A
            if issparse(adata_t.layers[input_layer_key])
            else adata_t.layers[input_layer_key]
        )

        celltypes_labels = adata_t.obs["celltype"].tolist()
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata_t.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

        # Evaluate cls cell embeddings
        if "cls" in include_types:
            logger.info("Evaluating cls cell embeddings")
            tokenized_all = tokenize_and_pad_batch(
                all_counts,
                gene_ids,
                max_len=max_seq_len,
                vocab=vocab,
                pad_token=pad_token,
                pad_value=pad_value,
                append_cls=True,  # append <cls> token at the beginning
                include_zero_gene=True,
            )
            all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
            src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=config.batch_size,
                    batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                    time_step=0,
                    return_np=True,
                )
            cell_embeddings = cell_embeddings / np.linalg.norm(
                cell_embeddings, axis=1, keepdims=True
            )
            
        return cell_embeddings


    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    define_wandb_metrcis()

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=config.batch_size,
            shuffle=True,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if config.do_train:
            train(
                model,
                loader=train_loader,
            )
        val_loss, val_mre = evaluate(
            model,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

            # eval on testdata
            results = eval_testdata(
                best_model,
                adata_t=adata_sorted if per_seq_batch_sample else adata,
                include_types=["cls"],
            )

        scheduler.step()

    gc.collect()

    print("scEval Finish!")
