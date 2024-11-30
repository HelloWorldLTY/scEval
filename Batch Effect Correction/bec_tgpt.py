import re
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# Setting parameter and file path

device = "cuda" if torch.cuda.is_available() else "cpu" 
tokenizer_file = "lixiangchun/transcriptome-gpt-1024-8-16-64" 
checkpoint = "lixiangchun/transcriptome-gpt-1024-8-16-64" ## Pretrained model
celltype_path = "./data/Muris_cell_labels.txt.gz" ## Cell type annotation
max_len = 64 ## Number of top genes used for analysis
text_file = "./data/Muris_gene_rankings.txt.gz"  ## Gene symbols ranked by exprssion

adata = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/largedataset/singlecellimmune_covid.h5ad")

adata.var_names = list(adata.var.feature_name)

def get_gene_token(adata):
    lines = []
    for i in adata.obs_names:
        adata_t = adata[i,:]
        reverse_index = np.argsort(adata_t.X.toarray()[0])[::-1]
        reverse_index = reverse_index[0:256]
        gene_list = adata_t.var_names.values[reverse_index]
        raw_gene = ''
        for index, gene in enumerate(gene_list):
            raw_gene += gene
            if index != len(gene_list)-1:
                raw_gene += ' '
        lines.append(raw_gene)
    return lines
    

lines = get_gene_token(adata)

# Extract features

class LineDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines
        self.regex = re.compile(r'\-|\.')
    def __getitem__(self, i):
        return self.regex.sub('_', self.lines[i])
    def __len__(self):
        return len(self.lines)

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
model = GPT2LMHeadModel.from_pretrained(checkpoint,output_hidden_states = True).transformer
model = model.to(device)
model.eval()

ds = LineDataset(lines)
dl = DataLoader(ds, batch_size=64)

Xs = []
for a in tqdm(dl, total=len(dl)):
    batch = tokenizer(a, max_length= max_len, truncation=True, padding=True, return_tensors="pt")

    for k, v in batch.items():
        batch[k] = v.to(device)

    with torch.no_grad():
        x = model(**batch)
    
    eos_idxs = batch.attention_mask.sum(dim=1) - 1
    xx = x.last_hidden_state
       
    result_list = [[] for i in range(len(xx))]

    for j, item in enumerate(xx):
        result_list[j] = item[1:int(eos_idxs[j]),:].mean(dim =0).tolist()
        
    Xs.extend(result_list)
    
features = np.stack(Xs)

adata_test=sc.AnnData(features)
adata_test.obs["batch"] = list(adata.obs["Site"])
adata_test.obs["batch"] = adata_test.obs["batch"].astype("category")

adata_test.obs["str_batch"] = adata_test.obs["batch"].astype("category")
adata_test.obs["celltype"] = list(adata.obs["cell_type"])
adata_test.obs["celltype"] = adata_test.obs["celltype"].astype("category")

adata_test.write_h5ad('immuneatlas_tgpt_all.h5ad')