import warnings
warnings.filterwarnings("ignore")

import hdf5plugin
import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.utils.data import stratified_sample_genes_by_sparsity
from CellPLM.pipeline.imputation import ImputationPipeline, ImputationDefaultPipelineConfig, ImputationDefaultModelConfig
from CellPLM.pipeline.experimental import symbol_to_ensembl

# !kill -9 3485071

## Specify important parameters before getting started

DATASET = 'Liver' # 'Lung'
PRETRAIN_VERSION = '20230926_85M'
DEVICE = 'cuda:0'

## Load Downstream Dataset
set_seed(11)

ref_data = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/scGPT/examples/mouse_scrnaseq.h5ad")
query_data = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/scGPT/examples/mouse_spatial.h5ad")

ref_data.var_names = symbol_to_ensembl(ref_data.var_names )
query_data.var_names = symbol_to_ensembl(query_data.var_names )
true_list = set(query_data)

ref_data.var_names_make_unique()
query_data.var_names_make_unique()

ref_data.obs['batch'] = '0'
query_data.obs['batch'] = '1'

train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)
query_data = train_data[train_data.obs['batch'] == '1']

target_genes = sorted(set(query_data.var_names) - true_list)
query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
query_data[:, target_genes].X = 0
train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)

train_data.obs['split'] = 'train'
train_data.obs['split'][train_data.obs['batch']==query_data.obs['batch'][-1]] = 'valid'
train_data.obs['split'][train_data.obs['batch']==ref_data.obs['batch'][-1]] = 'valid'

query_genes = [g for g in query_data.var.index if g not in target_genes]
query_batches = list(query_data.obs['batch'].unique())
ref_batches = list(ref_data.obs['batch'].unique())
batch_gene_list = dict(zip(list(query_batches) + list(ref_batches),
    [query_genes]*len(query_batches) + [ref_data.var.index.tolist()]*len(ref_batches)))

pipeline_config = ImputationDefaultPipelineConfig.copy()
model_config = ImputationDefaultModelConfig.copy()


pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='../ckpt')

pipeline.fit(train_data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            batch_gene_list = batch_gene_list, # Specify genes that are measured in each batch, see previous section for more details
            device = DEVICE,
            ) 

out = pipeline.predict(
        query_data, # An AnnData object
        pipeline_config, # The config dictionary we created previously, optional
        device = DEVICE,
    )

pipeline.score(
                query_data, # An AnnData object
                evaluation_config = {'target_genes': target_genes}, # The config dictionary we created previously, optional
                label_fields = ['truth'], # A field in .obsm that stores the ground-truth for evaluation
                device = DEVICE,
)  

import torch
torch.save(out, "imputed_cellplm.pkl")
