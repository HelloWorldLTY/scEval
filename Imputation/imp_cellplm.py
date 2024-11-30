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

ref_data.var_names_make_unique()
query_data.var_names_make_unique()

ref_data.obs['batch'] = ref_data.obs_names
query_data.obs['batch'] = query_data.obs_names

query_data.var_names

# ref_data.var_names = [i.upper() for i in ref_data.var_names]
# query_data.var_names = [i.upper() for i in query_data.var_names]
# target_genes = query_data.var_names
target_genes = ['ENSG00000206579']
query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
query_data[:, target_genes].X = 0
train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)

train_data.obs['split'] = 'train'
train_data.obs['split'][train_data.obs['batch']==query_data.obs['batch'][-1]] = 'valid'
train_data.obs['split'][train_data.obs['batch']==ref_data.obs['batch'][-1]] = 'valid'


query_data.obs['platform'] = 'merfish'

query_data.obsm['spatial'][:,0]

query_data.obs['x_FOV_px'] = query_data.obsm['spatial'][:,0]
query_data.obs['y_FOV_px'] = query_data.obsm['spatial'][:,1]

query_data.var.index

ref_data.var.index

query_var_new = []
for i in query_data.var.index:
    if "ENSG" in i:
        query_var_new.append(i)
ref_var_new = []
for i in ref_data.var.index:
    if "ENSG" in i:
        ref_var_new.append(i)

query_data = query_data[:,query_var_new]
ref_data = ref_data[:,ref_var_new]
## Specify gene to impute

query_genes = [g for g in query_data.var.index if g not in ['MRPL15']]
query_batches = list(query_data.obs['batch'].unique())
ref_batches = list(ref_data.obs['batch'].unique())
batch_gene_list = dict(zip(list(query_batches) + list(ref_batches),
    [query_genes]*len(query_batches) + [ref_data.var.index.tolist()]*len(ref_batches)))

## Overwrite parts of the default config
pipeline_config = ImputationDefaultPipelineConfig.copy()
model_config = ImputationDefaultModelConfig.copy()

pipeline_config, model_config

## Fine-tuning

pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                      overwrite_config=model_config,  # This is for overwriting part of the pretrain config
                                      pretrain_directory='/gpfs/gibbs/pi/zhao/tl688/CellPLM_cta/ckpt/')
pipeline.model

# batch_gene_list
pipeline.fit(train_data, # An AnnData object
            pipeline_config, # The config dictionary we created previously, optional
            split_field = 'split', #  Specify a column in .obs that contains split information
            train_split = 'train',
            valid_split = 'valid',
            batch_gene_list = batch_gene_list, # Specify genes that are measured in each batch, see previous section for more details
            device = DEVICE,
            ) 
