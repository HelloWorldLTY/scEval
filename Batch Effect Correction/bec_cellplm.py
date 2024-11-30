import warnings
warnings.filterwarnings("ignore")
import os

import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
import scanpy as sc
import matplotlib.pyplot as plt
import resource
# import rapids_singlecell as rsc  # For faster evaluation, we recommend the installation of rapids_singlecell.

## Specify important parameters before getting started

PRETRAIN_VERSION = '20230926_85M'
DEVICE = 'cuda'

set_seed(42)
filename = 'benchmark_trajectory'
data = ad.read_h5ad(f"/gpfs/gibbs/pi/zhao/tl688/trajectory_data/{filename}.h5ad")
data.obs_names_make_unique()

## Set up the pipeline

pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='/gpfs/gibbs/pi/zhao/tl688/CellPLM_cta/ckpt/')
pipeline.model

## Evaluation and Inference

embedding = pipeline.predict(data, # An AnnData object
                device=DEVICE) # Specify a gpu or cpu for model inference

data.obsm['emb'] = embedding.cpu().numpy()
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  / (1e6) )
# resource.getrusage(resource.RUSAGE_SELF).ru_utime
data.write_h5ad(f"./bec_cellplm/{filename}_cellplm.h5ad")


