from typing import List, Tuple, Dict, Union, Optional
import AnnData
import torch
import numpy as np
import scib
import scanpy as sc
import scipy
from scgpt.utils import set_seed
from sklearn.metrics import classification_report
import scipy.stats
import scvi 


class scEval_bench(object):

    def __init__(self, adata):
        self.label = 'scGPT'
        self.adata = adata # adata is raw data file for evaluating.
        self.pvalue = 0.005

    def bec_scvi(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass

    def bec_respan(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass

    def mdi_glue(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass

    def mdi_scjoint(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass

    def cta_tosica(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass

    def cta_tosica(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass

    def imp_tangram(self, ref_data, query_data, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass

    def imp_tangram(self, ref_data, query_data, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        pass






