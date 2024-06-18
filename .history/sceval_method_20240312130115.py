import AnnData
import torch
import numpy as np
import scib
import scanpy as sc
import scipy
import scipy.stats
import scvi 
import scglue
import networkx as nx
import TOSICA
import scanpy as sc
import numpy as np
import pandas as pd
import tangram as tg

from typing import List, Tuple, Dict, Union, Optional
from ResPAN import run_respan
from itertools import chain
from anndata import AnnData
from gears import PertData, GEARS
from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis

'''
To evaluate simulation, we used scDesign3 and Splatter, which are both designed in R. Please install R and refer their tutorials to run their codes.
'''
class scEval_bench(object):

    def __init__(self, adata):
        self.label = 'scGPT'
        self.adata = adata # adata is raw data file for evaluating.
        self.pvalue = 0.005

    def bec_scvi(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scvi'):
        adata = self.adata
        adata.raw = adata  # keep full dimension safe
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=2000,
            layer="counts",
            batch_key="batch",
            subset=True,
        )
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
        model.train()
        adata.obsm[emb_name] = model.get_latent_representation()
        return adata



    def bec_respan(self, batch_key = 'batch',label_key = 'celltype'):

        adata = self.adata
        # pre-processing
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        adata.X = adata.X.toarray().astype('float')
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
        adata = adata[:, adata.var['highly_variable']]
        # check if data is in sparse format
        if isinstance(adata.X, scipy.sparse.csr.csr_matrix): 
            adata_new = sc.AnnData(adata.X.todense())
            adata_new.obs = adata.obs.copy()
            adata_new.obs_names = adata.obs_names
            adata_new.var_names = adata.var_names
            adata_new.obs_names.name = 'CellID'
            adata_new.var_names.name = 'Gene'
            del adata
            adata = adata_new

        adata_new = run_respan(adata, batch_key=batch_key, epoch=300, batch=1024, reduction='pca', subsample=3000, seed=999)
        return adata_new

    # here atac data should contain peak infromation.
    def mdi_glue(self, rna, atac, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_glue'):
        result_folder = './'
        # Data preprocessing
        sc.pp.filter_cells(rna, min_genes=200)
        sc.pp.filter_genes(rna, min_cells=3)
        rna.layers["counts"] = rna.X.copy()
        sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
        sc.pp.normalize_total(rna)
        sc.pp.log1p(rna)
        sc.pp.scale(rna)
        sc.tl.pca(rna, n_comps=100, svd_solver="auto")
        sc.pp.filter_cells(atac, min_genes=200)
        sc.pp.filter_genes(atac, min_cells=3)
        scglue.data.lsi(atac, n_components=100, n_iter=15)

        # Graph construction
        guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)

        # Configure data
        scglue.models.configure_dataset(
            rna, "NB", use_highly_variable=True,
            use_layer="counts", use_rep="X_pca"
        )
        scglue.models.configure_dataset(
            atac, "NB", use_highly_variable=True,
            use_rep="X_lsi"
        )

        guidance_hvf = guidance.subgraph(chain(
            rna.var.query("highly_variable").index,
            atac.var.query("highly_variable").index
        )).copy()

        # Run GLUE
        glue = scglue.models.fit_SCGLUE(
            {"rna": rna, "atac": atac}, guidance_hvf,
            fit_kws={"directory": result_folder}
        )
        glue.save("%s/glue.dill" % result_folder)

        # Check integration consistency
        dx = scglue.models.integration_consistency(
            glue, {"rna": rna, "atac": atac}, guidance_hvf
        )
        print(dx)

        # KNN classifier
        rna.obsm[emb_name] = glue.encode_data("rna", rna)
        atac.obsm[emb_name] = glue.encode_data("atac", atac)

        return rna, atac

    def mdi_scjoint(self, rna_path, atac_path, result_folder, subset_rna, subset_atac, rna_new_annot, stage1_lr, stage3_lr, nepoch):
        from scjoint import run_scJoint
        run_scJoint(rna_path, atac_path, result_folder, subset_rna, subset_atac, rna_new_annot, stage1_lr, stage3_lr, nepoch)


    def cta_tosica(self, ref_data, query_data, batch_key = 'batch',label_key = 'celltype'):
        TOSICA.train(ref_data, gmt_path='human_gobp', label_name=label_key, epochs=3, project = "./")
        model_weight_path = './model-0.pth'
        new_adata = TOSICA.pre(query_data, model_weight_path = model_weight_path, project = "./")
        return new_adata
    
    def imp_tangram(self, ref_data, query_data, batch_key = 'batch',label_key = 'celltype', device='cpu'):

        # Please ensure ref_data has more genes than query_data.
        adata_sc = ref_data 
        adata_st = query_data
        sc.tl.rank_genes_groups(adata_sc, groupby=label_key, use_raw=False)
        markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
        markers = list(np.unique(markers_df.melt().value.values))
        tg.pp_adatas(adata_sc, adata_st, genes=markers)
        ad_map = tg.map_cells_to_space(adata_sc, adata_st,
            mode="cells",
        #     mode="clusters",
        #     cluster_label='cell_subclass',  # .obs field w cell types
            density_prior='rna_count_based',
            num_epochs=500,
            # device="cuda:0",
            device=device
        )
        ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc)
        return ad_ge

    def pert_gears(self, batch_key = 'batch',label_key = 'celltype', device='cuda"0', save_path = './'):
        pert_data = self.adata 
        # set up and train a model
        gears_model = GEARS(pert_data, device = device)
        gears_model.model_initialize(hidden_size = 64)
        gears_model.train(epochs = 20)
        gears_model.save_model(save_path)
        gears_model.load_pretrained(save_path)
        test_res = evaluate(gears_model.dataloader['test_loader'], gears_model.model, gears_model.config['uncertainty'], gears_model.device)
        test_metrics, test_pert_res = compute_metrics(test_res)
        return test_metrics




