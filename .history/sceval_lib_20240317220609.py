from anndata import AnnData
import torch
import numpy as np
import scib
import scanpy as sc
import scipy
import scipy.stats
from scgpt.utils import set_seed
from sklearn.metrics import classification_report
from typing import List, Tuple, Dict, Union, Optional

set_seed(0)
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "batch",
    label_key: str = "celltype",
    emb_name: str = "X_scGPT",
    notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed=emb_name,
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  
        ari_=True, 
        cell_cycle_=False,
        kBET_=True,  
        ilisi_=False,
        clisi_=False,
    )

    result_dict = results[0].to_dict()

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    print(results)
    return result_dict


def eval_scib_metrics_onlybio(
    adata: AnnData,
    batch_key: str = "batch",
    label_key: str = "celltype",
    emb_name: str = "X_scGPT",
    notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics_onlybio(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed=emb_name,
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  
        ari_=True,  
        cell_cycle_=False,
        kBET_=False,  
        ilisi_=False,
        clisi_=False,
    )

    result_dict = results[0].to_dict()
    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    print(results)
    return result_dict

def calculate_correlation_metric(y1, y2):
    cor = 0.0
    y1 = y1.float()
    y2 = y2.float()
    for id1, id2 in zip(y1, y2):
        
        cor_cal,_ = scipy.stats.pearsonr(id1,id2)
        cor += cor_cal.item()
    return cor


class scEval(object):

    def __init__(self, adata):
        self.label = 'scGPT'
        self.adata = adata # adata is the output of the model you plan to benchmark.
        self.pvalue = 0.005

    def evaluation_bec(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scGPT'):
        results = eval_scib_metrics(self.adata,batch_key,label_key, emb_name)
        return results
    

    def evaluation_cta_gfp(self, pred_label, true_label):
        results = classification_report(pred_label, true_label, digits=4)
        return results
    
    def evaluation_perturb_pred(self, pred_model, true_result): #assume the outputs are both in AnnData format. Rows are cells while columns are genes.
        cor_total = calculate_correlation_metric(pred_model.X.T, true_result.X.T)
        return {"correlation":cor_total / len(pred_model.X.T)}
    
    def evaluation_perturb_pred_gearsofficial(self, gears_model, pred_model ):
        from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis
        test_res = evaluate(gears_model.dataloader['test_loader'], pred_model)
        test_metrics, test_pert_res = compute_metrics(test_res)
        return test_metrics
    
    def evaluation_imputation_scrna(self, batch_key = 'batch',label_key = 'celltype', emb_name = 'X_scGPT'):
        results = eval_scib_metrics_onlybio(self.adata,batch_key,label_key, emb_name)
        return results
    
    def evaluation_imputation_spatial(self, adata_sp):
        adata_imp_new = self.adata[:, adata_sp.var_names]
        cor_list = []
        pval_list = []
        for item in adata_sp.var_names:
            adata1 = adata_sp[:,item]
            adata2 = adata_imp_new[:,item]
            cor, pval = scipy.stats.pearsonr(np.array(adata1.X.todense().T)[0], np.array(adata2.X.T)[0]) # for this step, please check the data form
            cor_list.append(cor)
            pval_list.append(pval)

        adata_imp_new.var['cor'] = cor_list 
        adata_imp_new.var['pval'] = pval_list

        mean_cor = np.mean(adata_imp_new.var['cor'].values)

        avg_sig = np.sum(adata_imp_new.var['pval'].values<self.pvalue)/len(adata_imp_new.var['pval'].values)
        return {"mean_cor":mean_cor, "avg_sign":avg_sig} 
    
    def evaluation_simulation(self, batch_key = 'batch',label_key = 'celltype', isbatch = True, emb_name = 'X_scGPT'):

        if isbatch:
            results = eval_scib_metrics(self.adata,batch_key,label_key, emb_name)
            return results 
        else:
            results = eval_scib_metrics_onlybio(self.adata,batch_key,label_key, emb_name)
            return results             




