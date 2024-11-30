from torch_geometric.loader import DataLoader
from gears_001 import PertData, GEARS
from gears_001.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears_001.utils import create_cell_graph_dataset_for_prediction

import scanpy as sc
import numpy as np
import sklearn



from sklearn.preprocessing import StandardScaler
def model_training(adata_train, emb_name = 'X_uce'):
    model = sklearn.linear_model.LinearRegression()
    train_data = np.concatenate([adata_train.obsm[emb_name], adata_train.obs['pert_condition'].values.reshape(-1,1)], axis=1)
    pred_data = adata_train.obsm['ground_truth']
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    model.fit(train_data, pred_data)
    return model,scaler



adata = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/scGPT/examples/tgpt_out/adata_train_adamson_tgpt_all.h5ad") #can replace it with other embeddings
adata.obsm['ground_truth'] = adata.layers['ground_truth'].copy()
model,scaler = model_training(adata, emb_name = 'X_tgpt')

from gears import PertData, GEARS

# get data
pert_data = PertData('./data')
# pert_data = PertData('./data_folder')
# load dataset in paper: norman, adamson, dixit.
pert_data.load(data_name = 'adamson')
# specify data split
pert_data.prepare_split(split = 'simulation', seed = 1)
# get dataloader with batch size
pert_data.get_dataloader(batch_size = 1024, test_batch_size = 1024)

adata = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/scGPT/examples/tgpt_out/adata_test_adamson_tgpt_all.h5ad")

adata.obsm['ground_truth'] = adata.layers['ground_truth'].copy()



import torch

def eval_perturb(
    loader: DataLoader, adata, model, scaler,obsm_name = 'X_uce'
):
    """
    Run model in inference mode using a given data loader
    """

    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        pert_cat.extend(batch.pert)
        
        adata_filter = adata[itr*1024:(itr+1)*1024]
        test_data = np.concatenate([adata_filter.obsm[obsm_name], adata_filter.obs['pert_condition'].values.reshape(-1,1)], axis=1)
        test_data = scaler.transform(test_data)
        p = model.predict(test_data)
#         print(p)
        t = batch.y.numpy()
        pred.extend(p)
        truth.extend(t)
        # Differentially expressed genes
        for itr, de_idx in enumerate(batch.de_idx):
            pred_de.append(p[itr, de_idx])
            truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = np.stack(pred)
    truth = np.stack(truth)
    results["pred"] = pred
    results["truth"] = truth

    pred_de = np.stack(pred_de)
    truth_de = np.stack(truth_de)
    results["pred_de"] = pred_de
    results["truth_de"] = truth_de

    return results

results = eval_perturb(pert_data.dataloader['test_loader'],adata,model,scaler, obsm_name = 'X_tgpt')

test_metrics, test_pert_res = compute_metrics(results)
print(test_metrics)


