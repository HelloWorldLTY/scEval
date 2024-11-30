# Environment settings
import scanpy as sc
sc.set_figure_params(dpi=100)
import time
import warnings
warnings.filterwarnings('ignore')

import src.scimilarity

from src.scimilarity.utils import lognorm_counts
from src.scimilarity import CellAnnotation, align_dataset

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report

annotation_path = './models/annotation_model_v1'
ca = CellAnnotation(model_path=annotation_path)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total':total_num, 'Trainable':trainable_num}

get_parameter_number(ca.model)


# Load the tutorial data.
# Replace data_path with your local file path.
# sc.read("/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/aizarani_liver.h5ad")
t1 = time.time()
data_path = "/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/aizarani_liver.h5ad"
adams = sc.read(data_path)
adams.layers['counts'] = adams.X
# adams.var_names =[i.upper() for i in adams.var_names]
adams = align_dataset(adams, ca.gene_order)
adams = lognorm_counts(adams)

print(adams)

adams.obsm['X_scimilarity'] = ca.get_embeddings(adams.X)

predictions, nn_idxs, nn_dists, nn_stats = ca.get_predictions_kNN(adams.obsm['X_scimilarity'])
adams.obs['predictions_unconstrained'] = predictions.values

celltype_counts = adams.obs.predictions_unconstrained.value_counts()
well_represented_celltypes = celltype_counts[celltype_counts>20].index
t2 = time.time()


print("time", t2-t1)
import resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  / (1e6) )
import torch
print(torch.cuda.max_memory_allocated()/1024/1024/1024)
