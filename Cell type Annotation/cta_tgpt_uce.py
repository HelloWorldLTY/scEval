import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode
import scanpy as sc
import sklearn
import warnings

sys.path.insert(0, "../")
import scgpt as scg

# extra dependency for similarity search
try:
    import faiss

    faiss_imported = True
except ImportError:
    faiss_imported = False
    print(
        "faiss not installed! We highly recommend installing it for fast similarity search."
    )
    print("To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")

warnings.filterwarnings("ignore", category=ResourceWarning)

ref_embed_adata = sc.read_h5ad("./tgpt_out/spaital_mouse_slideseqv2_tgpt_all.h5ad") # you can change it accordingly

ref_embed_adata

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report

train_obs,test_obs = train_test_split(
    ref_embed_adata.obs_names, random_state=42
)

adata_train = ref_embed_adata[train_obs]
adata_test = ref_embed_adata[test_obs]

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(adata_train.X, adata_train.obs.celltype) # or adata.obsm['emb'] for uce.

pred_label = clf.predict(adata_test.X)
true_label = adata_test.obs.celltype

print(classification_report(true_label, pred_label, digits=4))