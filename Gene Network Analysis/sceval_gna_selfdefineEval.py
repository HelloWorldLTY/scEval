import scanpy as sc
import numpy as np

import pandas as pd
from grn import GeneEmbedding
import seaborn as sns

adata = sc.read_h5ad("pbmc_tissue_gene_embeddings.h5ad")

#marker genes by COSG
mkr_set = {'Erythrocytes': ['KRT1', 'TMCC2', 'ARG1'],
 'Erythroid progenitors': ['GPT', 'SLC10A4', 'CCNE1'],
 'CD10+ B cells': ['AKAP12', 'CYGB', 'MME'],
 'Megakaryocyte progenitors': ['LY6G6F', 'PF4V1', 'CMTM5'],
 'HSPCs': ['CRHBP', 'ROBO4', 'NPR3'],
 'Monocyte progenitors': ['MS4A3', 'CTSG', 'AZU1'],
 'Plasmacytoid dendritic cells': ['KRT5', 'ASIP', 'EPHB1'],
 'CD20+ B cells': ['MS4A1', 'FCRL1', 'COL19A1'],
 'Plasma cells': ['CAV1', 'JSRP1', 'SPAG4'],
 'Monocyte-derived dendritic cells': ['ZNF366', 'CD1E', 'PKIB'],
 'CD14+ Monocytes': ['CYP1B1', 'NRG1', 'CYP27A1'],
 'CD16+ Monocytes': ['CDKN1C', 'HES4', 'LYPD2'],
 'CD4+ T cells': ['TSHZ2', 'CD40LG', 'TRAT1'],
 'CD8+ T cells': ['CD8B', 'S100B', 'NELL2'],
 'NK cells': ['KLRF1', 'KLRC1', 'NCAM1'],
 'NKT cells': ['SLC4A10', 'GZMK', 'LAG3']}


makerlist = []

for i in adata.obs['gene_name']:
    count = 0
    for ctp in mkr_set.keys():
        if i in mkr_set[ctp]:
            makerlist.append(ctp)
            count = 1
    if count ==0:
        makerlist.append(None)

adata.obs['new_marker'] = makerlist

sc.pl.umap(adata, color='new_marker', edges=True)


# specific pathway from scGPT suggestions
mole_list = pd.read_table("Participating Molecules [R-HSA-168256].tsv")

mole_list_dnarna = mole_list[ mole_list["MoleculeType"]  == 'DNA/RNA' ]

cofunction_gene = []
for i in mole_list_dnarna["MoleculeName"].values:
    gene = i.split(' ')[1]
    cofunction_gene.append(gene)

adata_HLA = adata_new[[True if ('HLA' in i )  else False for i in adata_new.obs['gene_name'].values]]
adata_CD = adata_new[[True if ('CD' in i) else False for i in adata_new.obs['gene_name'].values]]

intset_result = set(adata_HLA.obs['gene_name']).intersection(set(cofunction_gene))
print(len(intset_result) / len(adata_HLA))

intset_result = set(adata_CD.obs['gene_name']).intersection(set(cofunction_gene))
print(len(intset_result) / len(adata_CD))

