import scanpy as sc
import numpy as np

import pandas as pd
from grn import GeneEmbedding
import seaborn as sns
import gseapy as gp
adata = sc.read_h5ad("pbmc_tissue_gene_embeddings.h5ad")

#marker genes defined by the original paper filtered based on expression profiles.
mkr_set = {'Erythrocytes': ['CST3'],
 'Erythroid progenitors': ['GATA2'],
 'CD10+ B cells': ['MME'],
 'Megakaryocyte progenitors': ['PF4',	'ITGA2B',	'PPBP'],
 'HSPCs': ['CD34',	'PROCR'],
 'Monocyte progenitors': ['IRF8',	'CSF1R',	'LY86'],
 'Plasmacytoid dendritic cells': ['GZMB',	'IL3RA'],
 'CD20+ B cells': ['MS4A1'],
 'Plasma cells': [],
 'Monocyte-derived dendritic cells': ['CD1C','FCER1A'],
 'CD14+ Monocytes': ['CD14'],
 'CD16+ Monocytes': ['FCGR3A'],
 'CD4+ T cells': ['CD4'],
 'CD8+ T cells': ['CD8B', 'CD8A'],
 'NK cells': ['NKG7','GNLY'],
          }

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

adata_new = adata
cofunction_gene = []
for i in mole_list_dnarna["MoleculeName"].values:
    gene = i.split(' ')[1]
    cofunction_gene.append(gene)

adata_HLA = adata_new[[True if ('HLA' in i )  else False for i in adata_new.obs['gene_name'].values]]
adata_CD = adata_new[[True if ('CD' in i) else False for i in adata_new.obs['gene_name'].values]]

CD_genes = adata_new.obs['gene_name'].values

# Meta info about the number of terms (tests) in the databases
df_database = pd.DataFrame(
data = [['GO_Biological_Process_2021', 6036],
['GO_Molecular_Function_2021', 1274],
['Reactome_2022', 1818]],
columns = ['dataset', 'term'])

# Select desired database for query; here use Reactome as an example
databases = ['Reactome_2022']
m = df_database[df_database['dataset'].isin(databases)]['term'].sum()
# p-value correction for total number of tests done
p_thresh = 0.05/m

# Perform pathway enrichment analysis using the gseapy package in the Reactome database
df = pd.DataFrame()
enr_Reactome = gp.enrichr(gene_list=CD_genes,
                          gene_sets=databases,
                          organism='Human',
                          outdir='test/enr_Reactome',
                          cutoff=0.5)
out = enr_Reactome.results
out = out[out['P-value'] < p_thresh]
df = df.append(out, ignore_index=True)
df



