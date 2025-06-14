#!/bin/bash

#SBATCH --job-name=uce_getemb
#SBATCH --output=uce_getemb.txt

#change batch size does not handle large-scale dataset.

python eval_single_anndata.py --adata_path "/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/PBMC368k_raw.h5ad" --dir ./uce_emb_data/ --batch_size 50