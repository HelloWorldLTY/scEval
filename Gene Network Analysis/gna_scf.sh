#!/bin/bash
# Genemodule
python get_embedding_h5ad.py --task_name ihatest --input_type singlecell --output_type gene --pool_type all --tgthighres f1 --data_path "/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/Immune_ALL_human.h5ad" --save_path ./examples/genemodule/ --pre_normalized F --demo
