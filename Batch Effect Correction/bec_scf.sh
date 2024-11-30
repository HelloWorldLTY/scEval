#!/bin/bash
# bec
python get_embedding_h5ad.py --task_name benchmark_trajectory --input_type singlecell --output_type cell --pool_type all --tgthighres a5 --data_path "/gpfs/gibbs/pi/zhao/tl688/trajectory_data/benchmark_trajectory.h5ad" --save_path ./examples/bec/ --pre_normalized F --version rde