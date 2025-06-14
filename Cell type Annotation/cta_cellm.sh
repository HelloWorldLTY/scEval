#!/bin/bash
#SBATCH --job-name=celllm_cross_MBSpatial
#SBATCH --output=celllm_cross_MBSpatial.txt
DEVICE=0
python tasks/cell_task/ctc.py \
--device ${DEVICE} \
--config_path ./configs/ctc/cellLM.json \
--dataset MBSpatial_raw \
--dataset_path ../../scgpt_dataset \
--output_path ../ckpts/finetune_ckpts/celllm_cl_fintune.pth \
--mode train \
--epochs 10 \
--batch_size 3 \
--logging_steps 1000 \
--gradient_accumulation_steps 4 \
--patience 10
