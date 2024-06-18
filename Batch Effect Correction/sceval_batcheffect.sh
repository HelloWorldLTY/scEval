#!/bin/bash

#SBATCH --job-name=ft_job_bec_sceval
#SBATCH --output=ft_job_bec_sceval.txt
python sceval_batcheffect.py --dataset "/gpfs/ysm/pi/zhao/tl688/datasets/HumanPBMC_raw.loom"