#!/bin/bash

#SBATCH -A p_zhu
#SBATCH --partition=cpu
#SBATCH --job-name=extraction_feature
#SBATCH --time=128:00:00
#SBATCH --mem-per-cpu=20000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drgdk8@inf.elte.hu
#SBATCH --nodes=1


source ~/venv/bin/activate

conda activate mynmp

python /home/p_zhuzy/p_zhu/NMP/preprocess/extract_vgg_feature.py --dataset=vg --data_type=rela