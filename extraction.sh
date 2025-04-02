#!/bin/bash
#SBATCH -A extraction_feature
#SBATCH --partition=cpu
#SBATCH --job-name=extraction_feature
#SBATCH --time=128:00:00
#SBATCH --mem-per-cpu=300000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drgdk8@inf.elte.hu
srun -p cpu -c 8 --mem-per-cpu=2000 --pty bash --mail-type=ALL --mail-user=drgdk8@inf.elte.hu

source ~/venv/bin/activate

conda activate mynmp

python /home/p_zhuzy/p_zhu/NMP/preprocess/extract_vgg_feature.py --dataset=vg --data_type=rela