#!/bin/bash
#SBATCH -A p_zhu
#SBATCH --partition=cpu
#SBATCH --job-name=testam_training
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=20000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drgdk8@inf.elte.hu
#SBATCH --nodes=1

 


python /home/p_zhuzy/p_zhu/NMP/train_vrd.py --encoder=nmp --use-loc --mode=whole --feat-mode=full