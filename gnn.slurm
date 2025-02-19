#!/bin/bash
#SBATCH -J ahmedML-gnn
#SBATCH -p dzagnormal
#SBATCH -N 1
##SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --output=logs/DragPrediction_AhemedML_DragGNN_XL_r2_3000_0.0_1.0_batchsize16_10epochs_2kpoints_tsne_NeurIPS.out
#SBATCH --error=logs/DragPrediction_AhemedML_DragGNN_XL_r2_3000_0.0_1.0_batchsize16_10epochs_2kpoints_tsne_NeurIPS.err

source ~/miniconda3/bin/activate python3


python3 -u runner_gnn.py   # xxx.py 改为您的输入文件名#

