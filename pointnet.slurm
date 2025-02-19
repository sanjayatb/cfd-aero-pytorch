#!/bin/bash
#SBATCH -J ahmedML-pointnet
#SBATCH -p dzagnormal
#SBATCH -N 1
##SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --output=logs/DragPrediction_AhemedML_PointNet_r2_3000_0.0_1.0_batchsize16_200epochs_1.5kpoints_tsne_NeurIPS_test_2.out
#SBATCH --error=logs/DragPrediction_AhemedML_PointNet_r2_3000_0.0_1.0_batchsize16_200epochs_1.5kpoints_tsne_NeurIPS_test_2.err

source ~/miniconda3/bin/activate python3


python3 -u runner.py   # xxx.py 改为您的输入文件名#
