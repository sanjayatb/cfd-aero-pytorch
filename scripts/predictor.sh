#!/bin/bash

EXP_NAME=$1
LOG_DIR="logs"
mkdir -p $LOG_DIR

OUTPUT_LOG="${LOG_DIR}/${EXP_NAME}.out"
ERROR_LOG="${LOG_DIR}/${EXP_NAME}.err"

# Submit job using SBATCH
sbatch <<EOF
#!/bin/bash
#SBATCH -J $EXP_NAME
#SBATCH -p dzagnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --output=$OUTPUT_LOG
#SBATCH --error=$ERROR_LOG

source ~/miniconda3/bin/activate python3

python3 -u ../source/predictor.py
EOF
