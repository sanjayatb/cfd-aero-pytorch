#!/bin/bash
#SBATCH -J pointnet_grid
#SBATCH -p dzagnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/logs/pointnet_grid.out
#SBATCH --error=outputs/logs/pointnet_grid.err

# Load Conda environment
#source ~/miniconda3/bin/activate python3

# CSV file containing hyperparameter configurations
CSV_FILE="../configs/pointnet_hyper_parameters.csv"

# Skip the header and iterate through each row
tail -n +2 $CSV_FILE | while IFS=, read -r train_size batch_size epochs num_points lr dropout
do
    # Generate an experiment name dynamically
    EXP_NAME="ahmedml_pointnet_ts${train_size}_bs${batch_size}_epochs${epochs}_pts${num_points}_lr${lr}_drop${dropout}_$(date +%Y%m%d_%H%M%S)"

    # Print info
    echo "Running experiment: $EXP_NAME"

    # Run the Python script with parameters
    python3 -u ../source/runner.py \
        --batch-size "$batch_size" \
        --epochs "$epochs" \
        --num-points "$num_points" \
        --lr "$lr" \
        --dropout "$dropout" \
        --exp-name "$EXP_NAME"

    echo "Experiment $EXP_NAME completed."
done
