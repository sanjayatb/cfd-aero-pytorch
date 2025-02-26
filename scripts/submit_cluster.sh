#!/bin/bash
#SBATCH -J ahmedML_grid
#SBATCH -p dzagnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2

# Dynamically generated log files per run
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Load Conda environment
source ~/miniconda3/bin/activate python3

# CSV file containing hyperparameter configurations
CSV_FILE="../configs/pointnet_hyper_parameters.csv"

# Skip the header and iterate through each row
tail -n +2 $CSV_FILE | while IFS=, read -r train_size batch_size epochs num_points lr dropout
do
    # Generate an experiment name dynamically
    EXP_NAME="ahmedml_pointnet_ts${train_size}_bs${batch_size}_epochs${epochs}_pts${num_points}_lr${lr}_drop${dropout}_$(date +%Y%m%d)"

    # Set unique log files per run
    OUTPUT_LOG="${LOG_DIR}/${EXP_NAME}.out"
    ERROR_LOG="${LOG_DIR}/${EXP_NAME}.err"

    # Print info
    echo "Running experiment: $EXP_NAME"
    echo "Output Log: $OUTPUT_LOG"
    echo "Error Log: $ERROR_LOG"

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

python3 -u ../source/runner.py \
    --train-size "$train_size" \
    --batch-size "$batch_size" \
    --epochs "$epochs" \
    --num-points "$num_points" \
    --lr "$lr" \
    --dropout "$dropout" \
    --exp-name "$EXP_NAME"

EOF

    echo "Experiment $EXP_NAME submitted."
done
