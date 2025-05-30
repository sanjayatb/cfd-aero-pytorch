#!/bin/bash

# Directory for logs
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Load Conda environment
source ~/miniconda3/bin/activate python3

# Function to print usage
print_usage() {
    echo "Usage: $0 [batch_name] [hyper_parameters_file]"
    echo "Example: $0 Batch1 hyper_parameters.csv"
    exit 1
}

# Check for required arguments
if [[ $# -ne 2 ]]; then
    echo "Error: Invalid number of arguments."
    print_usage
fi

# Extract arguments
BATCH_NAME=$1
CSV_FILE=$2

# Ensure CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: File '$CSV_FILE' not found!"
    exit 1
fi

# Count total experiments
TOTAL_EXP=$(($(tail -n +2 "$CSV_FILE" | wc -l)))
if [[ "$TOTAL_EXP" -eq 0 ]]; then
    echo "Error: No experiments found in '$CSV_FILE'."
    exit 1
fi

# Initialize counter
RUNNING_COUNT=0

# Skip the header and iterate through each row
tail -n +2 "$CSV_FILE" | while IFS=, read -r dataset_name model_arch model_name sample_size batch_size epochs num_points lr dropout conv_layers fc_layers
do
    # Increment running experiment counter
    ((RUNNING_COUNT++))

    # Generate an experiment name dynamically
    EXP_NAME="$(date +%Y%m%d)_${BATCH_NAME}_${dataset_name}_${model_arch}_${model_name}_ds${sample_size}_bs${batch_size}_epochs${epochs}_pts${num_points}_lr${lr}_drop${dropout}_${conv_layers}_${fc_layers}"

    # Set unique log files per run
    OUTPUT_LOG="${LOG_DIR}/${EXP_NAME}.out"
    ERROR_LOG="${LOG_DIR}/${EXP_NAME}.err"

    # Print experiment progress
    echo "Submitting experiment $RUNNING_COUNT/$TOTAL_EXP: $EXP_NAME"
    echo "Output Log: $OUTPUT_LOG"
    echo "Error Log: $ERROR_LOG"

    # Submit job using SBATCH
    sbatch <<EOF
#!/bin/bash
#SBATCH -J $EXP_NAME
#SBATCH -p dzagnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --output=$OUTPUT_LOG
#SBATCH --error=$ERROR_LOG

source ~/miniconda3/bin/activate python3

python3 -u ../source/runner.py \
    --experiment-batch-name "$BATCH_NAME" \
    --dataset-name "$dataset_name" \
    --model-arch "$model_arch" \
    --model-name "$model_name" \
    --sample-size "$sample_size" \
    --batch-size "$batch_size" \
    --epochs "$epochs" \
    --num-points "$num_points" \
    --lr "$lr" \
    --dropout "$dropout" \
    --exp-name "$EXP_NAME" \
    --conv-layers "$conv_layers" \
    --fc-layers "$fc_layers"

EOF

    echo "Submitting experiment $RUNNING_COUNT/$TOTAL_EXP completed: $EXP_NAME"
    sleep 5
done
