#!/bin/bash

# CSV file containing hyperparameter configurations
CSV_FILE="../configs/hyper_parameters.csv"

# Function to print usage instructions
print_usage() {
    echo "Usage: $0 [dataset_name] [model_arch] [model_name]"
    echo "  dataset_name: Optional. Filter experiments by dataset name."
    echo "  model_arch:   Optional. Filter experiments by model architecture."
    echo "  model_name:   Optional. Filter experiments by model name."
    echo "Example:"
    echo "  $0 AhmedML"
    echo "  $0 AhmedML PointNet"
    echo "  $0 AhmedML PointNet RegPointNet"
    exit 1
}

# Check for too many arguments
if [[ $# -gt 3 ]]; then
    echo "Error: Too many arguments provided."
    print_usage
fi

# Extract command-line arguments (optional filters)
FILTER_DATASET_NAME=$1
FILTER_MODEL_ARCH=$2
FILTER_MODEL_NAME=$3

# Validate arguments against the CSV headers
VALID_DATASET_NAMES=$(tail -n +2 $CSV_FILE | cut -d, -f1 | sort -u)
VALID_MODEL_ARCHS=$(tail -n +2 $CSV_FILE | cut -d, -f2 | sort -u)
VALID_MODEL_NAMES=$(tail -n +2 $CSV_FILE | cut -d, -f3 | sort -u)

# Function to check if a value is valid
is_valid_value() {
    local value=$1
    local valid_list=$2
    if [[ -n "$value" && ! $(echo "$valid_list" | grep -Fx "$value") ]]; then
        return 1
    fi
    return 0
}

# Validate input arguments
if ! is_valid_value "$FILTER_DATASET_NAME" "$VALID_DATASET_NAMES"; then
    echo "Error: Invalid dataset_name '$FILTER_DATASET_NAME'"
    print_usage
fi

if ! is_valid_value "$FILTER_MODEL_ARCH" "$VALID_MODEL_ARCHS"; then
    echo "Error: Invalid model_arch '$FILTER_MODEL_ARCH'"
    print_usage
fi

if ! is_valid_value "$FILTER_MODEL_NAME" "$VALID_MODEL_NAMES"; then
    echo "Error: Invalid model_name '$FILTER_MODEL_NAME'"
    print_usage
fi

# Read the CSV and filter dynamically
tail -n +2 $CSV_FILE | while IFS=, read -r dataset_name model_arch model_name train_size batch_size epochs num_points lr dropout
do
    # Apply filtering if arguments are provided
    if [[ -n "$FILTER_DATASET_NAME" && "$dataset_name" != "$FILTER_DATASET_NAME" ]]; then
        continue
    fi
    if [[ -n "$FILTER_MODEL_ARCH" && "$model_arch" != "$FILTER_MODEL_ARCH" ]]; then
        continue
    fi
    if [[ -n "$FILTER_MODEL_NAME" && "$model_name" != "$FILTER_MODEL_NAME" ]]; then
        continue
    fi

    # Generate an experiment name dynamically
    EXP_NAME="$(date +%Y%m%d)_${dataset_name}_${model_arch}_${model_name}_ts${train_size}_bs${batch_size}_epochs${epochs}_pts${num_points}_lr${lr}_drop${dropout}"

    # Print info
    echo "Running experiment: $EXP_NAME"

    # Run the Python script with parameters
    python3 -u ../source/runner.py \
    --dataset-name "$dataset_name" \
    --model-arch "$model_arch" \
    --model-name "$model_name" \
    --train-size "$train_size" \
    --batch-size "$batch_size" \
    --epochs "$epochs" \
    --num-points "$num_points" \
    --lr "$lr" \
    --dropout "$dropout" \
    --exp-name "$EXP_NAME"

    echo "Experiment $EXP_NAME completed."
done
