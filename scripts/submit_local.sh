#!/bin/bash

# CSV file containing hyperparameter configurations
CSV_FILE="../configs/$1_hyper_parameters.csv"

# Skip the header and iterate through each row
tail -n +2 $CSV_FILE | while IFS=, read -r train_size batch_size epochs num_points lr dropout
do
    # Generate an experiment name dynamically
    EXP_NAME="ahmedml_$1_ts${train_size}_bs${batch_size}_epochs${epochs}_pts${num_points}_lr${lr}_drop${dropout}_$(date +%Y%m%d_%H%M%S)"

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
