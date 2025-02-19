import pandas as pd
import torch

from train.pointnet_train import extract_features_and_outputs, load_and_test_model, train_and_evaluate, setup_seed, \
    config, initialize_model, get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
print(device)


def model_train_and_evaluate():
    setup_seed(config['seed'])

    # List of fractions of the training data to use
    train_fractions = [1]
    results = {}

    for frac in train_fractions:
        print(f"Training on {frac * 100}% of the training data")
        print(device)
        model = initialize_model(config).to(device)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config['dataset_path'], config['aero_coeff'], config['subset_dir'],
            config['num_points'], config['batch_size'], train_frac=frac
        )

        best_model_path, final_model_path = train_and_evaluate(model, train_dataloader, val_dataloader, config)

        # Test the best model
        print("Testing the best model:")
        best_results = load_and_test_model(best_model_path, test_dataloader, device)

        # Test the final model
        print("Testing the final model:")
        final_results = load_and_test_model(final_model_path, test_dataloader, device)

        # Store results
        results[f"{int(frac * 100)}%_best"] = best_results
        results[f"{int(frac * 100)}%_final"] = final_results
    #    best_model_path = './outputs/models/'+config['exp_name']+'.pth'
     #   outputs, features = extract_features_and_outputs(best_model_path, train_dataloader, device, config)

    # Save the results to a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv('model_training_results_PC_normalized.csv')
    print("Results saved to model_training_results.csv")


if __name__ == '__main__':
    model_train_and_evaluate()
