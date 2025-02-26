import os

import pandas as pd
import torch

from source.config.dto import Config


class ResultCollector:
    config: Config

    def __init__(self, config):
        self.config = config

    def save_best_model(self, model_dict, best_mse):
        best_model_path = os.path.join(self.config.outputs.model.best_model_path,
                                       f'{self.config.exp_name}_best_model.pth')
        torch.save(model_dict, best_model_path)
        print(f"New best model saved with MSE: {best_mse:.6f}")

    def save_test_scores(self, scores):
        hyperparameters = self.config.parameters.data.__dict__ | self.config.parameters.model.__dict__

        ### TODO select parameter
        hyperparameters.pop("k")
        hyperparameters.pop("channels")
        hyperparameters.pop("linear_sizes")
        hyperparameters.pop("output_channels")

        new_entry = {"Model": self.config.model_name,
                     "Dataset": self.config.data.name, **hyperparameters, **scores,
                     "best_model": f'{self.config.exp_name}_best_model.pth'}

        # Define CSV file path
        csv_filename = os.path.join(self.config.outputs.model.best_scores_path,
                                    f"{self.config.model_name}_{self.config.data.name}_set_of_experiment_scores.csv")
        os.makedirs(self.config.outputs.model.best_scores_path, exist_ok=True)
        # Check if file exists
        if os.path.exists(csv_filename):
            # Load existing data
            df_existing = pd.read_csv(csv_filename)

            # Extract only hyperparameter columns
            hyperparam_keys = list(hyperparameters.keys())
            hyperparam_keys.append("Model")
            hyperparam_keys.append("Dataset")

            df_new = pd.DataFrame([new_entry])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

            df_unique = df_combined.drop_duplicates(subset=hyperparam_keys, keep='first')
            df_unique.to_csv(csv_filename, index=False)
            print("New experiment results appended to CSV.")
        else:
            # Create a new DataFrame and save it
            df_new = pd.DataFrame([new_entry])
            df_new.to_csv(csv_filename, index=False)
            print("New CSV file created and experiment results saved.")
