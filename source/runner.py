import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.config.loader import load_config, override_configs
from source.data.ahmed_ml_dataset import AhmedMLDataset
from source.data.dataset_loaders import DatasetLoaders
from source.model.pointnet import RegPointNet
from source.trainer.model_trainer import ModelTrainer
from source.trainer.pointnet_trainer import PointNetTrainer

from source.utils.env import setup_seed


class Runner:
    trainer: ModelTrainer

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Run Experiment")

        parser.add_argument("--data-name", type=str, required=False, default="AhmedML")
        parser.add_argument("--train-size", type=int, required=False)
        parser.add_argument("--batch-size", type=int, required=False)
        parser.add_argument("--epochs", type=int, required=False)
        parser.add_argument("--num-points", type=int, required=False)
        parser.add_argument("--lr", type=float, required=False)
        parser.add_argument("--dropout", type=float, required=False)
        parser.add_argument("--exp-name", type=str, required=False, default=None)

        args = parser.parse_args()
        override_configs(config, args)

    def run(self):
        setup_seed(config.environment.seed)

        self.trainer.setup_data()

        model = self.trainer.init_model()
        self.trainer.train_and_evaluate(model)

        self.trainer.load_and_test()


def model_train_and_evaluate():
    from source.train.pointnet_train import train_and_evaluate, initialize_model
    setup_seed(config.environment.seed)

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
        best_model_path = './outputs/models/' + config['exp_name'] + '.pth'
        outputs, features = extract_features_and_outputs(best_model_path, train_dataloader, device, config)

    # Save the results to a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv('model_training_results_PC_normalized.csv')
    print("Results saved to model_training_results.csv")


if __name__ == "__main__":
    config = load_config("../configs/pointnet_config.yml")

    loader = DatasetLoaders(config=config,
                            dataset=AhmedMLDataset(root_dir=config.data.stl_path,
                                                   csv_file=config.data.target_data_path,
                                                   num_points=config.parameters.data.num_points,
                                                   pointcloud_exist=False))

    runner = Runner(trainer=PointNetTrainer(config, RegPointNet, loader))

    runner.parse_args()
    runner.run()
