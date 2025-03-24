import os.path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score

from source.trainer.trainer_factory import TrainerFactory


class ResultViewer:

    @staticmethod
    def plot_true_vs_predicted_csv(folder, file_name):
        df = pd.read_csv(f"{folder}{file_name}")

        true_values = df["Targets"]
        predicted_values = df["Prediction"]

        r2 = r2_score(true_values, predicted_values)
        mse = mean_squared_error(true_values, predicted_values)

        def format_number(num):
            if abs(num) >= 1e3 or abs(num) <= 1e-3:  # Scientific notation condition
                return f"{num:.4e}"
            return f"{num:.4f}"

        r2_str = format_number(r2)
        mse_str = format_number(mse)

        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predicted_values, alpha=0.6, edgecolors="k", label="Predictions")
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], "r--",
                 label="Ideal Fit")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"True vs. Predicted Values (R2 = {r2_str}, MSE = {mse_str})")
        plt.legend()
        plt.grid()

        pattern = re.compile(
            r"(?P<date>\d{4}-\d{2}-\d{2})_"  # Match date (YYYY-MM-DD)
            r"(?P<metric>\w+)_"  # Match metric (e.g., Drag)
            r"(?P<timestamp>\d{8})_"  # Match timestamp (YYYYMMDD)
            r"(?P<model_type>[\w]+)_"  # Match model type (e.g., drivaer)
            r"best_variable_train_size_"  # Explicitly match this keyword
            r"(?P<dataset>\w+)_"  # Match dataset (e.g., DrivAerML)
            r"(?P<arch>\w+)_"  # Match model architecture (e.g., PointNet)
            r"(?P<model>\w+)_"  # Match model name (e.g., SimplePointNet)
            r"ts(?P<training_size>\d+)_"
            r"bs(?P<batch_size>\d+)_"
            r"epoc?h?s?(?P<epochs>\d+)_?"  # Allows both 'epochs' and 'epochs'
            r"pts(?P<points>\d+)_"
            r"lr(?P<learning_rate>[\d.]+)_"
            r"drop(?P<dropout>[\d.]+)_"
    r"\[(?P<conv_layers>[\d_]+)]_\[(?P<fc_layers>[\d_]+)]"  # Handles _ instead of :
        )

        # ✅ Extract components
        match = pattern.match(file_name)
        text_info = ""
        if match:
            extracted_data = match.groupdict()
            extracted_data["conv_layers"] = extracted_data["conv_layers"].replace(":", " → ")  # Format Conv layers
            extracted_data["fc_layers"] = extracted_data["fc_layers"].replace(":", " → ")  # Format FC layers

            # ✅ Print extracted data
            for key, value in extracted_data.items():
                if key in ("model_type", "timestamp"):
                    continue
                text_info += f"{key}: {value}\n"
        else:
            print("Filename format does not match.")

        plt.text(0.05, 0.95, text_info, fontsize=10, verticalalignment="top", transform=plt.gca().transAxes,
                 bbox=dict(facecolor="white", alpha=0.6))

        plt.savefig(f"../outputs/plots/drivaer_best_model_r2_{r2_str}.png")

    @staticmethod
    def plot_true_vs_predicted(config, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainer = TrainerFactory.get_pointnet_trainer(config)
        trainer.setup_data()

        model = trainer.init_model()
        param_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for key, value in param_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()  # Set model to evaluation mode

        test_loader = trainer.data_loaders.test_dataloader

        # 🚀 Run Inference
        true_values, predicted_values = [], []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device).squeeze()
                outputs = model(data).squeeze()

                true_values.extend(targets.cpu().numpy())
                predicted_values.extend(outputs.cpu().numpy())

        # 🚀 Convert to NumPy for Plotting
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)

        # 🚀 Plot True vs Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(true_values, predicted_values, alpha=0.6, edgecolors="k", label="Predictions")
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], "r--",
                 label="Ideal Fit")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("True vs. Predicted Values")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def training_size_vs_mse(result_file_path):
        # Load the CSV file
        df = pd.read_csv(result_file_path)

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Group by dataset and plot both Test MSE & Train MSE
        for i, (dataset_name, group) in enumerate(df.groupby("dataset")):
            # Sort the group by training_size before plotting
            group = group.sort_values(by="training_size")

            # Plot Train MSE (solid line)
            plt.plot(group["training_size"], group["Train Best MSE"], marker="o", linestyle="-",
                     label=f"{dataset_name} (Train)")

            # Plot Test MSE (dashed line)
            plt.plot(group["training_size"], group["Test MSE"], marker="s", linestyle="--",
                     label=f"{dataset_name} (Test)")

            # Find the best (lowest) Test MSE and corresponding training size
            best_row = group.loc[group["Test MSE"].idxmin()]
            best_test_mse = best_row["Test MSE"]
            best_training_size = best_row["training_size"]

            # Annotate the best Test MSE point on the plot
            plt.annotate(
                f"{dataset_name}: Best Test MSE: {best_test_mse:.4f}, Size: {best_training_size}",
                xy=(best_training_size, best_test_mse),
                xytext=(best_training_size - 100, best_test_mse + 0.00001),  # Offset for better visibility
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red"
            )

        # Labels and title
        plt.xlabel("Training Set Size")
        plt.ylabel("MSE")
        plt.title(f"Train & Test MSE over Training Set Size.")

        # Add legend for different datasets
        plt.legend(title="Dataset & MSE Type")
        # Enable minor ticks and set subgrid spacing
        plt.minorticks_on()
        plt.grid(True, which='major', linestyle='-', linewidth=0.8)  # Main grid
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5)  # Subgrid

        # Set minor tick spacing (approximation, adjust if needed)
        plt.xticks(range(int(df["training_size"].min()), int(df["training_size"].max()) + 20, 20))

        # Extract constant hyperparameters (assuming all rows have the same hyperparameters)
        first_row = df.iloc[0]
        hyper_params_text = (
            f"Model Arch: {first_row['Model Arc']}\n"
            f"Model Name: {first_row['Model']}\n"
            f"Number of points: {first_row['num_points']}\n"
            f"Learning Rate: {first_row['lr']}\n"
            f"Batch Size: {first_row['batch_size']}\n"
            f"Epochs: {first_row['epochs']}\n"
            f"Dropout: {first_row['dropout']}\n"
            f"Optimizer: {first_row['optimizer']}"
        )

        # Add the hyperparameters as text on the plot
        plt.text(
            0.02, 0.98, hyper_params_text,
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black")
        )

        # Save the plot
        plt.savefig("../outputs/plots/train_test_mse_over_training_size.png")

    @staticmethod
    def training_size_vs_r2(result_file_path):
        # Load the CSV file
        df = pd.read_csv(result_file_path)

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Group by dataset and plot both Test MSE & Train MSE
        for dataset_name, group in df.groupby("dataset"):
            # Sort the group by training_size before plotting
            group = group.sort_values(by="training_size")

            # Plot Train MSE (solid line)
            plt.plot(group["training_size"], group["Best model R2"], marker="o", linestyle="-",
                     label=f"{dataset_name} (Train)")

            # Plot Test MSE (dashed line)
            plt.plot(group["training_size"], group["Test R2"], marker="s", linestyle="--",
                     label=f"{dataset_name} (Test)")

        # Labels and title
        plt.xlabel("Training Set Size")
        plt.ylabel("R2")
        plt.title(f"Train & Test R2 over Training Set Size.")

        # Add legend for different datasets
        plt.legend(title="Dataset & MSE Type")
        plt.grid(True)

        # Extract constant hyperparameters (assuming all rows have the same hyperparameters)
        first_row = df.iloc[0]
        hyper_params_text = (
            f"Model Arch: {first_row['Model Arc']}\n"
            f"Model Name: {first_row['Model']}\n"
            f"Number of points: {first_row['num_points']}\n"
            f"Learning Rate: {first_row['lr']}\n"
            f"Batch Size: {first_row['batch_size']}\n"
            f"Epochs: {first_row['epochs']}\n"
            f"Dropout: {first_row['dropout']}\n"
            f"Optimizer: {first_row['optimizer']}"
        )

        # Add the hyperparameters as text on the plot
        plt.text(
            0.02, 0.98, hyper_params_text,
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black")
        )

        # Save the plot
        plt.savefig("../outputs/plots/train_test_r2_over_training_size.png")

    @staticmethod
    def num_of_points_vs_mse(result_file_path):
        # Load the CSV file
        df = pd.read_csv(result_file_path)

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Group by dataset and plot both Test MSE & Train MSE
        for i, (dataset_name, group) in enumerate(df.groupby("dataset")):
            # Sort the group by training_size before plotting
            group = group.sort_values(by="num_points")

            # Plot Train MSE (solid line)
            plt.plot(group["num_points"], group["Train Best MSE"], marker="o", linestyle="-",
                     label=f"{dataset_name} (Train)")

            # Plot Test MSE (dashed line)
            plt.plot(group["num_points"], group["Test MSE"], marker="s", linestyle="--",
                     label=f"{dataset_name} (Test)")

            # Find the best (lowest) Test MSE and corresponding training size
            best_row = group.loc[group["Test MSE"].idxmin()]
            best_test_mse = best_row["Test MSE"]
            best_num_points = best_row["num_points"]

            # Annotate the best Test MSE point on the plot
            plt.annotate(
                f"{dataset_name}: Best Test MSE: {best_test_mse:.4f}, Size: {best_num_points}",
                xy=(best_num_points, best_test_mse),
                xytext=(best_num_points - 50000, best_test_mse + 0.01),  # Offset for better visibility
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=9, color="red"
            )

        # Labels and title
        plt.xlabel("Number of points")
        plt.ylabel("MSE")
        plt.title(f"Train & Test MSE over Number of points.")
        plt.grid(True)
        # Add legend for different datasets
        plt.legend(title="Dataset & MSE Type")
        # Enable minor ticks and set subgrid spacing

        # Extract constant hyperparameters (assuming all rows have the same hyperparameters)
        first_row = df.iloc[0]
        hyper_params_text = (
            f"Model Arch: {first_row['Model Arc']}\n"
            f"Model Name: {first_row['Model']}\n"
            f"Training Size: {first_row['training_size']}\n"
            f"Learning Rate: {first_row['lr']}\n"
            f"Batch Size: {first_row['batch_size']}\n"
            f"Epochs: {first_row['epochs']}\n"
            f"Dropout: {first_row['dropout']}\n"
            f"Optimizer: {first_row['optimizer']}"
        )

        # Add the hyperparameters as text on the plot
        plt.text(
            0.02, 0.98, hyper_params_text,
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black")
        )

        # Save the plot
        plt.savefig("../outputs/plots/train_test_mse_over_num_points.png")

    @staticmethod
    def num_of_points_vs_r2(result_file_path):
        # Load the CSV file
        df = pd.read_csv(result_file_path)

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Group by dataset and plot both Test MSE & Train MSE
        for dataset_name, group in df.groupby("dataset"):
            # Sort the group by training_size before plotting
            group = group.sort_values(by="num_points")

            # Plot Train MSE (solid line)
            plt.plot(group["num_points"], group["Best model R2"], marker="o", linestyle="-",
                     label=f"{dataset_name} (Train)")

            # Plot Test MSE (dashed line)
            plt.plot(group["num_points"], group["Test R2"], marker="s", linestyle="--",
                     label=f"{dataset_name} (Test)")

        # Labels and title
        plt.xlabel("Number of points")
        plt.ylabel("MSE")
        plt.title(f"Train & Test R2 over Number of points.")

        # Add legend for different datasets
        plt.legend(title="Dataset & MSE Type")
        plt.grid(True)

        # Extract constant hyperparameters (assuming all rows have the same hyperparameters)
        first_row = df.iloc[0]
        hyper_params_text = (
            f"Model Arch: {first_row['Model Arc']}\n"
            f"Model Name: {first_row['Model']}\n"
            f"Training Size: {first_row['training_size']}\n"
            f"Learning Rate: {first_row['lr']}\n"
            f"Batch Size: {first_row['batch_size']}\n"
            f"Epochs: {first_row['epochs']}\n"
            f"Dropout: {first_row['dropout']}\n"
            f"Optimizer: {first_row['optimizer']}"
        )

        # Add the hyperparameters as text on the plot
        plt.text(
            0.02, 0.98, hyper_params_text,
            transform=plt.gca().transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black")
        )

        # Save the plot
        plt.savefig("../outputs/plots/train_test_r2_over_num_points.png")

    def plot_r2_for_model(self, result_file_path):
        # Load the CSV file
        df = pd.read_csv(result_file_path)

        # Create the plot
        plt.figure(figsize=(8, 6))
        grouped = df.groupby(["Model", "dataset"])["Best model R2"].mean().unstack()
        # Plot grouped data
        grouped.plot(kind="bar", figsize=(10, 5))

        # Customize plot
        plt.xlabel("Model")
        plt.ylabel("R² Score")
        plt.title("R² Scores for Different Models and Datasets")
        plt.legend(title="Dataset")
        plt.xticks(rotation=10)
        plt.grid(axis="y")

        # Save the plot
        plt.savefig("../outputs/plots/r2_over_each_model.png")
