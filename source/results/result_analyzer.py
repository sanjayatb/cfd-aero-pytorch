import matplotlib.pyplot as plt
import pandas as pd


class ResultViewer:

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
                xytext=(best_training_size - 100, best_test_mse + 0.02),  # Offset for better visibility
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
        plt.ylabel("MSE")
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
