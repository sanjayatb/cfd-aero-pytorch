import json
import os
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from source.config.dto import Config
import pyvista as pv

class ResultCollector:
    config: Config

    def __init__(self, config):
        self.config = config
        self.result_dict = {}
        self.test_data_loader = None
        self.indices = []
        self.all_data = []
        self.targets = []
        self.predictions = []

    def add_r2_scores(self, score):
        if "r2_score" not in self.result_dict:
            self.result_dict["r2_score"] = [score]
        else:
            self.result_dict.get("r2_score").append(score)

    def add_rel_l2_scores(self, score):
        if "rel_l2_score" not in self.result_dict:
            self.result_dict["rel_l2_score"] = [score]
        else:
            self.result_dict.get("rel_l2_score").append(score)

    def save_best_model(self, model_dict, best_mse):
        best_model_path = os.path.join(
            self.config.outputs.model.best_model_path,
            f"{self.config.exp_name}_best_model.pth",
        )
        os.makedirs(self.config.outputs.model.best_model_path, exist_ok=True)
        torch.save(model_dict, best_model_path)
        print(f"New best model saved with MSE: {best_mse:.6f}")

    def save_test_scores(self, scores, predictor=False):
        raw_hyperparameters = (
            self.config.parameters.data.__dict__ | self.config.parameters.model.__dict__
        )

        ### TODO select parameter
        raw_hyperparameters.pop("k")
        raw_hyperparameters.pop("output_channels")
        if self.config.parameters.data.data_id_load_random:
            raw_hyperparameters.pop("training_size")
            raw_hyperparameters.pop("validation_size")
            raw_hyperparameters.pop("test_size")

        def _make_serializable(value):
            if isinstance(value, (list, tuple, dict)):
                return json.dumps(value, sort_keys=True)
            return value

        new_entry = {
            "Model Arc": self.config.model_arch,
            "Model": self.config.model_name,
            **{k: _make_serializable(v) for k, v in raw_hyperparameters.items()},
            **scores,
            "best_model": f"{self.config.exp_name}_best_model.pth",
        }

        # Define CSV file path
        date = datetime.now().strftime("%Y-%m-%d")
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)

        csv_filename = os.path.join(
            self.config.outputs.model.best_scores_path,
            f"{date}_{dataset_conf.target_col_alias}_{self.config.experiment_batch_name}_experiment_scores.csv",
        )
        os.makedirs(self.config.outputs.model.best_scores_path, exist_ok=True)
        # Check if file exists
        if os.path.exists(csv_filename):
            # Load existing data
            df_existing = pd.read_csv(csv_filename)

            # Extract only hyperparameter columns
            hyperparam_keys = list(raw_hyperparameters.keys())
            hyperparam_keys.append("Model Arc")
            hyperparam_keys.append("Model")

            df_new = pd.DataFrame([new_entry])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

            df_combined["conv_layers"] = df_combined["conv_layers"].astype(str)
            df_combined["fc_layers"] = df_combined["fc_layers"].astype(str)

            df_unique = df_combined.drop_duplicates(
                subset=hyperparam_keys, keep="first"
            )
            df_unique.to_csv(csv_filename, index=False)
            print("New experiment results appended to CSV.")
        else:
            # Create a new DataFrame and save it
            df_new = pd.DataFrame([new_entry])
            df_new.to_csv(csv_filename, index=False)
            print("New CSV file created and experiment results saved.")

    def save_epoch_data(self):
        date = datetime.now().strftime("%Y-%m-%d")
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        csv_filename = os.path.join(
            self.config.outputs.model.best_scores_path,
            f"{date}_{dataset_conf.target_col_alias}_{self.config.exp_name}_r2_scores.csv",
        )

        data = np.array(self.result_dict.get("r2_score"), dtype=float)
        indexed_data = np.column_stack((np.arange(1, len(data) + 1), data))
        np.savetxt(csv_filename, indexed_data, fmt="%.4f", delimiter=",", header="Index,Value", comments="")
        print("Epoch data CSV saved successfully!")

    def save_predictions(self, predictor=False):
        date = datetime.now().strftime("%Y-%m-%d")
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        default_name = f"{date}_{dataset_conf.target_col_alias}_{self.config.exp_name}_predictions_on_test_set.csv"

        if not predictor:
            csv_filename = os.path.join(
                self.config.outputs.model.best_scores_path,
                default_name,
            )
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        else:
            target_path = self.config.predictor.test_output_path
            if not target_path:
                csv_filename = os.path.join(
                    self.config.outputs.model.best_scores_path,
                    default_name,
                )
                os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
            else:
                root = target_path.rstrip("/\\")
                _, ext = os.path.splitext(root)
                if ext:
                    output_dir = os.path.dirname(root)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    csv_filename = root
                else:
                    output_dir = root
                    os.makedirs(output_dir, exist_ok=True)
                    csv_filename = os.path.join(
                        output_dir,
                        f"{self.config.exp_name}_predictions.csv",
                    )

        data = pd.DataFrame({"Targets": self.targets, "Prediction": self.predictions})
        indexed_data = np.column_stack((np.arange(1, len(data) + 1), data))
        np.savetxt(csv_filename, indexed_data, fmt="%d,%.4f,%.4f", delimiter=",", header="Index,Targets,Prediction", comments="")
        print("Prediction vs targets file saved successfully!")

    def min_max_denormalize(self, normalized_data: torch.Tensor, min_vals: torch.Tensor,
                            max_vals: torch.Tensor) -> torch.Tensor:
        """
        Reverts data from normalized [0, 1] range back to original values.

        Args:
            normalized_data (Tensor): Normalized data
            min_vals (Tensor): Minimum values used for normalization
            max_vals (Tensor): Maximum values used for normalization

        Returns:
            Tensor: Denormalized data
        """
        return normalized_data * (max_vals - min_vals + 1e-8) + min_vals

    def pressure_denorm(self, normalized_data: torch.Tensor):
        return normalized_data * 30**2

    def save_pressure_prediction(self, min_data, max_data, min_target, max_target, predictor=False):

        file_name = self.test_data_loader.dataset.dataset.data_frame['Design'][0]

        camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                           (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                           (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]

        data_tensor = torch.tensor(self.all_data, dtype=torch.float32)

        vertices_denorm = self.min_max_denormalize(data_tensor, min_data, max_data)
        vertices = vertices_denorm.cpu().numpy()[0]
        actual_point_cloud = pv.PolyData(vertices)
        pressure_point_cloud = pv.PolyData(vertices)

        actual_tensor = torch.tensor(self.targets, dtype=torch.float32)
        predict_tensor = torch.tensor(self.predictions, dtype=torch.float32)
        #actual_denorm = self.min_max_denormalize(actual_tensor, min_target.squeeze(dim=2), max_target.squeeze(dim=2))
        #predict_denorm = self.min_max_denormalize(predict_tensor, min_target.squeeze(dim=2), max_target.squeeze(dim=2))
        actual_denorm = self.pressure_denorm(actual_tensor)
        predict_denorm = self.pressure_denorm(predict_tensor)

        actual_point_cloud["pressure"] = actual_denorm[0]  # add scalar field
        pressure_point_cloud["pressure"] = predict_denorm[0]  # add scalar field

        num_points = vertices.shape[0]
        plotter = pv.Plotter(shape=(1, 3), title=f"Ground Truth vs Prediction vs Error: {file_name}", off_screen=True)
        #plotter = pv.Plotter(shape=(1, 3), title=f"Ground Truth vs Prediction vs Error: {file_name}")

        plotter.subplot(0, 0)
        plotter.add_text(f"Ground Truth Pressure\nPoints: {num_points}", position="upper_left", font_size=10, color="black")
        plotter.add_points(
            actual_point_cloud,
            scalars="pressure",
            cmap="jet",
            point_size=3,
            render_points_as_spheres=True,
        )
        plotter.camera_position = camera_position

        plotter.subplot(0, 1)
        plotter.add_text(f"Pressure Prediction\nPoints: {num_points}", position="upper_left", font_size=10, color="black")
        plotter.add_points(
            pressure_point_cloud,
            scalars="pressure",
            cmap="jet",
            point_size=3,
            render_points_as_spheres=True,
        )
        plotter.add_scalar_bar(title="Pressure")
        plotter.camera_position = camera_position

        # Plot Error
        error_tensor = (predict_denorm - actual_denorm).abs()
        error_point_cloud = pv.PolyData(vertices)
        error_point_cloud["error"] = error_tensor[0]  # assuming shape [1, N, 1]

        # Compute relative L2 error
        norm_rel_l2 = torch.norm(predict_tensor - actual_tensor) / torch.norm(actual_tensor)
        norm_rel_l2_value = norm_rel_l2.item()
        rel_l2 = torch.norm(predict_denorm - actual_denorm) / torch.norm(actual_denorm)
        rel_l2_value = rel_l2.item()

        plotter.subplot(0, 2)
        plotter.add_text(
            f"Pressure Absolute Error\nPoints: {num_points}\nNormalize Rel L2: {norm_rel_l2_value:.4f}\nActual Rel L2: {rel_l2_value:.4f}",
            position="upper_left",
            font_size=10,
            color="black"
        )
        plotter.add_points(
            error_point_cloud,
            scalars="error",
            cmap="jet",
            point_size=3,
            render_points_as_spheres=True,
        )
        plotter.camera_position = camera_position

        #plotter.show()
        plotter.screenshot(f"../outputs/plots/pressure_plot_{file_name}.png")
