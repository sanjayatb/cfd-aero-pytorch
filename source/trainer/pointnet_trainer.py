import os
import time

import numpy as np
import torch
import trimesh
from torch import optim
from tqdm import tqdm
from typing_extensions import override
from source.trainer.model_trainer import ModelTrainer
import torch.nn.functional as F

from source.utils.model import compute_r2_score, compute_rel_l2_score


class PointNetTrainer(ModelTrainer):
    def __init__(self, config, model, data_loaders):
        super().__init__(config, model, data_loaders)
        self._train_best_mse = None
        self._train_best_r2 = 0
        self._train_best_rel_l2 = 0
        self._train_size = 0
        self._val_size = 0
        self._test_size = 0
        self._training_time = 0

    @override
    def setup_data(self):
        super().setup_data()

    @override
    def init_model(self):
        this_model = self.model(config=self.config).to(self.device)
        # model = PointTransformer().to(device)
        # If CUDA is enabled and more than one GPU is available, wrap the model in a DataParallel module
        # to enable parallel computation across multiple GPUs. Specifically, use GPUs with IDs 0, 1, 2, and 3.
        if self.config.environment.cuda and torch.cuda.device_count() > 1:
            this_model = torch.nn.DataParallel(
                this_model, device_ids=self.config.environment.device_id
            )

        # Return the initialized model
        return this_model

    @override
    def train_and_evaluate(self, model: torch.nn.Module):
        train_losses, val_losses = [], []
        self._train_size = len(self.data_loaders.train_dataloaders[0].dataset)
        self._val_size = len(self.data_loaders.val_dataloaders[0].dataset)
        # Record the start time of training for performance analysis
        training_start_time = time.time()
        best_epoch = None
        # Initialize the optimizer based on configuration; default to Adam if 'adam' is specified, else use SGD
        if self.config.parameters.model.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.parameters.model.lr,
                weight_decay=1e-4,
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.parameters.model.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )

        # Initialize a learning rate scheduler to adjust the learning rate based on validation loss performance
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50, factor=0.1)

        # Best model tracking variables
        best_mse = float("inf")
        best_rel_l2 = float("inf")
        best_r2 = float("-inf")  # Initialize with the worst possible R²

        # Training loop
        epochs = self.config.parameters.model.epochs
        for epoch in range(epochs):
            # Timing each epoch for performance analysis
            epoch_start_time = time.time()
            model.train()  # Set model to training mode
            total_loss, total_r2 = 0, 0

            # Iterate over batches of data
            for data, targets, _, _ in tqdm(
                    self.data_loaders.train_dataloaders[0],
                    desc=f"Epoch {epoch + 1}/{epochs} [Training]",
            ):
                data, targets = data.to(self.device), targets.to(self.device).squeeze()
                data = data.permute(0, 2, 1)  # Adjust data dimensions if necessary

                # Forward pass
                optimizer.zero_grad()
                outputs = model(data)
                loss = F.mse_loss(outputs.squeeze(), targets)

                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()

                # Aggregate statistics
                total_loss += loss.item()

            # Calculate average loss and R² for the epoch
            avg_loss = total_loss / len(self.data_loaders.train_dataloaders[0])
            train_losses.append(avg_loss)

            # Epoch summary
            epoch_duration = time.time() - epoch_start_time
            print(
                f"Epoch {epoch + 1} Training Loss: {avg_loss:.6f} Time: {epoch_duration:.2f}s"
            )

            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_loss, val_r2 = 0, 0
            inference_times = []
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, targets, _, _ in tqdm(
                        self.data_loaders.val_dataloaders[0],
                        desc=f"Epoch {epoch + 1}/{epochs} [Validation]",
                ):
                    inference_start_time = time.time()
                    data, targets = (
                        data.to(self.device),
                        targets.to(self.device).squeeze(),
                    )
                    data = data.permute(0, 2, 1)

                    outputs = model(data)
                    # loss = F.mse_loss(outputs.squeeze(), targets)
                    loss = F.mse_loss(outputs.view(-1), targets.view(-1))
                    val_loss += loss.item()
                    all_preds.append(outputs.squeeze().cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                    inference_duration = time.time() - inference_start_time
                    inference_times.append(inference_duration)

            # Concatenate all predictions and targets
            dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
            all_preds = np.concatenate(all_preds) if dataset_conf.target_col_alias == "Drag" else np.concatenate(
                all_preds, axis=0).flatten()
            all_targets = np.concatenate(all_targets) if dataset_conf.target_col_alias == "Drag" else np.concatenate(
                all_targets, axis=0).flatten()

            # Compute R² for the entire validation dataset
            val_r2 = compute_r2_score(all_targets, all_preds)
            val_rel_l2 = compute_rel_l2_score(all_targets, all_preds)
            self.result_collector.add_r2_scores(val_r2)
            self.result_collector.add_rel_l2_scores(val_rel_l2)

            avg_val_loss = val_loss / len(self.data_loaders.val_dataloaders[0])
            val_losses.append(avg_val_loss)
            avg_inference_time = sum(inference_times) / len(inference_times)

            # Validation summary
            print(
                f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.8f}, Avg Inference Time: {avg_inference_time:.4f}s"
            )
            print(f"Epoch {epoch + 1} Validation R²: {val_r2:.4f}, Validation RelL2: {val_rel_l2:.4f}")

            # Update the best model if the current model outperforms previous models
            if dataset_conf.target_col_alias == "Drag":
                compare_exp = (val_r2 > best_r2) or (val_r2 == best_r2 and avg_val_loss < best_mse)
            else:
                compare_exp = (val_rel_l2 < best_rel_l2) or (val_rel_l2 == best_rel_l2 and avg_val_loss < best_mse)

            if compare_exp:
                best_epoch = epoch + 1
                best_mse = avg_val_loss
                best_r2 = val_r2
                best_rel_l2 = val_rel_l2
                self._train_best_r2 = val_r2
                self._train_best_rel_l2 = val_rel_l2
                self._train_best_mse = best_mse
                self.result_collector.save_best_model(model.state_dict(), best_mse)

            scheduler.step(avg_val_loss)  # Update the learning rate based on the validation loss

        self._training_time = time.time() - training_start_time
        print(
            f"\nTraining Complete! Best Validation MSE: {best_mse:.8f}, Best R²: {best_r2:.4f}, Best RelL2: {best_rel_l2:.4f} found at epoch {best_epoch}")
        print(f"Total Training Time: {self._training_time:.2f} seconds")
        # self.result_collector.save_epoch_data()

    @override
    def train_and_evaluate_kflod(self, model: torch.nn.Module):
        """Train and evaluate using K-Fold Cross-Validation while optimizing R² and minimizing MSE."""

        best_model_state = None
        best_mse = float("inf")
        best_r2 = float("-inf")  # Initialize with the worst possible R²

        # Record the start time of training for performance analysis
        training_start_time = time.time()

        for fold, (train_loader, val_loader) in enumerate(
                zip(self.data_loaders.train_dataloaders, self.data_loaders.val_dataloaders)):
            print(f"Fold {fold + 1}/{self.config.parameters.data.k_folds}")

            # Initialize a fresh model for each fold
            model = self.init_model()
            self._train_size = len(train_loader.dataset)
            self._val_size = len(val_loader.dataset)

            # Choose optimizer
            # Initialize the optimizer based on configuration; default to Adam if 'adam' is specified, else use SGD
            if self.config.parameters.model.optimizer == "adam":
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.config.parameters.model.lr,
                    weight_decay=1e-4,
                )
            else:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=self.config.parameters.model.lr,
                    momentum=0.9,
                    weight_decay=1e-4,
                )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50, factor=0.1)

            for epoch in range(self.config.parameters.model.epochs):
                # Timing each epoch for performance analysis
                epoch_start_time = time.time()
                model.train()
                total_loss = 0

                for data, targets, _, _ in tqdm(train_loader,
                                          desc=f"Fold {fold + 1}/{self.config.parameters.data.k_folds} -> Epoch {epoch + 1}/{self.config.parameters.model.epochs} [Training]"):
                    data, targets = data.to(self.device), targets.to(self.device).squeeze()
                    data = data.permute(0, 2, 1)  # Adjust data dimensions if necessary

                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = F.mse_loss(outputs.squeeze(), targets)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                epoch_duration = time.time() - epoch_start_time
                print(
                    f"Fold {fold + 1}/{self.config.parameters.data.k_folds} -> Epoch {epoch + 1} Training Loss: {avg_train_loss:.6f},  Time: {epoch_duration:.2f}s")

                # Validation Phase
                model.eval()
                val_loss = 0
                all_preds, all_targets = [], []
                inference_times = []
                with torch.no_grad():
                    for data, targets in tqdm(val_loader,
                                              desc=f"Fold {fold + 1}/{self.config.parameters.data.k_folds} -> Epoch {epoch + 1}/{self.config.parameters.model.epochs} [Validation]"):
                        inference_start_time = time.time()
                        data, targets = data.to(self.device), targets.to(self.device).squeeze()
                        data = data.permute(0, 2, 1)

                        outputs = model(data)
                        loss = F.mse_loss(outputs.squeeze(), targets)
                        val_loss += loss.item()
                        all_preds.append(outputs.squeeze().cpu().numpy())
                        all_targets.append(targets.cpu().numpy())

                        inference_duration = time.time() - inference_start_time
                        inference_times.append(inference_duration)

                avg_val_loss = val_loss / len(val_loader)
                dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
                all_preds = np.concatenate(all_preds) if dataset_conf.target_col_alias == "Drag" else np.concatenate(
                    all_preds, axis=0).flatten()
                all_targets = np.concatenate(
                    all_targets) if dataset_conf.target_col_alias == "Drag" else np.concatenate(all_targets,
                                                                                                axis=0).flatten()
                val_r2 = compute_r2_score(all_targets, all_preds)

                print(
                    f"Fold {fold + 1}/{self.config.parameters.data.k_folds} -> Epoch {epoch + 1} Validation Loss: {avg_val_loss:.6f}, Validation R²: {val_r2:.4f}")

                # **Improved Model Selection Criteria**
                if (val_r2 > best_r2) or (val_r2 == best_r2 and avg_val_loss < best_mse):
                    best_mse = avg_val_loss
                    best_r2 = val_r2
                    best_model_state = model.state_dict()
                    self._train_best_r2 = val_r2
                    self._train_best_mse = best_mse

                scheduler.step(avg_val_loss)

        # Save the best model across all folds
        if best_model_state is not None:
            self.result_collector.save_best_model(best_model_state, best_mse)
            print("Best model saved!")

        self._training_time = time.time() - training_start_time
        print(f"K-Fold Cross-Validation Complete! Best MSE: {best_mse:.6f}, Best R²: {best_r2:.4f}")

    @override
    def test(self, model: torch.nn.Module, predictor=False):
        model.eval()  # Set the model to evaluation mode
        total_mse, total_mae, total_r2 = 0, 0, 0
        max_mae = 0
        total_inference_time = 0  # To track total inference time
        total_samples = 0  # To count the total number of samples processed
        all_data = []
        all_preds = []
        all_targets = []
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)

        self._test_size = len(self.data_loaders.test_dataloader.dataset)
        # Disable gradient calculation
        with torch.no_grad():
            for data, targets, (min_data, max_data), (min_target, max_target) in self.data_loaders.test_dataloader:
                start_time = time.time()  # Start time for inference

                _data, targets = data.to(self.device), targets.to(self.device).squeeze()
                all_data.append(_data.cpu().numpy())
                data = _data.permute(0, 2, 1)
                outputs = model(data)

                end_time = time.time()  # End time for inference
                inference_time = end_time - start_time
                total_inference_time += (
                    inference_time  # Accumulate total inference time
                )

                mse = F.mse_loss(
                    outputs.squeeze(), targets
                )  # Mean Squared Error (MSE)
                mae = F.l1_loss(
                    outputs.squeeze(), targets
                )  # Mean Absolute Error (MAE),

                # Collect predictions and targets for R² calculation
                all_preds.append(outputs.squeeze().cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                # Accumulate metrics to compute averages later
                total_mse += mse.item()
                total_mae += mae.item()
                max_mae = max(max_mae, mae.item())
                total_samples += targets.size(0)  # Increment total sample count

        # Compute average metrics over the entire test set
        avg_mse = total_mse / len(self.data_loaders.test_dataloader)
        avg_mae = total_mae / len(self.data_loaders.test_dataloader)
        # Concatenate all predictions and targets
        all_data = np.concatenate(all_data)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        self.result_collector.test_data_loader = self.data_loaders.test_dataloader
        self.result_collector.all_data = all_data
        self.result_collector.targets = all_targets
        self.result_collector.predictions = all_preds

        test_r2 = compute_r2_score(all_targets.flatten(), all_preds.flatten())
        test_rel_l2 = compute_rel_l2_score(all_targets.flatten(), all_preds.flatten())

        if dataset_conf.target_col_alias == "Drag":
            self.result_collector.save_predictions(predictor)
        #else:
        #    self.result_collector.save_pressure_prediction(min_data, max_data, min_target, max_target, predictor)
        # Output test results
        print(
            f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {test_r2:.4f}, Test RelL2: {test_rel_l2:.4f}"
        )
        print(
            f"Total inference time: {total_inference_time:.2f}s for {total_samples} samples"
        )
        if not predictor:
            scores = {
                "Train size": self._train_size,
                "Validation size": self._val_size,
                "Test size": self._test_size,
                "Best model R2": self._train_best_r2,
                "Test R2": test_r2,
                "Train Best MSE": self._train_best_mse,
                "Test MSE": avg_mse,
                "Best model RelL2": self._train_best_rel_l2,
                "Test RelL2": test_rel_l2,
                "Test MAE": avg_mae,
                "Test Max MAE": max_mae,
                "Train Time(s)": f"{self._training_time:.2f}",
                "Total Inference Time(s)": f"{total_inference_time:.2f}",
            }
            self.result_collector.save_test_scores(scores)

    def remove_module_prefix(self, state_dict):
        """Remove 'module.' prefix from keys (for models trained using DataParallel)."""
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k  # strip 'module.'
            new_state_dict[name] = v
        return new_state_dict

    @override
    def load_model(self, file_path=None):
        """Load a saved model and test it, returning the test results."""
        this_model = self.model(config=self.config).to(self.device)
        if self.config.environment.cuda and torch.cuda.device_count() > 1:
            this_model = torch.nn.DataParallel(
                this_model, device_ids=self.config.environment.device_id
            )

        if file_path is None:
            best_model_path = os.path.join(
                self.config.outputs.model.best_model_path,
                f"{self.config.exp_name}_best_model.pth",
            )
        else:
            best_model_path = f"{self.config.base_path}/{file_path}"
        print(f"Load best model: {best_model_path}")

        if not os.path.exists(best_model_path):
            print("No valuable model.")
        else:
            if self.config.environment.cuda:
                state_dict = torch.load(best_model_path, map_location=self.device)
            else:
                state_dict = self.remove_module_prefix(torch.load(best_model_path, map_location=self.device))

            this_model.load_state_dict(state_dict)
            return this_model

    def load_and_preprocess_stl(
            self,
            file_path: str,
            num_points: int = 250000,
            device: str = "cpu"
    ) -> torch.Tensor:
        """
        Load an STL file and convert it into a normalized point cloud tensor for PointNet.

        Args:
            file_path (str): Path to the STL file.
            num_points (int): Number of points to sample from the surface.
            device (str): 'cpu' or 'cuda' — where to load the tensor.

        Returns:
            torch.Tensor: Tensor of shape (1, 3, N), ready for PointNet.
        """
        mesh = trimesh.load_mesh(file_path)

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Invalid STL file or unsupported mesh type: {file_path}")

        # Sample points uniformly on the surface of the mesh
        points, _ = trimesh.sample.sample_surface(mesh, num_points)

        # Normalize the point cloud to zero-mean and unit sphere
        points = points - points.mean(axis=0)  # center
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale  # scale to unit ball

        # Convert to torch tensor: shape (1, 3, num_points)
        point_tensor = torch.tensor(points, dtype=torch.float32).T.unsqueeze(0).to(device)

        return point_tensor

    def predict_one(self, model: torch.nn.Module, file_path: str):
        """
        Predict for a single sample or file input (already preprocessed to tensor).
        """
        data_tensor = self.load_and_preprocess_stl(file_path)  # → should return torch.Tensor
        model.eval()

        with torch.no_grad():
            data = data_tensor.to(self.device)

            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time

            prediction = output.squeeze().cpu().numpy()

            print(f"Prediction: {prediction}")
            print(f"Inference time: {inference_time:.4f}s")

            return prediction
