import os
import time

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing_extensions import override

import torch.nn.functional as F

from source.trainer.model_trainer import ModelTrainer
from source.utils.model import r2_score


class GNNTrainer(ModelTrainer):

    def __init__(self, config, model, data_loaders):
        super().__init__(config, model, data_loaders)
        self._train_best_mse = None
        self._train_best_r2 = None

    @override
    def setup_data(self):
        self.data_loaders.init_loaders()

    @override
    def init_model(self) -> torch.nn.Module:
        this_model = self.model(config=self.config).to(self.device)
        # model = PointTransformer().to(device)
        # If CUDA is enabled and more than one GPU is available, wrap the model in a DataParallel module
        # to enable parallel computation across multiple GPUs. Specifically, use GPUs with IDs 0, 1, 2, and 3.
        if self.config.environment.cuda and torch.cuda.device_count() > 1:
            this_model = torch.nn.DataParallel(this_model, device_ids=self.config.environment.device_id)

        # Return the initialized model
        return this_model

    @override
    def train_and_evaluate(self, model: torch.nn.Module):
        train_losses, val_losses = [], []
        training_start_time = time.time()  # Start timing for training

        # Initialize the Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.parameters.model.lr, weight_decay=1e-4) \
            if self.config.parameters.model.optimizer == 'adam' else optim.SGD(
            model.parameters(), lr=self.config.parameters.model.lr, momentum=0.9, weight_decay=1e-4)

        # Initialize the learning rate scheduler (ReduceLROnPlateau) to reduce the learning rate based on validation loss
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.1, verbose=True)

        best_mse = float('inf')  # Initialize the best MSE as infinity

        # Training loop over the specified number of epochs
        epochs = self.config.parameters.model.epochs
        for epoch in range(epochs):
            epoch_start_time = time.time()  # Start timing for this epoch
            model.train()  # Set the model to training mode
            total_loss = 0

            # Iterate over the training data
            for data in tqdm(self.data_loaders.train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]"):
                data = data.to(self.device)  # Move data to the gpu

                optimizer.zero_grad()
                outputs = model(data)
                loss = F.mse_loss(outputs.squeeze(dim=1), data.y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()  # Accumulate the loss

            epoch_duration = time.time() - epoch_start_time
            # Calculate and print the average training loss for the epoch
            avg_loss = total_loss / len(self.data_loaders.train_dataloader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.6f} Time: {epoch_duration:.2f}s")

            # Validation phase
            model.eval()  # Set the model to evaluation mode
            val_loss, val_r2 = 0, 0
            inference_times = []
            all_preds = []
            all_targets = []

            # No gradient computation needed during validation
            with torch.no_grad():
                # Iterate over the validation data
                for data in tqdm(self.data_loaders.val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]"):
                    inference_start_time = time.time()
                    data = data.to(self.device)
                    outputs = model(data)
                    loss = F.mse_loss(outputs.squeeze(dim=1), data.y)
                    val_loss += loss.item()

                    # Collect predictions and targets for R² calculation
                    all_preds.append(outputs.squeeze(dim=1).cpu().numpy())
                    all_targets.append(data.y.cpu().numpy())

                    inference_duration = time.time() - inference_start_time
                    inference_times.append(inference_duration)

            # Calculate and print the average validation loss for the epoch
            avg_val_loss = val_loss / len(self.data_loaders.val_dataloader)
            val_losses.append(avg_val_loss)
            avg_inference_time = sum(inference_times) / len(inference_times)
            print(
                f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, Avg Inference Time: {avg_inference_time:.4f}s")

            # Concatenate predictions and targets
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)

            # Compute R² for the entire validation dataset
            val_r2 = r2_score(all_preds, all_targets)
            print(f"Validation R²: {val_r2:.4f}")

            # Check if this is the best model based on MSE
            if avg_val_loss < best_mse:
                best_mse = avg_val_loss
                self.result_collector.save_best_model(model.state_dict(), best_mse)
                self._train_best_mse = best_mse
                self._train_best_r2 = val_r2.item()

            # Step the scheduler based on the validation loss
            scheduler.step(avg_val_loss)

        training_duration = time.time() - training_start_time
        print(f"Total training time: {training_duration:.2f}s")
        # # Save the final model state to disk
        # model_path = os.path.join('models', f'{self.config.exp_name}_final_model.pth')
        # torch.save(model.state_dict(), model_path)
        # print(f"Model saved to {model_path}")

    @override
    def test(self, model: torch.nn.Module):
        model.eval()  # Set the model to evaluation mode
        total_mse, total_mae = 0, 0
        max_mae = 0
        total_inference_time = 0  # To track total inference time
        total_samples = 0  # To count the total number of samples processed
        all_preds = []
        all_targets = []
        # Disable gradient calculation
        with torch.no_grad():
            for data in self.data_loaders.test_dataloader:
                start_time = time.time()  # Start time for inference

                data = data.to(self.device)
                outputs = model(data)

                end_time = time.time()  # End time for inference
                inference_time = end_time - start_time
                total_inference_time += inference_time  # Accumulate total inference time

                mse = F.mse_loss(outputs.squeeze(dim=1), data.y)  # Mean Squared Error (MSE)
                mae = F.l1_loss(outputs.squeeze(dim=1), data.y)  # Mean Absolute Error (MAE)

                # Collect predictions and targets for R² calculation
                all_preds.append(outputs.squeeze(dim=1).cpu().numpy())
                all_targets.append(data.y.cpu().numpy())

                # Accumulate metrics to compute averages later
                total_mse += mse.item()
                total_mae += mae.item()
                max_mae = max(max_mae, mae.item())
                total_samples += data.y.size(0)  # Increment total sample count

        # Concatenate predictions and targets
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Compute R² for the entire test dataset
        test_r2 = r2_score(all_preds, all_targets)
        # Compute average metrics over the entire test set
        avg_mse = total_mse / len(self.data_loaders.test_dataloader)
        avg_mae = total_mae / len(self.data_loaders.test_dataloader)

        # Output test results
        print(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {test_r2:.4f}")
        print(f"Total inference time: {total_inference_time:.2f}s for {total_samples} samples")

        scores = {'Train Best MSE': self._train_best_mse, 'Best model R2': self._train_best_r2, 'Test MSE': avg_mse, 'Test MAE': avg_mae,
                  'Test Max MAE': max_mae, 'Test R2': test_r2.item()}
        self.result_collector.save_test_scores(scores)

    @override
    def load_and_test(self):
        """Load a saved model and test it."""
        this_model = self.model(config=self.config).to(self.device)
        if self.config.environment.cuda and torch.cuda.device_count() > 1:
            this_model = torch.nn.DataParallel(this_model, device_ids=self.config.environment.device_id)

        best_model_path = os.path.join(self.config.outputs.model.best_model_path,
                                       f'{self.config.exp_name}_best_model.pth')
        print(f"Load best model: {best_model_path}")
        this_model.load_state_dict(torch.load(best_model_path))

        self.test(this_model)
