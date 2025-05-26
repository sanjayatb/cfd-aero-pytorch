from abc import abstractmethod

import torch

from source.config.dto import Config
from source.data.dataset_loaders import DatasetLoaders
from source.model.base import Model
from source.results.result_collector import ResultCollector


class ModelTrainer:
    device: torch.device
    config: Config
    dataset_loaders: DatasetLoaders
    model: Model
    result_collector: ResultCollector

    def __init__(self, config, model, data_loaders):
        self.config = config
        self.model = model
        self.data_loaders = data_loaders
        self.result_collector = ResultCollector(self.config)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.environment.cuda else "cpu"
        )
        print(self.device)

    @abstractmethod
    def setup_data(self):
        self.data_loaders.init_loaders()
        train_len = len(self.data_loaders.train_dataloaders[0].dataset)
        val_len = len(self.data_loaders.val_dataloaders[0].dataset)
        test_len = len(self.data_loaders.test_dataloader.dataset)
        print(f"Train size: {train_len}, "
              f"Validation size: {val_len}, "
              f"Test size: {test_len}, "
              f"Total Dataset size: {train_len + val_len + test_len}.")

    @abstractmethod
    def init_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def train_and_evaluate(self, model: torch.nn.Module):
        pass

    @abstractmethod
    def train_and_evaluate_kflod(self, model: torch.nn.Module):
        pass

    @abstractmethod
    def test(self, model: torch.nn.Module, predictor=False):
        pass

    @abstractmethod
    def load_model(self, file_path=None):
        pass
