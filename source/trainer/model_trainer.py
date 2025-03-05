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

    @abstractmethod
    def init_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def train_and_evaluate(self, model: torch.nn.Module):
        pass

    @abstractmethod
    def test(self, model: torch.nn.Module):
        pass

    @abstractmethod
    def load_and_test(self):
        pass
