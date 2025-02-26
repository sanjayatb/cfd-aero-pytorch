import os.path
from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    name: str
    stl_path: str
    target_data_path: str
    target_col: List[str]
    subset_dir: str


@dataclass
class EnvironmentConfig:
    cuda: bool
    device_id: List[int]
    seed: int
    num_workers: int


@dataclass
class DataParams:
    training_size: int
    validation_size: int
    test_size: int
    num_points: int


@dataclass
class ModelParams:
    lr: float
    batch_size: int
    epochs: int
    dropout: float
    emb_dims: int
    k: int
    optimizer: str
    channels: List[int]
    linear_sizes: List[int]
    output_channels: int


@dataclass
class Parameters:
    data: DataParams
    model: ModelParams


@dataclass
class ModelOutput:
    best_model_path: str
    best_scores_path: str


@dataclass
class OutputsConfig:
    log_path: str
    preprocessed_data: str
    model: ModelOutput


@dataclass
class Config:
    model_name: str
    exp_name: str
    base_path: str
    data: DataConfig
    environment: EnvironmentConfig
    parameters: Parameters
    outputs: OutputsConfig

    def __post_init__(self):
        self.data.stl_path = os.path.join(self.base_path, self.data.stl_path)
        self.data.target_data_path = os.path.join(self.base_path, self.data.target_data_path)
        self.data.subset_dir = os.path.join(self.base_path, self.data.subset_dir)
        self.outputs.preprocessed_data = os.path.join(self.base_path, self.outputs.preprocessed_data)
        self.outputs.log_path = os.path.join(self.base_path, self.outputs.log_path)
        self.outputs.model.best_model_path = os.path.join(self.base_path, self.outputs.model.best_model_path)
        self.outputs.model.best_scores_path = os.path.join(self.base_path, self.outputs.model.best_scores_path)
