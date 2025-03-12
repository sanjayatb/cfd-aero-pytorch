import os.path
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DatasetsConfig:
    stl_path: str
    target_data_path: str
    id_col: str
    target_col: str
    target_col_alias: str
    subset_dir: str


@dataclass
class EnvironmentConfig:
    cuda: bool
    device_id: List[int]
    seed: int
    num_workers: int


@dataclass
class DataParams:
    dataset: str
    training_size: int
    validation_size: int
    test_size: int
    num_points: Optional[int] = field(default=None)


@dataclass
class ModelParams:
    lr: float
    batch_size: int
    epochs: int
    dropout: float
    emb_dims: int
    k: int
    optimizer: str
    conv_layers: List[int]
    fc_layers: List[int]
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
    model_arch: str
    model_name: str
    experiment_batch_name: str
    exp_name: str
    base_path: str
    datasets: Dict[str, DatasetsConfig]
    environment: EnvironmentConfig
    parameters: Parameters
    outputs: OutputsConfig

    def __post_init__(self):
        for _, dataset in self.datasets.items():
            dataset.stl_path = os.path.join(self.base_path, dataset.stl_path)
            dataset.target_data_path = os.path.join(
                self.base_path, dataset.target_data_path
            )
            dataset.subset_dir = os.path.join(self.base_path, dataset.subset_dir)
        self.outputs.preprocessed_data = os.path.join(
            self.base_path, self.outputs.preprocessed_data
        )
        self.outputs.log_path = os.path.join(self.base_path, self.outputs.log_path)
        self.outputs.model.best_model_path = os.path.join(
            self.base_path, self.outputs.model.best_model_path
        )
        self.outputs.model.best_scores_path = os.path.join(
            self.base_path, self.outputs.model.best_scores_path
        )
