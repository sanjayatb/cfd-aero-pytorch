import os.path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


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
    data_id_load_random: bool
    max_total_samples: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    k_folds: int
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
    point_transformer: Dict[str, Any] = field(default_factory=dict)


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
class Predictor:
    enable: bool = False
    best_model_path: Optional[str] = None
    test_file_path: Optional[str] = None
    test_output_path: Optional[str] = None
    test_stl_path: Optional[str] = None
    test_target_path: Optional[str] = None

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
    predictor: Predictor

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
        if self.predictor.best_model_path and not os.path.isabs(
            self.predictor.best_model_path
        ):
            self.predictor.best_model_path = os.path.join(
                self.base_path, self.predictor.best_model_path
            )
        if self.predictor.test_file_path and not os.path.isabs(
            self.predictor.test_file_path
        ):
            self.predictor.test_file_path = os.path.join(
                self.base_path, self.predictor.test_file_path
            )
        if self.predictor.test_output_path and not os.path.isabs(
            self.predictor.test_output_path
        ):
            self.predictor.test_output_path = os.path.join(
                self.base_path, self.predictor.test_output_path
            )
        if self.predictor.test_stl_path and not os.path.isabs(
            self.predictor.test_stl_path
        ):
            self.predictor.test_stl_path = os.path.join(
                self.base_path, self.predictor.test_stl_path
            )
        if self.predictor.test_target_path and not os.path.isabs(
            self.predictor.test_target_path
        ):
            self.predictor.test_target_path = os.path.join(
                self.base_path, self.predictor.test_target_path
            )
