from datetime import datetime

import yaml

from source.config.dto import (
    Config,
    EnvironmentConfig,
    DataParams,
    ModelParams,
    OutputsConfig,
    DatasetsConfig,
    Parameters,
    ModelOutput, Predictor,
)

def load_config(file_path: str) -> Config:
    """Load YAML configuration into the Config dataclass."""
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(
        model_arch=config_dict["model_arch"],
        model_name=config_dict["model_name"],
        experiment_batch_name=config_dict["experiment_batch_name"],
        exp_name=config_dict.get("exp_name"),
        base_path=config_dict.get("base_path"),
        datasets={
            k: DatasetsConfig(**v) for k, v in config_dict.get("datasets").items()
        },
        environment=EnvironmentConfig(**config_dict["environment"]),
        parameters=Parameters(
            data=DataParams(**config_dict["parameters"]["data"]),
            model=ModelParams(**config_dict["parameters"]["model"]),
        ),
        outputs=OutputsConfig(
            model=ModelOutput(**config_dict.get("outputs", {}).get("model")),
            preprocessed_data=config_dict["outputs"]["preprocessed_data"],
            log_path=config_dict["outputs"]["log_path"],
        ),
        predictor=Predictor(**config_dict.get("predictor", {}))
    )


def override_configs(config: Config, args):
    if args.model_arch:
        config.model_arch = args.model_arch
    if args.model_name:
        config.model_name = args.model_name
    if args.experiment_batch_name:
        config.experiment_batch_name = args.experiment_batch_name
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.dataset_name:
        config.parameters.data.dataset = args.dataset_name
    if args.sample_size:
        config.parameters.data.max_total_samples = args.sample_size
    if args.train_size:
        config.parameters.data.training_size = args.train_size
    if args.batch_size:
        config.parameters.model.batch_size = args.batch_size
    if args.epochs:
        config.parameters.model.epochs = args.epochs
    if args.num_points:
        config.parameters.data.num_points = args.num_points
    if args.lr:
        config.parameters.model.lr = args.lr
    if args.dropout:
        config.parameters.model.dropout = args.dropout
    if args.conv_layers:
        config.parameters.model.conv_layers = list(map(int, args.conv_layers.strip("[]").split(":")))
    if args.fc_layers:
        config.parameters.model.fc_layers = list(map(int, args.fc_layers.strip("[]").split(":")))

    if not config.exp_name:
        date = datetime.now().strftime("%Y-%m-%d")
        dataset_conf = config.datasets.get(config.parameters.data.dataset)
        config.exp_name = (
            f"{date}_exp_{dataset_conf.target_col_alias}_{config.parameters.data.dataset}_{config.model_arch}_{config.model_name}"
            f"_ts{config.parameters.data.training_size}"
            f"_bs{config.parameters.model.batch_size}"
            f"_epochs{config.parameters.model.epochs}"
            f"_np{config.parameters.data.num_points}"
            f"_lr{config.parameters.model.lr}"
            f"_dropout{config.parameters.model.dropout}"
            f"_cl{config.parameters.model.conv_layers}"
            f"_fc{config.parameters.model.fc_layers}"
        )

    print(f"🚀 Running Experiment: {config.exp_name}")
    print(f"Experiment Batch Name: {config.experiment_batch_name}")
    print(
        f"🔹 Batch Size: {config.parameters.model.batch_size}, "
        f"Epochs: {config.parameters.model.epochs}, Num Points: {config.parameters.data.num_points}"
    )
    print(
        f"🔹 Learning Rate: {config.parameters.model.lr}, Dropout: {config.parameters.model.dropout}"
    )
    dataset_conf = config.datasets.get(config.parameters.data.dataset)
    print(
        f"🔹 Id Column: '{dataset_conf.id_col}', Target Column: '{dataset_conf.target_col}'"
    )
