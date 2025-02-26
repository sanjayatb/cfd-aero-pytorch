import yaml

from source.config.dto import Config, EnvironmentConfig, DataParams, ModelParams, OutputsConfig, DataConfig, Parameters, \
    ModelOutput


def load_config(file_path: str) -> Config:
    """Load YAML configuration into the Config dataclass."""
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(
        model_name=config_dict["model_name"],
        exp_name=config_dict.get("exp_name"),
        base_path=config_dict.get("base_path"),
        data=DataConfig(**config_dict["data"]),
        environment=EnvironmentConfig(**config_dict["environment"]),
        parameters=Parameters(data=DataParams(**config_dict["parameters"]["data"]),
                              model=ModelParams(**config_dict["parameters"]["model"])),
        outputs=OutputsConfig(
            model=ModelOutput(**config_dict.get("outputs", {}).get('model')),
            preprocessed_data=config_dict["outputs"]["preprocessed_data"],
            log_path=config_dict["outputs"]["log_path"]),
    )


def override_configs(config: Config, args):
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.data_name:
        config.data.name = args.data_name
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

    if not config.exp_name:
        config.exp_name = (f"exp_{config.data.name}"
                           f"_ts{config.parameters.data.training_size}"
                           f"_bs{config.parameters.model.batch_size}"
                           f"_epochs{config.parameters.model.epochs}"
                           f"_np{config.parameters.data.num_points}"
                           f"_lr{config.parameters.model.lr}"
                           f"_dropout{config.parameters.model.dropout}")

    print(f"🚀 Running Experiment: {config.exp_name}")
    print(f"🔹 Batch Size: {config.parameters.model.batch_size}, "
          f"Epochs: {config.parameters.model.epochs}, Num Points: {config.parameters.data.num_points}")
    print(f"🔹 Learning Rate: {config.parameters.model.lr}, Dropout: {config.parameters.model.dropout}")
