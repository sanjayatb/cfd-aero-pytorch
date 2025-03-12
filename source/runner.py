import argparse
import importlib
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.config.dto import Config
from source.data.enums import CFDDataset, ModelArchitecture
from source.trainer.trainer_factory import TrainerFactory

from source.config.loader import load_config, override_configs
from source.utils.env import setup_seed
import inspect


class Runner:
    config: Config = None

    @staticmethod
    def available_models(model_arc):
        if model_arc == ModelArchitecture.POINT_NET.value:
            module_name = "source.model.pointnet"
        elif model_arc == ModelArchitecture.GNN.value:
            module_name = "source.model.gnn"
        else:
            raise NotImplemented(f"{model_arc} is not implemented")
        try:
            module = importlib.import_module(module_name)  # Import the module
            classes = {
                name: cls
                for name, cls in inspect.getmembers(module, inspect.isclass)
                if cls.__module__ == module_name
            }  # Filter only classes from this module
            return classes.keys()
        except ModuleNotFoundError:
            raise ImportError(f"Module '{module_name}' not found!")

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Run Experiment")

        parser.add_argument(
            "--model-arch",
            type=str,
            required=True,
            choices=[e.value for e in ModelArchitecture],
        )
        args, unknown = parser.parse_known_args()

        available_models = self.available_models(args.model_arch)

        parser.add_argument(
            "--model-name", type=str, required=True, choices=available_models
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            required=True,
            choices=[e.value for e in CFDDataset],
        )
        parser.add_argument("--experiment-batch-name", type=str, required=False)
        parser.add_argument("--train-size", type=int, required=False)
        parser.add_argument("--batch-size", type=int, required=False)
        parser.add_argument("--epochs", type=int, required=False)
        parser.add_argument("--num-points", type=int, required=False)
        parser.add_argument("--lr", type=float, required=False)
        parser.add_argument("--dropout", type=float, required=False)
        parser.add_argument("--exp-name", type=str, required=False, default=None)
        parser.add_argument("--conv-layers", type=str, required=False, default=None)
        parser.add_argument("--fc-layers", type=str, required=False, default=None)

        args = parser.parse_args()

        self.config = load_config("../configs/system_config.yml")

        override_configs(self.config, args)

    def run(self):
        setup_seed(self.config.environment.seed)

        if self.config.model_arch == ModelArchitecture.POINT_NET.value:
            trainer = TrainerFactory.get_pointnet_trainer(self.config)
        elif self.config.model_arch == ModelArchitecture.GNN.value:
            trainer = TrainerFactory.get_gnn_trainer(self.config)
        elif self.config.model_arch == ModelArchitecture.FNO.value:
            raise NotImplemented("FNO not implemented")
        else:
            raise NotImplemented(f"{self.config.model_arch} not implemented.")

        trainer.setup_data()
        model = trainer.init_model()
        trainer.train_and_evaluate(model)
        trainer.load_and_test()


if __name__ == "__main__":
    runner = Runner()
    runner.parse_args()
    runner.run()
