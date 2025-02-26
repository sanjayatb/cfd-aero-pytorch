import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.config.dto import Config
from source.data.enums import CFDDataset
from source.trainer.trainer_factory import TrainerFactory

from source.config.loader import load_config, override_configs
from source.utils.env import setup_seed


class Runner:
    config: Config = None

    def parse_args(self, model_arch):
        parser = argparse.ArgumentParser(description="Run Experiment")

        if model_arch == "PointNet":
            self.config = load_config("../configs/pointnet_config.yml")
        elif model_arch == "GNN":
            self.config = load_config("../configs/gnn_config.yml")

        parser.add_argument("--model-name", type=str, required=False, default="PointNet")
        parser.add_argument("--data-name", type=str, required=False, default="AhmedML")
        parser.add_argument("--train-size", type=int, required=False)
        parser.add_argument("--batch-size", type=int, required=False)
        parser.add_argument("--epochs", type=int, required=False)
        parser.add_argument("--num-points", type=int, required=False)
        parser.add_argument("--lr", type=float, required=False)
        parser.add_argument("--dropout", type=float, required=False)
        parser.add_argument("--exp-name", type=str, required=False, default=None)

        args = parser.parse_args()
        override_configs(self.config, args)

    def run(self, model_architecture):
        setup_seed(self.config.environment.seed)

        if model_architecture == "PointNet":
            trainer = TrainerFactory.get_pointnet_trainer(self.config, CFDDataset.AHMED_ML, "RegPointNet")
        elif model_architecture == "GNN":
            trainer = TrainerFactory.get_gnn_trainer(self.config, CFDDataset.AHMED_ML, "DragGNN_XL")
        elif model_architecture == "FNO":
            raise NotImplemented("FNO not implemented")
        else:
            raise NotImplemented(f"{model_architecture} not implemented.")

        trainer.setup_data()
        model = trainer.init_model()
        trainer.train_and_evaluate(model)
        trainer.load_and_test()


if __name__ == "__main__":
    runner = Runner()
    runner.parse_args("GNN")
    runner.run("GNN")
