import argparse
import os

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.config.loader import load_config
from source.data.enums import ModelArchitecture
from source.trainer.trainer_factory import TrainerFactory
from source.utils.env import setup_seed


class Predictor:

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Make prediction.")
        parser.add_argument("--stl", type=str, required=False, default=None)
        return parser.parse_args()

    def predict(self, args):

        config = load_config("../configs/system_config.yml")
        setup_seed(config.environment.seed)

        if config.model_arch == ModelArchitecture.POINT_NET.value:
            trainer = TrainerFactory.get_pointnet_trainer(config)
        elif config.model_arch == ModelArchitecture.GNN.value:
            trainer = TrainerFactory.get_gnn_trainer(config)
        elif config.model_arch == ModelArchitecture.FNO.value:
            raise NotImplemented("FNO not implemented")
        else:
            raise NotImplemented(f"{config.model_arch} not implemented.")

        trainer.setup_data()
        loaded_model = trainer.load_model(config.predictor.best_model_path)
        if args.stl is None:
            trainer.test(loaded_model, True)
        else:
            trainer.predict_one(loaded_model, args.stl)


if __name__ == "__main__":
    runner = Predictor()
    args = runner.parse_args()
    runner.predict(args)
