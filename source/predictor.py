import argparse


class Predictor:

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Make prediction.")

        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Absolute path of model"
        )
        parser.add_argument("--stl", type=str, required=True)

    def predict(self):

        if self.config.model_arch == ModelArchitecture.POINT_NET.value:
            trainer = TrainerFactory.get_pointnet_trainer(self.config)
        elif self.config.model_arch == ModelArchitecture.GNN.value:
            trainer = TrainerFactory.get_gnn_trainer(self.config)
        elif self.config.model_arch == ModelArchitecture.FNO.value:
            raise NotImplemented("FNO not implemented")
        else:
            raise NotImplemented(f"{self.config.model_arch} not implemented.")









if __name__ == "__main__":
    runner = Predictor()
    runner.parse_args()
    runner.predict()
