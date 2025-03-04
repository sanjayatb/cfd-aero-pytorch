import importlib

from source.data.ahmed_ml_dataset import AhmedMLDataset, AhmedMLGNNDataset
from source.data.dataset_loaders import DatasetLoaders, GeoDatasetLoaders
from source.data.drivaer_ml_dataset import DrivAerNetDataset, DrivAerNetGNNDataset
from source.data.enums import CFDDataset
from source.data.windsor_ml_dataset import WindsorMLDataset, WindsorMLGNNDataset
from source.trainer.gnn_trainer import GNNTrainer
from source.trainer.pointnet_trainer import PointNetTrainer


class TrainerFactory:

    @staticmethod
    def get_model_class(module_name, class_name):
        """
        Dynamically import a model class from a module.

        :param module_name: Module where the class is located (e.g., "source.model.pointnet")
        :param class_name: Name of the class (e.g., "RegPointNet")
        :return: Class object
        """
        try:
            module = importlib.import_module(module_name)  # Import the module
            model_class = getattr(module, class_name)  # Get the class from the module
            return model_class  # Return the class itself
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Error loading model '{class_name}' from '{module_name}': {e}")

    @staticmethod
    def get_pointnet_trainer(config):

        dataset_conf = config.datasets.get(config.parameters.data.dataset)
        if config.parameters.data.dataset == CFDDataset.AHMED_ML.value:
            dataset = AhmedMLDataset(config=config,
                                     root_dir=dataset_conf.stl_path,
                                     csv_file=dataset_conf.target_data_path,
                                     num_points=config.parameters.data.num_points,
                                     pointcloud_exist=False)

        elif config.parameters.data.dataset == CFDDataset.WINDSOR_ML.value:
            dataset = WindsorMLDataset(config=config,
                                       root_dir=dataset_conf.stl_path,
                                       csv_file=dataset_conf.target_data_path,
                                       num_points=config.parameters.data.num_points,
                                       pointcloud_exist=False)
        elif config.parameters.data.dataset == CFDDataset.DRIVAER_ML.value:
            dataset = DrivAerNetDataset(config=config,
                                        root_dir=dataset_conf.stl_path,
                                        csv_file=dataset_conf.target_data_path,
                                        num_points=config.parameters.data.num_points,
                                        pointcloud_exist=False)
        else:
            raise NotImplementedError()

        loader = DatasetLoaders(config=config, dataset=dataset)

        trainer = PointNetTrainer(config, TrainerFactory.get_model_class("source.model.pointnet", config.model_name),
                                  loader)

        return trainer

    @staticmethod
    def get_gnn_trainer(config):

        dataset_conf = config.datasets.get(config.parameters.data.dataset)
        if config.parameters.data.dataset == CFDDataset.AHMED_ML.value:
            dataset = AhmedMLGNNDataset(config=config,
                                        root_dir=dataset_conf.stl_path,
                                        csv_file=dataset_conf.target_data_path)
        elif config.parameters.data.dataset == CFDDataset.WINDSOR_ML.value:
            dataset = WindsorMLGNNDataset(config=config,
                                          root_dir=dataset_conf.stl_path,
                                          csv_file=dataset_conf.target_data_path)
        elif config.parameters.data.dataset == CFDDataset.DRIVAER_ML.value:
            dataset = DrivAerNetGNNDataset(config=config,
                                           root_dir=dataset_conf.stl_path,
                                           csv_file=dataset_conf.target_data_path)
        else:
            raise NotImplementedError(f"{config.parameters.data.dataset} is not handled")
        loader = GeoDatasetLoaders(config=config, dataset=dataset)

        trainer = GNNTrainer(config, TrainerFactory.get_model_class("source.model.gnn", config.model_name), loader)

        return trainer
