from source.data.ahmed_ml_dataset import AhmedMLDataset, AhmedMLGNNDataset
from source.data.dataset_loaders import DatasetLoaders, GeoDatasetLoaders
from source.data.enums import CFDDataset
from source.data.windsor_dataset import WindsorDataset
from source.model.gnn import DragGNN_XL
from source.model.pointnet import RegPointNet
from source.trainer.gnn_trainer import GNNTrainer
from source.trainer.pointnet_trainer import PointNetTrainer


class TrainerFactory:

    @staticmethod
    def get_pointnet_trainer(config, dataset_name: CFDDataset, model_name):

        if dataset_name == CFDDataset.AHMED_ML:
            dataset = AhmedMLDataset(root_dir=config.data.stl_path,
                                     csv_file=config.data.target_data_path,
                                     num_points=config.parameters.data.num_points,
                                     pointcloud_exist=False)

        elif dataset_name == CFDDataset.WINDSOR_ML:
            dataset = WindsorDataset()
        else:
            raise NotImplementedError()

        loader = DatasetLoaders(config=config, dataset=dataset)

        if model_name == "RegPointNet":
            trainer = PointNetTrainer(config, RegPointNet, loader)
        else:
            raise NotImplementedError()

        return trainer

    @staticmethod
    def get_gnn_trainer(config, dataset_name: CFDDataset, model_name):

        if dataset_name == CFDDataset.AHMED_ML:
            dataset = AhmedMLGNNDataset(root_dir=config.data.stl_path,
                                        csv_file=config.data.target_data_path)

        elif dataset_name == CFDDataset.WINDSOR_ML:
            dataset = WindsorDataset()
        else:
            raise NotImplementedError()

        loader = GeoDatasetLoaders(config=config, dataset=dataset)

        if model_name == "DragGNN_XL":
            trainer = GNNTrainer(config, DragGNN_XL, loader)
        else:
            raise NotImplementedError()

        return trainer
