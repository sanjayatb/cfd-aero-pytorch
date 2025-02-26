import os
from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch_geometric.data import Data, Dataset as GeoDataset, DataLoader as GeoDataLoader

from source.config.dto import Config


class DatasetLoaders:
    config: Config
    dataset: Any
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def init_loaders(self):
        def create_subset(dataset: Dataset, ids_file) -> Dataset:
            try:
                with open(os.path.join(self.config.data.subset_dir, ids_file), 'r') as file:
                    subset_ids_str = file.read().split()
                subset_ids = list(map(int, subset_ids_str))
                subset_indices = dataset.data_frame[dataset.data_frame['run'].isin(subset_ids)].index.tolist()
                # print(subset_indices)
                return Subset(dataset, subset_indices)
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

        full_dataset = self.dataset
        # Create training subset using the corresponding subset file
        train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
        train_size = self.config.parameters.data.training_size
        # Reduce the size of the training dataset if train_frac is less than 1.0
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

        val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
        test_dataset = create_subset(full_dataset, 'test_design_ids.txt')
        # Initialize DataLoaders for each subset

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.parameters.model.batch_size,
                                           shuffle=True, drop_last=True,
                                           num_workers=self.config.environment.num_workers)

        self.val_dataloader = DataLoader(val_dataset, batch_size=self.config.parameters.model.batch_size,
                                         shuffle=False, drop_last=True,
                                         num_workers=self.config.environment.num_workers)

        self.test_dataloader = DataLoader(test_dataset, batch_size=self.config.parameters.model.batch_size,
                                          shuffle=False,
                                          drop_last=True, num_workers=self.config.environment.num_workers)


class GeoDatasetLoaders:
    config: Config
    dataset: GeoDataset
    train_dataloader: GeoDataLoader
    val_dataloader: GeoDataLoader
    test_dataloader: GeoDataLoader

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def init_loaders(self):
        def create_subset(dataset: GeoDataset, ids_file) -> Dataset:
            try:
                with open(os.path.join(self.config.data.subset_dir, ids_file), 'r') as file:
                    subset_ids_str = file.read().split()
                subset_ids = list(map(int, subset_ids_str))
                subset_indices = dataset.data_frame[dataset.data_frame['run'].isin(subset_ids)].index.tolist()
                # print(subset_indices)
                return Subset(dataset, subset_indices)
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

        full_dataset = self.dataset
        # Create training subset using the corresponding subset file
        train_dataset = create_subset(full_dataset, 'train_design_ids.txt')
        train_size = self.config.parameters.data.training_size
        # Reduce the size of the training dataset if train_frac is less than 1.0
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

        val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
        test_dataset = create_subset(full_dataset, 'test_design_ids.txt')
        # Initialize DataLoaders for each subset

        self.train_dataloader = GeoDataLoader(train_dataset, batch_size=self.config.parameters.model.batch_size,
                                           shuffle=True, drop_last=True,
                                           num_workers=self.config.environment.num_workers)

        self.val_dataloader = GeoDataLoader(val_dataset, batch_size=self.config.parameters.model.batch_size,
                                         shuffle=False, drop_last=True,
                                         num_workers=self.config.environment.num_workers)

        self.test_dataloader = GeoDataLoader(test_dataset, batch_size=self.config.parameters.model.batch_size,
                                          shuffle=False,
                                          drop_last=True, num_workers=self.config.environment.num_workers)
