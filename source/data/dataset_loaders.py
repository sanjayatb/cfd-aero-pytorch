import os
import random
from typing import Any, List

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch_geometric.data import (
    Dataset as GeoDataset,
    DataLoader as GeoDataLoader,
)

from source.config.dto import Config
from source.data.enums import CFDDataset
from sklearn.model_selection import train_test_split
import csv


class DatasetLoaders:
    config: Config
    dataset: Any
    train_dataloaders: List[DataLoader] = []
    val_dataloaders: List[DataLoader] = []
    test_dataloader: DataLoader

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def write_to_file(self, file_name, values):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            for name in values:
                writer.writerow([name])

    def init_loaders(self):

        def get_id_list(ids_file):
            try:
                dataset_conf = self.config.datasets.get(
                    self.config.parameters.data.dataset
                )
                with open(
                        os.path.join(dataset_conf.subset_dir, ids_file), "r"
                ) as file:
                    return file.read().split()
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

        def create_subset(dataset: Dataset, ids_list) -> Dataset:
            # Filter the dataset DataFrame based on subset IDs
            dataset_conf = self.config.datasets.get(
                self.config.parameters.data.dataset
            )
            subset_indices = dataset.data_frame[
                dataset.data_frame[dataset_conf.id_col].isin(ids_list)
            ].index.tolist()
            return Subset(dataset, subset_indices)

        full_dataset = self.dataset
        assert (self.config.parameters.data.train_ratio +
                self.config.parameters.data.val_ratio
                + self.config.parameters.data.test_ratio == 1), "Ratios must sum to 1"
        # Create training subset using the corresponding subset file or random k folds
        if self.config.parameters.data.data_id_load_random:
            if self.config.parameters.data.k_folds == 1:
                file_names = self.dataset.get_all_file_names()
                # Shuffle the list randomly
                random.seed(self.config.environment.seed)
                random.shuffle(file_names)
                n_total = len(file_names)
                n_train = int(self.config.parameters.data.train_ratio * n_total)
                n_val = int(self.config.parameters.data.val_ratio * n_total)
                train_data_ids = file_names[:n_train]
                val_data_ids = file_names[n_train:n_train + n_val]
                test_data_ids = file_names[n_train + n_val:]
                train_dataset = create_subset(full_dataset, train_data_ids)
                val_dataset = create_subset(full_dataset, val_data_ids)

                # Initialize DataLoaders for each subset
                self.train_dataloaders.append(DataLoader(
                    train_dataset,
                    batch_size=self.config.parameters.model.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.config.environment.num_workers,
                ))

                self.val_dataloaders.append(DataLoader(
                    val_dataset,
                    batch_size=self.config.parameters.model.batch_size,
                    shuffle=False,
                    drop_last=True,
                    num_workers=self.config.environment.num_workers,
                ))
                os.makedirs(f"../outputs/subset_dynamic/{self.config.experiment_batch_name}/", exist_ok=True)
                self.write_to_file(
                    f"../outputs/subset_dynamic/{self.config.experiment_batch_name}/train_design_ids.txt",
                    train_data_ids)
                self.write_to_file(f"../outputs/subset_dynamic/{self.config.experiment_batch_name}/val_design_ids.txt",
                                   val_data_ids)
                self.write_to_file(f"../outputs/subset_dynamic/{self.config.experiment_batch_name}/test_design_ids.txt",
                                   test_data_ids)

            else:
                train_val_data_ids, test_data_ids = train_test_split(self.dataset.get_all_file_names(),
                                                                     test_size=self.config.parameters.data.test_ratio,
                                                                     random_state=42)
                kf = KFold(n_splits=self.config.parameters.data.k_folds, shuffle=True, random_state=42)
                for train_ids_list, val_ids_list in kf.split(train_val_data_ids):
                    train_data_ids = [train_val_data_ids[idx] for idx in train_ids_list]
                    val_data_ids = [train_val_data_ids[idx] for idx in val_ids_list]

                    train_dataset = create_subset(full_dataset, train_data_ids)
                    val_dataset = create_subset(full_dataset, val_data_ids)

                    # Initialize DataLoaders for each subset
                    self.train_dataloaders.append(DataLoader(
                        train_dataset,
                        batch_size=self.config.parameters.model.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.config.environment.num_workers,
                    ))

                    self.val_dataloaders.append(DataLoader(
                        val_dataset,
                        batch_size=self.config.parameters.model.batch_size,
                        shuffle=False,
                        drop_last=True,
                        num_workers=self.config.environment.num_workers,
                    ))
        else:
            train_data_ids = get_id_list("train_design_ids.txt")
            val_data_ids = get_id_list("val_design_ids.txt")
            test_data_ids = get_id_list("test_design_ids.txt")

            train_dataset = create_subset(full_dataset, train_data_ids)
            train_size = self.config.parameters.data.training_size
            # Reduce the size of the training dataset if train_frac is less than 1.0
            train_dataset, _ = random_split(
                train_dataset, [train_size, len(train_dataset) - train_size]
            )
            val_dataset = create_subset(full_dataset, val_data_ids)

            # Initialize DataLoaders for each subset
            self.train_dataloaders.append(DataLoader(
                train_dataset,
                batch_size=self.config.parameters.model.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.config.environment.num_workers,
            ))

            self.val_dataloaders.append(DataLoader(
                val_dataset,
                batch_size=self.config.parameters.model.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.config.environment.num_workers,
            ))

        test_dataset = create_subset(full_dataset, test_data_ids)

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.parameters.model.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.config.environment.num_workers,
        )


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
                dataset_conf = self.config.datasets.get(
                    self.config.parameters.data.dataset
                )
                if self.config.parameters.data.dataset != CFDDataset.DRIVAER_ML.value:
                    with open(
                            os.path.join(dataset_conf.subset_dir, ids_file), "r"
                    ) as file:
                        subset_ids_str = file.read().split()
                    subset_ids = list(map(int, subset_ids_str))
                    dataset_conf = self.config.datasets.get(
                        self.config.parameters.data.dataset
                    )
                    subset_indices = dataset.data_frame[
                        dataset.data_frame[dataset_conf.id_col].isin(subset_ids)
                    ].index.tolist()
                    # print(subset_indices)
                    return Subset(dataset, subset_indices)
                else:
                    with open(
                            os.path.join(dataset_conf.subset_dir, ids_file), "r"
                    ) as file:
                        subset_ids = file.read().split()
                    # Filter the dataset DataFrame based on subset IDs
                    dataset_conf = self.config.datasets.get(
                        self.config.parameters.data.dataset
                    )
                    subset_indices = dataset.data_frame[
                        dataset.data_frame[dataset_conf.id_col].isin(subset_ids)
                    ].index.tolist()
                    return Subset(dataset, subset_indices)
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

        full_dataset = self.dataset
        # Create training subset using the corresponding subset file
        train_dataset = create_subset(full_dataset, "train_design_ids.txt")
        train_size = self.config.parameters.data.training_size
        # Reduce the size of the training dataset if train_frac is less than 1.0
        train_dataset, _ = random_split(
            train_dataset, [train_size, len(train_dataset) - train_size]
        )

        val_dataset = create_subset(full_dataset, "val_design_ids.txt")
        test_dataset = create_subset(full_dataset, "test_design_ids.txt")
        # Initialize DataLoaders for each subset

        self.train_dataloader = GeoDataLoader(
            train_dataset,
            batch_size=self.config.parameters.model.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.environment.num_workers,
        )

        self.val_dataloader = GeoDataLoader(
            val_dataset,
            batch_size=self.config.parameters.model.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.config.environment.num_workers,
        )

        self.test_dataloader = GeoDataLoader(
            test_dataset,
            batch_size=self.config.parameters.model.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.config.environment.num_workers,
        )
