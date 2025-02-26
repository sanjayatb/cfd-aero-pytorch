

def get_dataloaders(dataset_path: str, aero_coeff: str, subset_dir: str, num_points: int, batch_size: int,
                    train_frac: float = 1.0) -> tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        dataset_path (str): The file path to the dataset directory containing the STL files.
        aero_coeff (str): The path to the CSV file with metadata for the models.
        subset_dir (str): The directory containing the subset files (train, val, test).
        num_points (int): The number of points to sample from each point cloud in the dataset.
        batch_size (int): The number of samples per batch to load.
        train_frac (float): Fraction of the training data to be used for training.

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and test DataLoader.
    """
    # Initialize the full dataset
    full_dataset = AhmedMLDataset(root_dir=dataset_path, csv_file=aero_coeff, num_points=num_points,
                                  pointcloud_exist=False)

    # Helper function to create subsets from IDs in text files
    def create_subset(dataset, ids_file):
        try:
            with open(os.path.join(subset_dir, ids_file), 'r') as file:
                subset_ids_str = file.read().split()
            subset_ids = list(map(int, subset_ids_str))
            subset_indices = dataset.data_frame[dataset.data_frame['run'].isin(subset_ids)].index.tolist()
            # print(subset_indices)
            return Subset(dataset, subset_indices)
        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError(f"Error loading subset file {ids_file}: {e}")

    # Create training subset using the corresponding subset file
    train_dataset = create_subset(full_dataset, 'train_design_ids.txt')

    # Reduce the size of the training dataset if train_frac is less than 1.0
    if train_frac < 1.0:
        train_size = int(len(train_dataset) * train_frac)
        train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

    # Initialize DataLoaders for each subset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16)
    val_dataset = create_subset(full_dataset, 'val_design_ids.txt')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16)
    test_dataset = create_subset(full_dataset, 'test_design_ids.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=16)

    return train_dataloader, val_dataloader, test_dataloader