import os
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import trimesh
from torch.utils.data import Dataset, random_split
import pyvista as pv
import seaborn as sns
from typing import Callable, Optional, Tuple, List
from torch_geometric.data import Data

from source.config.dto import Config
from source.data.augmentation import DataAugmentation
from source.data.enums import CFDDataset


class WindsorMLDataset(Dataset):
    """
    PyTorch Dataset class for the Windsor dataset, handling loading, transforming, and augmenting 3D car models.
    """

    def __init__(
            self,
            config: Config,
            root_dir: str,
            csv_file: str,
            num_points: int,
            transform: Optional[Callable] = None,
            pointcloud_exist: bool = False,
    ):
        """
        Initializes the WindsorMLDataset instance.

        Args:
            root_dir: Directory containing the STL files for 3D car models.
            csv_file: Path to the CSV file with metadata for the models.
            num_points: Fixed number of points to sample from each 3D model.
            transform: Optional transform function to apply to each sample.
            pointcloud_exist (bool): Whether the point clouds already exist as .pt files.
        """
        self.root_dir = root_dir
        self.config = config
        try:
            self.data_frame = pd.read_csv(csv_file)
            id_column = self.config.datasets[CFDDataset.WINDSOR_ML.value].id_col
            self.data_frame[id_column] = "windsor_" + self.data_frame[id_column].astype("str")
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise

        self.transform = transform
        self.num_points = num_points
        self.augmentation = DataAugmentation()
        self.pointcloud_exist = pointcloud_exist
        self.cache = {}

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def get_all_file_names(self):
        folder_path = Path(self.root_dir)
        file_names = [file.stem for file in folder_path.iterdir() if file.is_file()]
        file_names = sorted(file_names)
        if len(file_names) > self.config.parameters.data.max_total_samples:
            return file_names[0:self.config.parameters.data.max_total_samples]
        else:
            return file_names

    def min_max_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.
        """
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def z_score_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data using z-score normalization (standard score).
        """
        mean_vals = data.mean(dim=0, keepdim=True)
        std_vals = data.std(dim=0, keepdim=True)
        normalized_data = (data - mean_vals) / std_vals
        return normalized_data

    def mean_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [-1, 1] based on mean and range.
        """
        mean_vals = data.mean(dim=0, keepdim=True)
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        normalized_data = (data - mean_vals) / (max_vals - min_vals)
        return normalized_data

    def _sample_or_pad_vertices(
            self, vertices: torch.Tensor, num_points: int
    ) -> torch.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a torch.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.size(0)
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        elif num_vertices < num_points:
            padding = torch.zeros((num_points - num_vertices, 3), dtype=torch.float32)
            vertices = torch.cat((vertices, padding), dim=0)
        return vertices

    def _load_point_cloud(self, design_id: str) -> Optional[torch.Tensor]:
        load_path = os.path.join(self.root_dir, f"{design_id}.pt")
        if os.path.exists(load_path) and os.path.getsize(load_path) > 0:
            try:
                return torch.load(load_path)
            except (EOFError, RuntimeError) as e:
                # logging.error(f"Failed to load point cloud file {load_path}: {e}")
                return None
        else:
            # logging.error(f"Point cloud file {load_path} does not exist or is empty.")
            return None

    def __getitem__(
            self, idx: int, apply_augmentations: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample and its corresponding label from the dataset, with an option to apply augmentations.

        Args:
            idx (int): Index of the sample to retrieve.
            apply_augmentations (bool, optional): Whether to apply data augmentations. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sample (point cloud) and its label (Cd value).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx in self.cache:
            return self.cache[idx]
        while True:
            row = self.data_frame.iloc[idx]
            dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
            design_id = row[dataset_conf.id_col]
            cd_value = row[dataset_conf.target_col]

            if self.pointcloud_exist:
                vertices = self._load_point_cloud(design_id)

                if vertices is None:
                    # logging.warning(f"Skipping design {design_id} because point cloud is not found or corrupted.")
                    idx = (idx + 1) % len(self.data_frame)
                    continue
            else:
                geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")
                try:
                    mesh = trimesh.load(geometry_path, force="mesh")
                    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
                    vertices = self._sample_or_pad_vertices(vertices, self.num_points)
                except Exception as e:
                    logging.error(
                        f"Failed to load STL file: {geometry_path}. Error: {e}"
                    )
                    raise

            if apply_augmentations:
                vertices = self.augmentation.translate_pointcloud(vertices.numpy())
                vertices = self.augmentation.jitter_pointcloud(vertices)

            if self.transform:
                vertices = self.transform(vertices)

            point_cloud_normalized = self.min_max_normalize(vertices)
            cd_value = torch.tensor(float(cd_value), dtype=torch.float32).view(-1)

            self.cache[idx] = (point_cloud_normalized, cd_value)
            return point_cloud_normalized, cd_value

    def split_data(
            self,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            train_ratio: The proportion of the data to be used for training.
            val_ratio: The proportion of the data to be used for validation.
            test_ratio: The proportion of the data to be used for testing.

        Returns:
            Indices for the training, validation, and test sets.
        """
        assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
        num_samples = len(self)
        indices = list(range(num_samples))
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        test_size = num_samples - train_size - val_size
        train_indices, val_indices, test_indices = random_split(
            indices, [train_size, val_size, test_size]
        )
        return train_indices, val_indices, test_indices

    def visualize_mesh(self, idx):
        """
        Visualize the STL mesh for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file corresponding to the design ID at the given index,
        wraps it using PyVista for visualization, and then sets up a PyVista plotter to display the mesh.
        """
        row = self.data_frame.iloc[idx]
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        design_id = row[dataset_conf.id_col]
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        try:
            mesh = trimesh.load(geometry_path, force="mesh")
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        pv_mesh = pv.wrap(mesh)
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color="lightgrey", show_edges=True)
        plotter.add_axes()

        camera_position = [
            (-11.073024242161921, -5.621499358347753, 5.862225824910342),
            (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
            (0.34000174095454166, 0.10379556639001211, 0.9346792479485448),
        ]
        plotter.camera_position = camera_position
        plotter.show()

    def visualize_mesh_with_node(self, idx):
        """
        Visualizes the mesh for a specific design from the dataset with nodes highlighted.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file and highlights the nodes (vertices) of the mesh using spheres.
        It uses seaborn to obtain visually distinct colors for the mesh and nodes.
        """
        row = self.data_frame.iloc[idx]
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        design_id = row[dataset_conf.id_col]
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        try:
            mesh = trimesh.load(geometry_path, force="mesh")
            pv_mesh = pv.wrap(mesh)
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        plotter = pv.Plotter()
        sns_blue = sns.color_palette("colorblind")[0]

        plotter.add_mesh(
            pv_mesh, color="lightgrey", show_edges=True, edge_color="black"
        )
        nodes = pv_mesh.points
        plotter.add_points(
            nodes, color=sns_blue, point_size=5, render_points_as_spheres=True
        )
        plotter.add_axes()
        plotter.show()

    def visualize_point_cloud(self, idx):
        """
        Visualizes the point cloud for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function retrieves the vertices for the specified design, converts them into a point cloud,
        and uses the z-coordinate for color mapping. PyVista's Eye-Dome Lighting is enabled for improved depth perception.
        """
        # Retrieve vertices and corresponding CD value for the specified index
        vertices, _ = self.__getitem__(idx)
        vertices = vertices.numpy()

        # Convert vertices to a PyVista PolyData object for visualization
        point_cloud = pv.PolyData(vertices)
        colors = vertices[:, 2]  # Using the z-coordinate for color mapping
        point_cloud["colors"] = colors  # Add the colors to the point cloud

        # Set up the PyVista plotter
        plotter = pv.Plotter()

        # Add the point cloud to the plotter with color mapping based on the z-coordinate
        plotter.add_points(
            point_cloud,
            scalars="colors",
            cmap="Blues",
            point_size=3,
            render_points_as_spheres=True,
        )

        # Enable Eye-Dome Lighting for better depth perception
        plotter.enable_eye_dome_lighting()

        # Add axes for orientation and display the plotter window
        plotter.add_axes()
        camera_position = [
            (-11.073024242161921, -5.621499358347753, 5.862225824910342),
            (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
            (0.34000174095454166, 0.10379556639001211, 0.9346792479485448),
        ]

        # Set the camera position
        plotter.camera_position = camera_position

        plotter.show()

    def visualize_augmentations(self, idx):
        """
        Visualizes various augmentations applied to the point cloud of a specific design in the dataset.

        Args:
            idx (int): Index of the sample in the dataset to be visualized.

        This function retrieves the original point cloud for the specified design and then applies a series of augmentations,
        including translation, jittering, and point dropping. Each version of the point cloud (original and augmented) is then
        visualized in a 2x2 grid using PyVista to illustrate the effects of these augmentations.
        """
        # Retrieve the original point cloud without applying any augmentations
        vertices, _ = self.__getitem__(idx, apply_augmentations=False)
        original_pc = pv.PolyData(vertices.numpy())

        # Apply translation augmentation to the original point cloud
        translated_pc = self.augmentation.translate_pointcloud(vertices.numpy())
        # Apply jitter augmentation to the translated point cloud
        jittered_pc = self.augmentation.jitter_pointcloud(translated_pc)
        # Apply point dropping augmentation to the jittered point cloud
        dropped_pc = self.augmentation.drop_points(jittered_pc)

        # Initialize a PyVista plotter with a 2x2 grid for displaying the point clouds
        plotter = pv.Plotter(shape=(2, 2))

        # Display the original point cloud in the top left corner of the grid
        plotter.subplot(0, 0)  # Select the first subplot
        plotter.add_text("Original Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(
            original_pc, color="black", point_size=3
        )  # Add the original point cloud to the plot

        # Display the translated point cloud in the top right corner of the grid
        plotter.subplot(0, 1)  # Select the second subplot
        plotter.add_text("Translated Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(
            pv.PolyData(translated_pc.numpy()), color="lightblue", point_size=3
        )  # Add the translated point cloud to the plot

        # Display the jittered point cloud in the bottom left corner of the grid
        plotter.subplot(1, 0)  # Select the third subplot
        plotter.add_text("Jittered Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(
            pv.PolyData(jittered_pc.numpy()), color="lightgreen", point_size=3
        )  # Add the jittered point cloud to the plot

        # Display the dropped point cloud in the bottom right corner of the grid
        plotter.subplot(1, 1)  # Select the fourth subplot
        plotter.add_text("Dropped Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(
            pv.PolyData(dropped_pc.numpy()), color="salmon", point_size=3
        )  # Add the dropped point cloud to the plot

        # Display the plot with all point clouds
        plotter.show()


class WindsorMLGNNDataset(Dataset):
    """
    PyTorch Dataset for loading and processing the Windsor dataset into graph format suitable for GNNs.
    """

    def __init__(
            self, config: Config, root_dir: str, csv_file: str, normalize: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the directory containing the STL files.
            csv_file (str): Path to the CSV file containing metadata such as aerodynamic coefficients.
            normalize (bool): Whether to normalize the node features.
        """
        self.root_dir = root_dir
        self.config = config
        self.data_frame = pd.read_csv(csv_file)
        self.normalize = normalize
        self.cache = {}

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data_frame)

    def min_max_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the data to the range [0, 1] based on min and max values.

        Args:
            data (torch.Tensor): The input data tensor to be normalized.

        Returns:
            torch.Tensor: The normalized data tensor.
        """
        min_vals, _ = data.min(dim=0, keepdim=True)
        max_vals, _ = data.max(dim=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def __getitem__(self, idx: int) -> Data:
        """
        Get a graph data item for GNN processing.

        Args:
            idx (int): Index of the item.

        Returns:
            Data: A PyTorch Geometric Data object containing edge_index, x (node features), and y (target variable).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx in self.cache:
            return self.cache[idx]

        row = self.data_frame.iloc[idx]
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        design_id = row[dataset_conf.id_col]
        stl_path = os.path.join(self.root_dir, f"{design_id}.stl")
        cd_value = row[dataset_conf.target_col]

        # Load the mesh from STL
        try:
            mesh = trimesh.load(stl_path, force="mesh")
        except Exception as e:
            logging.error(f"Failed to load STL file: {stl_path}. Error: {e}")
            raise

        # Convert mesh to graph
        edge_index = torch.tensor(np.array(mesh.edges).T, dtype=torch.long)
        x = torch.tensor(
            mesh.vertices, dtype=torch.float
        )  # Using vertex positions as features

        if self.normalize:
            x = self.min_max_normalize(x)

        y = torch.tensor([cd_value], dtype=torch.float)  # Target variable as tensor

        # Create a graph data object
        data = Data(x=x, edge_index=edge_index, y=y)

        self.cache[idx] = data
        return data

    def visualize_mesh_with_node(self, idx: int) -> None:
        """
        Visualizes the mesh of a given sample index with triangles in light grey and nodes highlighted as spheres.

        Args:
            idx (int): Index of the sample to visualize.
        """
        row = self.data_frame.iloc[idx]
        dataset_conf = self.config.datasets.get(self.config.parameters.data.dataset)
        design_id = row[dataset_conf.id_col]
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        try:
            mesh = trimesh.load(geometry_path, force="mesh")
            pv_mesh = pv.wrap(mesh)
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        plotter = pv.Plotter()
        sns_blue = sns.color_palette("colorblind")[0]

        # Add the mesh to the plotter with light grey color
        plotter.add_mesh(
            pv_mesh, color="lightgrey", show_edges=True, edge_color="black"
        )

        # Highlight nodes as spheres
        nodes = pv_mesh.points
        plotter.add_points(
            nodes, color=sns_blue, point_size=5, render_points_as_spheres=True
        )  # Increase point_size as needed

        plotter.add_axes()
        camera_position = [
            (-11.073024242161921, -5.621499358347753, 5.862225824910342),
            (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
            (0.34000174095454166, 0.10379556639001211, 0.9346792479485448),
        ]

        # Set the camera position
        plotter.camera_position = camera_position
        plotter.show()

    def visualize_graph(self, idx: int) -> None:
        """
        Visualizes the graph representation of the 3D mesh using PyVista.

        Args:
            idx (int): Index of the sample to visualize.
        """
        data = self[idx]  # Get the data object
        mesh = pv.PolyData(data.x.numpy())  # Create a PyVista mesh from node features

        # Create edges array suitable for PyVista
        edges = data.edge_index.t().numpy()
        lines = np.full((edges.shape[0], 3), 2, dtype=np.int_)
        lines[:, 1:] = edges

        mesh.lines = lines
        mesh["scalars"] = np.random.rand(mesh.n_points)  # Random colors for nodes

        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh,
            show_edges=True,
            line_width=1,
            color="white",
            point_size=8,
            render_points_as_spheres=True,
        )
        plotter.add_scalar_bar("Scalar Values", "scalars")

        # Optional: highlight edges for clarity
        edge_points = mesh.points[edges.flatten()]
        lines = pv.lines_from_points(edge_points)
        plotter.add_mesh(lines, color="blue", line_width=2)

        plotter.show()
