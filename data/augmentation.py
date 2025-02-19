import torch
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """
    @staticmethod
    def translate_pointcloud(pointcloud: torch.Tensor, translation_range: Tuple[float, float] = (2./3., 3./2.)) -> torch.Tensor:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a torch.Tensor.
        """
        # Randomly choose translation factors and apply them to the pointcloud
        xyz1 = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return torch.tensor(translated_pointcloud, dtype=torch.float32)

    @staticmethod
    def jitter_pointcloud(pointcloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a torch.Tensor.
        """
        # Add Gaussian noise and clip to the specified range
        N, C = pointcloud.shape
        jittered_pointcloud = pointcloud + torch.clamp(sigma * torch.randn(N, C), -clip, clip)
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a torch.Tensor.
        """
        # Calculate the number of points to drop
        num_drop = int(drop_rate * pointcloud.size(0))
        # Generate random indices for points to drop
        drop_indices = np.random.choice(pointcloud.size(0), num_drop, replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.size(0)), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud