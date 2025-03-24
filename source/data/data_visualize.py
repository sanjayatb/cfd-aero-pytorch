import os
import pyvista as pv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import trimesh
import numpy as np


class InputDataViewer:
    def show_targets(self, file_path):
        data = pd.read_csv(file_path)

        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

        # Create a figure to hold the subplots
        plt.figure(figsize=(20, 10))

        # Histogram of Average Cd
        plt.subplot(2, 2, 1)
        sns.histplot(data["Average Cd"], kde=True)
        plt.title("Histogram of Average Drag Coefficient (Cd)")

        # Histogram of Average Cl
        plt.subplot(2, 2, 2)
        sns.histplot(data["Average Cl"], kde=True)
        plt.title("Histogram of Average Lift Coefficient (Cl)")

        # Scatter plot of Average Cd vs. Average Cl
        plt.subplot(2, 2, 3)
        sns.scatterplot(x="Average Cd", y="Average Cl", data=data)
        plt.title("Average Drag Coefficient (Cd) vs. Average Lift Coefficient (Cl)")

        # Box plot of all aerodynamic coefficients
        plt.subplot(2, 2, 4)
        melted_data = data.melt(
            value_vars=["Average Cd", "Average Cl", "Average Cl_f", "Average Cl_r"],
            var_name="Coefficient",
            value_name="Value",
        )
        sns.boxplot(x="Coefficient", y="Value", data=melted_data)
        plt.title("Box Plot of Aerodynamic Coefficients")

        plt.tight_layout()
        plt.show()

    def view_stl(self, folder_path):
        # List all .stl files in the folder
        stl_files = [f for f in os.listdir(folder_path) if f.endswith(".stl")]

        # Since we're going for a 2x3 grid, we'll take the first 6 .stl files for visualization
        stl_files_to_visualize = stl_files[:6]

        # Initialize a PyVista plotter with a 2x3 subplot grid
        plotter = pv.Plotter(shape=(2, 3))

        # Load and add each mesh to its respective subplot
        for i, file_name in enumerate(stl_files_to_visualize):
            # Calculate the subplot position
            row = i // 3  # Integer division determines the row
            col = i % 3  # Modulus determines the column

            # Activate the subplot at the calculated position
            plotter.subplot(row, col)

            # Load the mesh from file
            mesh = pv.read(os.path.join(folder_path, file_name))

            # Add the mesh to the current subplot
            plotter.add_mesh(mesh, color="lightgrey", show_edges=True)

            # Optional: Adjust the camera position or other settings here

        # Show the plotter window with all subplots
        plotter.show()

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

    def visualize_point_cloud(self, file_path, num_points=100000):
        """
        Visualizes the point cloud for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function retrieves the vertices for the specified design, converts them into a point cloud,
        and uses the z-coordinate for color mapping. PyVista's Eye-Dome Lighting is enabled for improved depth perception.
        """
        # Retrieve vertices and corresponding CD value for the specified index
        try:
            mesh = trimesh.load(file_path, force="mesh")
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
            vertices = self._sample_or_pad_vertices(vertices, num_points)
        except Exception as e:
            print(f"Failed to load STL file: {file_path}. Error: {e}")
            raise

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
        plt.savefig(f"../outputs/plots/point_cloud_{num_points}.png")


    def visualize_augmentations(self, file_path, num_points=100000):
        """
        Visualizes various augmentations applied to the point cloud of a specific design in the dataset.

        Args:
            idx (int): Index of the sample in the dataset to be visualized.

        This function retrieves the original point cloud for the specified design and then applies a series of augmentations,
        including translation, jittering, and point dropping. Each version of the point cloud (original and augmented) is then
        visualized in a 2x2 grid using PyVista to illustrate the effects of these augmentations.
        """
        # Retrieve the original point cloud without applying any augmentations
        try:
            mesh = trimesh.load(file_path, force="mesh")
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
            vertices = self._sample_or_pad_vertices(vertices, num_points)
        except Exception as e:
            print(f"Failed to load STL file: {file_path}. Error: {e}")
            raise
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
