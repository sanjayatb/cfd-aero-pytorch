import os
from typing import Any
import pyvista as pv
import vtk


def read_stl(file_path: str) -> vtk.vtkPolyData:
    """
    Read an STL file and return the polydata.

    Parameters
    ----------
    file_path : str
        Path to the STL file.

    Returns
    -------
    vtkPolyData
        The polydata read from the STL file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .stl extension
    if not file_path.endswith(".stl"):
        raise ValueError(f"Expected a .stl file, got {file_path}")

    # Create an STL reader
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the polydata
    polydata = reader.GetOutput()

    # Check if polydata is valid
    if polydata is None:
        raise ValueError(f"Failed to read polydata from {file_path}")

    return polydata

def read_vtp(file_path: str) -> Any:  # TODO add support for older format (VTK)
    """
    Read a VTP file and return the polydata.

    Parameters
    ----------
    file_path : str
        Path to the VTP file.

    Returns
    -------
    vtkPolyData
        The polydata read from the VTP file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtp extension
    if not file_path.endswith(".vtp"):
        raise ValueError(f"Expected a .vtp file, got {file_path}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the polydata
    polydata = reader.GetOutput()

    # Check if polydata is valid
    if polydata is None:
        raise ValueError(f"Failed to read polydata from {file_path}")

    return polydata

def read_vtk(file_path: str) -> Any:  # TODO add support for older format (VTK)
    """
    Read a VTP file and return the polydata.

    Parameters
    ----------
    file_path : str
        Path to the VTP file.

    Returns
    -------
    vtkPolyData
        The polydata read from the VTP file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtk extension
    if not file_path.endswith(".vtk"):
        raise ValueError(f"Expected a .vtk file, got {file_path}")

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the polydata
    polydata = reader.GetOutput()

    # Check if polydata is valid
    if polydata is None:
        raise ValueError(f"Failed to read polydata from {file_path}")

    return polydata

def read_vtu(file_path: str) -> Any:
    """
    Read a VTU file and return the unstructured grid data.

    Parameters
    ----------
    file_path : str
        Path to the VTU file.

    Returns
    -------
    vtkUnstructuredGrid
        The unstructured grid data read from the VTU file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtu extension
    if not file_path.endswith(".vtu"):
        raise ValueError(f"Expected a .vtu file, got {file_path}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the unstructured grid data
    grid = reader.GetOutput()

    # Check if grid is valid
    if grid is None:
        raise ValueError(f"Failed to read unstructured grid data from {file_path}")

    return grid

def convert_to_triangular_mesh(
    polydata, write=False, output_filename="surface_mesh_triangular.vtu"
):
    """Converts a vtkPolyData object to a triangular mesh."""
    tet_filter = vtk.vtkDataSetTriangleFilter()
    tet_filter.SetInputData(polydata)
    tet_filter.Update()

    tet_mesh = pv.wrap(tet_filter.GetOutput())

    if write:
        tet_mesh.save(output_filename)

    return tet_mesh

def fetch_mesh_vertices(mesh):
    """Fetches the vertices of a mesh."""
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = [points.GetPoint(i) for i in range(num_points)]
    return vertices

