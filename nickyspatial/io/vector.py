"""Manages vector data I/O, supporting formats like Shapefile and GeoJSON.

This module typically offers utilities for handling attributes, geometries, and coordinate reference systems.
"""

import os

import geopandas as gpd


def read_vector(vector_path):
    """Read a vector file into a GeoDataFrame.

    Parameters:
    -----------
    vector_path : str
        Path to the vector file

    Returns:
    --------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with vector data
    """
    return gpd.read_file(vector_path)


def write_vector(gdf, output_path):
    """Write a GeoDataFrame to a vector file.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to write
    output_path : str
        Path to the output vector file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_extension = os.path.splitext(output_path)[1].lower()

    if file_extension == ".shp":
        gdf.to_file(output_path)
    elif file_extension == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError(f"Unsupported vector format: {file_extension}")


def layer_to_vector(layer, output_path):
    """Save a layer's objects to a vector file.

    Parameters:
    -----------
    layer : Layer
        Layer to save
    output_path : str
        Path to the output vector file
    """
    if layer.objects is None:
        raise ValueError("Layer has no vector objects")

    write_vector(layer.objects, output_path)
