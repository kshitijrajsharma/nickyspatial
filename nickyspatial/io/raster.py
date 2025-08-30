# -*- coding: utf-8 -*-
"""Handles raster input and output operations, including reading and saving multi-band images.

Functions in this module may also provide metadata parsing and coordinate transform tools.
"""

import os

import numpy as np
import rasterio
from rasterio.transform import from_origin


def read_raster(raster_path):
    """Read a raster file and return its data, transform, and CRS.

    Parameters:
    -----------
    raster_path : str
        Path to the raster file

    Returns:
    --------
    image_data : numpy.ndarray
        Array with raster data values
    transform : affine.Affine
        Affine transformation for the raster
    crs : rasterio.crs.CRS
        Coordinate reference system
    """
    with rasterio.open(raster_path) as src:
        image_data = src.read()
        transform = src.transform
        crs = src.crs

    return image_data, transform, crs


def write_raster(output_path, data, transform, crs, nodata=None):
    """Write raster data to a file.

    Parameters:
    -----------
    output_path : str
        Path to the output raster file
    data : numpy.ndarray
        Array with raster data values
    transform : affine.Affine
        Affine transformation for the raster
    crs : rasterio.crs.CRS
        Coordinate reference system
    nodata : int or float, optional
        No data value
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(data.shape) == 2:
        data = data.reshape(1, *data.shape)

    height, width = data.shape[-2], data.shape[-1]
    count = data.shape[0]

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def layer_to_raster(layer, output_path, column=None, nodata=0):
    """Save a layer to a raster file.

    Parameters:
    -----------
    layer : Layer
        Layer to save
    output_path : str
        Path to the output raster file
    column : str, optional
        Column to rasterize (if saving from vector objects)
    nodata : int or float, optional
        No data value
    """
    from rasterio import features

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if layer.raster is not None and column is None:
        write_raster(
            output_path,
            layer.raster.reshape(1, *layer.raster.shape),
            layer.transform,
            layer.crs,
            nodata,
        )
        return

    if layer.objects is not None and column is not None:
        if column not in layer.objects.columns:
            raise ValueError(f"Column '{column}' not found in layer objects")

        objects = layer.objects
        col_values = objects[column]
        if np.issubdtype(col_values.dtype, np.number):
            shapes = [(geom, float(val)) for geom, val in zip(objects.geometry, col_values, strict=False)]
        else:
            unique_vals = col_values.unique()
            val_map = {val: idx for idx, val in enumerate(unique_vals)}
            print(f"Mapping categorical values: {val_map}")
            shapes = [(geom, val_map[val]) for geom, val in zip(objects.geometry, col_values, strict=False)]
        if layer.raster is not None:
            if len(layer.raster.shape) == 3:
                height, width = layer.raster.shape[1], layer.raster.shape[2]
            else:
                height, width = layer.raster.shape
            out_shape = (height, width)
        else:
            bounds = objects.total_bounds
            resolution = 10
            if layer.transform:
                resolution = abs(layer.transform.a)
            width = int((bounds[2] - bounds[0]) / resolution)
            height = int((bounds[3] - bounds[1]) / resolution)
            out_shape = (height, width)
            if layer.transform is None:
                layer.transform = from_origin(bounds[0], bounds[3], resolution, resolution)

        output = np.ones(out_shape, dtype=np.float32) * nodata

        features.rasterize(shapes, out=output, transform=layer.transform, fill=nodata)

        write_raster(
            output_path,
            output.reshape(1, *out_shape),
            layer.transform,
            layer.crs,
            nodata,
        )
    else:
        raise ValueError("Layer must have either raster data or objects with a specified column")
