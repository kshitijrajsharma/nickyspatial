# -*- coding: utf-8 -*-
"""Spectral indices calculation module."""

import numpy as np


def attach_ndvi(layer, nir_column="NIR_mean", red_column="Red_mean", output_column="NDVI"):
    """Calculate NDVI (Normalized Difference Vegetation Index) for objects in a layer.

    Parameters:
    -----------
    layer : Layer
        Layer to calculate NDVI for
    nir_column : str
        Column containing NIR band values
    red_column : str
        Column containing Red band values
    output_column : str
        Column to store NDVI values

    Returns:
    --------
    ndvi_stats : dict
        Dictionary with NDVI statistics
    """
    if layer.objects is None or nir_column not in layer.objects.columns or red_column not in layer.objects.columns:
        return {}

    nir = layer.objects[nir_column]
    red = layer.objects[red_column]

    denominator = nir + red
    mask = denominator != 0

    ndvi = np.zeros(len(layer.objects))
    ndvi[mask] = (nir[mask] - red[mask]) / denominator[mask]

    layer.objects[output_column] = ndvi

    ndvi_stats = {
        "mean": ndvi.mean(),
        "min": ndvi.min(),
        "max": ndvi.max(),
        "std": np.std(ndvi),
        "median": np.median(ndvi),
    }

    return ndvi_stats


def attach_spectral_indices(layer, bands=None):
    """Calculate multiple spectral indices for objects in a layer.

    Parameters:
    -----------
    layer : Layer
        Layer to calculate indices for
    bands : dict, optional
        Dictionary mapping band names to column names

    Returns:
    --------
    indices : dict
        Dictionary with calculated indices
    """
    if layer.objects is None:
        return {}

    if bands is None:
        bands = {
            "blue": "Blue_mean",
            "green": "Green_mean",
            "red": "Red_mean",
            "nir": "NIR_mean",
        }

    for _band_name, column in bands.items():
        if column not in layer.objects.columns:
            print(f"Warning: Band column '{column}' not found. Some indices may not be calculated.")

    indices = {}

    if "nir" in bands and "red" in bands:
        if bands["nir"] in layer.objects.columns and bands["red"] in layer.objects.columns:
            ndvi = attach_ndvi(layer, bands["nir"], bands["red"], "NDVI")
            indices["NDVI"] = ndvi

    if "green" in bands and "nir" in bands:
        if bands["green"] in layer.objects.columns and bands["nir"] in layer.objects.columns:
            green = layer.objects[bands["green"]]
            nir = layer.objects[bands["nir"]]

            denominator = green + nir
            mask = denominator != 0

            ndwi = np.zeros(len(layer.objects))
            ndwi[mask] = (green[mask] - nir[mask]) / denominator[mask]

            layer.objects["NDWI"] = ndwi

            indices["NDWI"] = {
                "mean": ndwi.mean(),
                "min": ndwi.min(),
                "max": ndwi.max(),
                "std": np.std(ndwi),
            }

    return indices
