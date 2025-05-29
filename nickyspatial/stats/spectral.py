# -*- coding: utf-8 -*-
"""Spectral indices calculation module."""

import re
from typing import Dict

import numpy as np


class SpectralIndexCalculator:
    """A flexible spectral index calculator that supports custom formulas.

    Supports predefined indices from the awesome-spectral-indices catalogue.
    """

    def __init__(self):
        """Initialize the calculator with predefined indices."""
        self.predefined_indices = {
            "NDVI": {
                "formula": "(NIR - RED) / (NIR + RED)",
                "description": "Normalized Difference Vegetation Index",
                "reference": "Rouse et al. 1974",
            },
            "NDWI": {
                "formula": "(GREEN - NIR) / (GREEN + NIR)",
                "description": "Normalized Difference Water Index",
                "reference": "McFeeters 1996",
            },
            "NDBI": {
                "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
                "description": "Normalized Difference Built-up Index",
                "reference": "Zha et al. 2003",
            },
            "EVI": {
                "formula": "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
                "description": "Enhanced Vegetation Index",
                "reference": "Huete et al. 2002",
            },
            "SAVI": {
                "formula": "((NIR - RED) / (NIR + RED + 0.5)) * 1.5",
                "description": "Soil Adjusted Vegetation Index",
                "reference": "Huete 1988",
            },
            "GNDVI": {
                "formula": "(NIR - GREEN) / (NIR + GREEN)",
                "description": "Green Normalized Difference Vegetation Index",
                "reference": "Gitelson et al. 1996",
            },
            "NBR": {
                "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
                "description": "Normalized Burn Ratio",
                "reference": "Key & Benson 2006",
            },
            "MNDWI": {
                "formula": "(GREEN - SWIR1) / (GREEN + SWIR1)",
                "description": "Modified Normalized Difference Water Index",
                "reference": "Xu 2006",
            },
        }

    def _parse_formula(self, formula: str, bands: Dict[str, str]) -> str:
        """Parse a formula string and replace band names with actual column references."""
        parsed_formula = formula.upper()

        # Sort band names by length (descending) to avoid partial replacements
        sorted_bands = sorted(bands.items(), key=lambda x: len(x[0]), reverse=True)

        for band_name, column_name in sorted_bands:
            pattern = r"\b" + re.escape(band_name.upper()) + r"\b"
            parsed_formula = re.sub(pattern, f"bands['{column_name}']", parsed_formula)

        return parsed_formula

    def _evaluate_formula(self, formula: str, bands_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate a mathematical formula using band data."""
        safe_dict = {
            "bands": bands_data,
            "np": np,
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
            "sqrt": np.sqrt,
            "log": np.log,
            "exp": np.exp,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
        }

        try:
            result = eval(formula, safe_dict)
            if isinstance(result, (int, float)):
                result = np.full(len(next(iter(bands_data.values()))), result)
            return result.astype(float)
        except Exception as err:
            raise ValueError(f"Error evaluating formula '{formula}': {str(err)}") from err

    def _calculate_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for an array of values."""
        return {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "count": len(values),
        }


# Global calculator instance
_calculator = SpectralIndexCalculator()


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

    return _calculator._calculate_statistics(ndvi)


def attach_spectral_indices(layer, bands=None):
    """Calculate multiple spectral indices for objects in a layer.

    This function is designed to work with Layer.attach_function().

    Parameters:
    -----------
    layer : Layer
        Layer to calculate indices for
    bands : dict, optional
        Dictionary mapping band names to column names

    Returns:
    --------
    indices : dict
        Dictionary with calculated indices and their statistics
    """
    if layer.objects is None:
        return {"error": "Layer has no objects."}

    if bands is None:
        bands = {"blue": "band_1_mean", "green": "band_2_mean", "red": "band_3_mean", "nir": "band_4_mean"}

    ## Check available columns
    available_bands = {}
    for _band_name, column in bands.items():
        if column in layer.objects.columns:
            available_bands[_band_name] = column
        else:
            print(f"Warning: Band column '{column}' not found. Some indices may not be calculated.")
    if not available_bands:
        return {"error": "No valid bands found in layer objects."}

    indices = {}

    # NDVI (Normalized Difference Vegetation Index)
    if "nir" in available_bands and "red" in available_bands:
        if bands["nir"] in layer.objects.columns and bands["red"] in layer.objects.columns:
            ndvi = attach_ndvi(layer, bands["nir"], bands["red"], "NDVI")
            indices["NDVI"] = ndvi

    # NDWI (Normalized Difference Water Index)
    if "green" in bands and "nir" in available_bands:
        green_col = available_bands["green"]
        nir_col = available_bands["nir"]

        green = layer.objects[green_col]
        nir = layer.objects[nir_col]

        denominator = green + nir
        mask = denominator != 0

        ndwi = np.zeros(len(layer.objects))
        ndwi[mask] = (green[mask] - nir[mask]) / denominator[mask]

        layer.objects["NDWI"] = ndwi

        indices["NDWI"] = _calculator._calculate_statistics(ndwi)

    return indices


def attach_spectral_index(layer, index_name, formula=None, bands=None, output_column=None):
    """Calculate a single spectral index using a custom or predefined formula.

    This function is designed to work with Layer.attach_function().

    Parameters:
    -----------
    layer : Layer
        Layer object with objects DataFrame
    index_name : str
        Name of the index to calculate
    formula : str, optional
        Custom formula to use. If None, uses predefined formula for index_name
    bands : dict, optional
        Dictionary mapping band names to column names
    output_column : str, optional
        Name of output column. If None, uses index_name

    Returns:
    --------
    dict
        Dictionary with index statistics and metadata
    """
    if layer.objects is None:
        return {"error": "Layer has no objects"}

    # Set default bands mapping
    if bands is None:
        bands = {
            "BLUE": "band_1_mean",
            "GREEN": "band_2_mean",
            "RED": "band_3_mean",
            "NIR": "band_4_mean",
            "SWIR1": "band_5_mean",
            "SWIR2": "band_6_mean",
        }

    # Use predefined formula if no custom formula provided
    if formula is None:
        if index_name.upper() in _calculator.predefined_indices:
            formula = _calculator.predefined_indices[index_name.upper()]["formula"]
        else:
            return {"error": f"No predefined formula for '{index_name}' and no custom formula provided"}

    # Set output column name
    if output_column is None:
        output_column = index_name.upper()

    # Check available columns
    available_bands = {}
    for band_name, column_name in bands.items():
        if column_name in layer.objects.columns:
            available_bands[band_name] = column_name

    if not available_bands:
        return {"error": "No required band columns found in layer"}

    try:
        # Parse formula
        parsed_formula = _calculator._parse_formula(formula, available_bands)

        # Prepare band data
        bands_data = {}
        for _band_name, column_name in available_bands.items():
            bands_data[column_name] = layer.objects[column_name].values

        # Calculate index
        index_values = _calculator._evaluate_formula(parsed_formula, bands_data)

        # Add to layer
        layer.objects[output_column] = index_values

        # Calculate statistics
        stats = _calculator._calculate_statistics(index_values)

        # Prepare result
        result = {
            "index_name": index_name,
            "formula": formula,
            "output_column": output_column,
            "bands_used": available_bands,
            "statistics": stats,
        }

        # Add description if predefined index
        if index_name.upper() in _calculator.predefined_indices:
            result["description"] = _calculator.predefined_indices[index_name.upper()]["description"]
            result["reference"] = _calculator.predefined_indices[index_name.upper()]["reference"]

        return result

    except Exception as e:
        return {"error": f"Failed to calculate {index_name}: {str(e)}"}


def attach_multiple_spectral_indices(layer, indices, bands=None):
    """Calculate multiple spectral indices at once.

    This function is designed to work with Layer.attach_function().

    Parameters:
    -----------
    layer : Layer
        Layer object with objects DataFrame
    indices : list or dict
        List of predefined index names or dict with {index_name: formula}
    bands : dict, optional
        Dictionary mapping band names to column names

    Returns:
    --------
    dict
        Dictionary with results for each index
    """
    if layer.objects is None:
        return {"error": "Layer has no objects"}

    results = {}

    if isinstance(indices, list):
        # List of predefined indices
        for index_name in indices:
            result = attach_spectral_index(layer, index_name, bands=bands)
            results[index_name] = result
    else:
        # Dictionary with custom formulas
        for index_name, formula in indices.items():
            result = attach_spectral_index(layer, index_name, formula, bands)
            results[index_name] = result

    return results


def get_available_indices():
    """Get a list of all available predefined spectral indices.

    Returns:
    --------
    dict
        Dictionary with index names as keys and their metadata as values
    """
    return _calculator.predefined_indices.copy()


def add_custom_index(name, formula, description="", reference=""):
    """Add a custom index to the global predefined indices.

    Parameters:
    -----------
    name : str
        Name of the index
    formula : str
        Mathematical formula
    description : str, optional
        Description of the index
    reference : str, optional
        Reference/citation for the index
    """
    _calculator.predefined_indices[name.upper()] = {"formula": formula, "description": description, "reference": reference}
