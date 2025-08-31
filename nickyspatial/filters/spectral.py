# -*- coding: utf-8 -*-
"""Performs spectral-based manipulations of imagery, including band arithmetic and transformations.

It supports generating new spectral bands or combinations to highlight specific features.
It also includes functions for enhancing contrast and applying spectral filters based on mathematical expressions.
This module is designed to work with raster .
The functions here include contrast enhancement, spectral filtering, and band arithmetic.
Not a great fan of these but might be handy sometime
"""

import numpy as np

from ..core.layer import Layer


def enhance_contrast(
    source_layer,
    percentile_min=2,
    percentile_max=98,
    layer_manager=None,
    layer_name=None,
):
    """Enhance contrast in source layer raster data.

    Parameters:
    -----------
    source_layer : Layer
        Source layer with raster data
    percentile_min : float
        Lower percentile for contrast stretching
    percentile_max : float
        Upper percentile for contrast stretching
    layer_manager : LayerManager, optional
        Layer manager to add the result layer to
    layer_name : str, optional
        Name for the result layer

    Returns:
    --------
    result_layer : Layer
        Layer with enhanced contrast
    """
    if source_layer.raster is None:
        raise ValueError("Source layer must have raster data")

    if not layer_name:
        layer_name = f"{source_layer.name}_enhanced"

    result_layer = Layer(name=layer_name, parent=source_layer, layer_type="filter")
    result_layer.transform = source_layer.transform
    result_layer.crs = source_layer.crs
    result_layer.objects = source_layer.objects.copy() if source_layer.objects is not None else None

    result_layer.metadata = {
        "filter_type": "enhance_contrast",
        "percentile_min": percentile_min,
        "percentile_max": percentile_max,
    }

    enhanced_raster = source_layer.raster.copy()

    p_min = np.percentile(enhanced_raster, percentile_min)
    p_max = np.percentile(enhanced_raster, percentile_max)

    enhanced_raster = np.clip(enhanced_raster, p_min, p_max)
    enhanced_raster = (enhanced_raster - p_min) / (p_max - p_min)

    result_layer.raster = enhanced_raster

    if layer_manager:
        layer_manager.add_layer(result_layer)

    return result_layer


def spectral_filter(source_layer, expression, layer_manager=None, layer_name=None):
    """Apply a spectral filter based on a mathematical expression.

    Parameters:
    -----------
    source_layer : Layer
        Source layer with segment statistics
    expression : str
        Mathematical expression to apply (e.g., "NDVI > 0.5")
    layer_manager : LayerManager, optional
        Layer manager to add the result layer to
    layer_name : str, optional
        Name for the result layer

    Returns:
    --------
    result_layer : Layer
        Layer with filtered segments
    """
    import numexpr as ne

    if not layer_name:
        layer_name = f"{source_layer.name}_spectral_filtered"

    result_layer = Layer(name=layer_name, parent=source_layer, layer_type="filter")
    result_layer.transform = source_layer.transform
    result_layer.crs = source_layer.crs

    result_layer.metadata = {"filter_type": "spectral_filter", "expression": expression}

    objects = source_layer.objects.copy()

    try:
        local_dict = {col: objects[col].values for col in objects.columns if col != "geometry"}
        mask = ne.evaluate(expression, local_dict=local_dict)
        mask = np.array(mask, dtype=bool)

        filtered_objects = objects.iloc[mask]
        result_layer.objects = filtered_objects

        if source_layer.raster is not None:
            kept_ids = set(filtered_objects["segment_id"])
            segments_raster = source_layer.raster.copy()
            raster_mask = np.isin(segments_raster, list(kept_ids))
            segments_raster[~raster_mask] = 0
            result_layer.raster = segments_raster

    except Exception as e:
        raise ValueError(f"Error applying spectral filter: {str(e)}") from e

    if layer_manager:
        layer_manager.add_layer(result_layer)

    return result_layer
