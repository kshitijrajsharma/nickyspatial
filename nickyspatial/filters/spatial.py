# -*- coding: utf-8 -*-
"""Implements spatial operations like smoothing and morphological transformations.

These filters can modify the geometry or arrangement of pixel values to enhance or simplify data for object analysis.
The functions here include smoothing boundaries, merging small segments, and selecting segments based on area.
These operations are essential for preparing data for object-based image analysis, especially in remote sensing applications.
The functions are designed to work with raster data and can be applied to layers created from segmentation algorithms.
"""

import geopandas as gpd
import numpy as np

from ..core.layer import Layer


def smooth_boundaries(source_layer, iterations=1, layer_manager=None, layer_name=None):
    """Smooth segment boundaries by applying morphological operations.

    Parameters:
    -----------
    source_layer : Layer
        Source layer with segments to smooth
    iterations : int
        Number of smoothing iterations to apply
    layer_manager : LayerManager, optional
        Layer manager to add the result layer to
    layer_name : str, optional
        Name for the result layer

    Returns:
    --------
    result_layer : Layer
        Layer with smoothed segment boundaries
    """
    if not layer_name:
        layer_name = f"{source_layer.name}_smoothed"

    result_layer = Layer(name=layer_name, parent=source_layer, layer_type="filter")
    result_layer.transform = source_layer.transform
    result_layer.crs = source_layer.crs
    result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

    result_layer.metadata = {
        "filter_type": "smooth_boundaries",
        "iterations": iterations,
    }

    objects = source_layer.objects.copy()

    smoothed_geometries = []
    for geom in objects.geometry:
        smoothed_geom = geom
        for _ in range(iterations):
            buffer_distance = np.sqrt(smoothed_geom.area) * 0.01
            smoothed_geom = smoothed_geom.buffer(-buffer_distance).buffer(buffer_distance * 2)

        if not smoothed_geom.is_valid:
            smoothed_geom = smoothed_geom.buffer(0)

        smoothed_geometries.append(smoothed_geom)

    objects.geometry = smoothed_geometries
    result_layer.objects = objects

    if layer_manager:
        layer_manager.add_layer(result_layer)

    return result_layer


def merge_small_segments(source_layer, min_size, attribute="area_pixels", layer_manager=None, layer_name=None):
    """Merge small segments with their largest neighbor.

    Parameters:
    -----------
    source_layer : Layer
        Source layer with segments to merge
    min_size : float
        Minimum segment size threshold
    attribute : str
        Attribute to use for size comparison
    layer_manager : LayerManager, optional
        Layer manager to add the result layer to
    layer_name : str, optional
        Name for the result layer

    Returns:
    --------
    result_layer : Layer
        Layer with merged segments
    """
    if not layer_name:
        layer_name = f"{source_layer.name}_merged"

    result_layer = Layer(name=layer_name, parent=source_layer, layer_type="filter")
    result_layer.transform = source_layer.transform
    result_layer.crs = source_layer.crs

    result_layer.metadata = {
        "filter_type": "merge_small_segments",
        "min_size": min_size,
        "attribute": attribute,
    }

    objects = source_layer.objects.copy()
    small_segments = objects[objects[attribute] < min_size]

    if len(small_segments) == 0:
        result_layer.objects = objects
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer

    for idx, small_segment in small_segments.iterrows():
        if idx not in objects.index:
            continue

        neighbors = objects[objects.index != idx].overlay(
            gpd.GeoDataFrame(geometry=[small_segment.geometry], crs=objects.crs),
            how="intersection",
        )

        if len(neighbors) == 0:
            continue

        largest_neighbor_idx = neighbors[attribute].idxmax()

        largest_neighbor = objects.loc[largest_neighbor_idx]
        merged_geometry = largest_neighbor.geometry.union(small_segment.geometry)

        objects.at[largest_neighbor_idx, "geometry"] = merged_geometry
        objects.at[largest_neighbor_idx, attribute] += small_segment[attribute]

        objects = objects.drop(idx)

    if source_layer.raster is not None:
        segments_raster = source_layer.raster.copy()

        old_to_new = {}
        for _idx, obj in objects.iterrows():
            old_id = obj["segment_id"]
            old_to_new[old_id] = old_id

        for idx, small_segment in small_segments.iterrows():
            if idx not in objects.index:
                old_id = small_segment["segment_id"]

                touching_segments = objects.intersects(small_segment.geometry)
                if any(touching_segments):
                    new_id = objects[touching_segments].iloc[0]["segment_id"]
                    old_to_new[old_id] = new_id

        for old_id, new_id in old_to_new.items():
            if old_id != new_id:
                segments_raster[segments_raster == old_id] = new_id

        result_layer.raster = segments_raster

    result_layer.objects = objects

    if layer_manager:
        layer_manager.add_layer(result_layer)

    return result_layer


def select_by_area(
    source_layer,
    min_area=None,
    max_area=None,
    area_column="area_units",
    layer_manager=None,
    layer_name=None,
):
    """Select segments based on area.

    Parameters:
    -----------
    source_layer : Layer
        Source layer with segments to filter
    min_area : float, optional
        Minimum area threshold
    max_area : float, optional
        Maximum area threshold
    area_column : str
        Column containing area values
    layer_manager : LayerManager, optional
        Layer manager to add the result layer to
    layer_name : str, optional
        Name for the result layer

    Returns:
    --------
    result_layer : Layer
        Layer with filtered segments
    """
    if not layer_name:
        layer_name = f"{source_layer.name}_area_filtered"

    result_layer = Layer(name=layer_name, parent=source_layer, layer_type="filter")
    result_layer.transform = source_layer.transform
    result_layer.crs = source_layer.crs

    result_layer.metadata = {
        "filter_type": "select_by_area",
        "min_area": min_area,
        "max_area": max_area,
        "area_column": area_column,
    }

    objects = source_layer.objects.copy()

    if min_area is not None:
        objects = objects[objects[area_column] >= min_area]

    if max_area is not None:
        objects = objects[objects[area_column] <= max_area]

    result_layer.objects = objects

    if source_layer.raster is not None:
        kept_ids = set(objects["segment_id"])

        segments_raster = source_layer.raster.copy()
        mask = np.isin(segments_raster, list(kept_ids))

        segments_raster[~mask] = 0

        result_layer.raster = segments_raster

    if layer_manager:
        layer_manager.add_layer(result_layer)

    return result_layer
