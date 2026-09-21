"""Spatial statistics for layers in NickySpatial."""

import numpy as np


def attach_area_stats(layer, area_column="area_units", by_class=None):
    """Calculate area statistics for objects in a layer.

    Parameters:
    -----------
    layer : Layer
        Layer to calculate statistics for
    area_column : str
        Column containing area values
    by_class : str, optional
        Column to group by (e.g., 'classification')

    Returns:
    --------
    stats : dict
        Dictionary with area statistics
    """
    if layer.objects is None or area_column not in layer.objects.columns:
        return {}

    total_area = layer.objects[area_column].sum()

    if by_class and by_class in layer.objects.columns:
        class_areas = {}
        class_percentages = {}

        for class_value, group in layer.objects.groupby(by_class):
            if class_value is None:
                continue

            class_area = group[area_column].sum()
            class_percentage = (class_area / total_area * 100).round(2)

            class_areas[class_value] = class_area
            class_percentages[class_value] = class_percentage

        stats = {
            "total_area": total_area,
            "class_areas": class_areas,
            "class_percentages": class_percentages,
        }
    else:
        areas = layer.objects[area_column]

        stats = {
            "total_area": total_area,
            "min_area": areas.min(),
            "max_area": areas.max(),
            "mean_area": areas.mean(),
            "median_area": areas.median(),
            "std_area": areas.std(),
        }

    return stats


def attach_shape_metrics(layer):
    """Calculate shape metrics for objects in a layer.

    Parameters:
    -----------
    layer : Layer
        Layer to calculate metrics for

    Returns:
    --------
    metrics : dict
        Dictionary with shape metrics
    """
    if layer.objects is None:
        return {}

    layer.objects["perimeter"] = layer.objects.geometry.length

    if "area_units" not in layer.objects.columns:
        layer.objects["area_units"] = layer.objects.geometry.area

    layer.objects["shape_index"] = (
        (layer.objects["perimeter"] / (2 * np.sqrt(np.pi * layer.objects["area_units"])))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    layer.objects["compactness"] = (
        (4 * np.pi * layer.objects["area_units"] / (layer.objects["perimeter"] ** 2)).replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    metrics = {
        "shape_index": {
            "mean": layer.objects["shape_index"].mean(),
            "min": layer.objects["shape_index"].min(),
            "max": layer.objects["shape_index"].max(),
            "std": layer.objects["shape_index"].std(),
        },
        "compactness": {
            "mean": layer.objects["compactness"].mean(),
            "min": layer.objects["compactness"].min(),
            "max": layer.objects["compactness"].max(),
            "std": layer.objects["compactness"].std(),
        },
    }

    return metrics


def attach_neighbor_stats(layer):
    """Calculate neighborhood statistics for objects in a layer.

    Parameters:
    -----------
    layer : Layer
        Layer to calculate statistics for

    Returns:
    --------
    stats : dict
        Dictionary with neighborhood statistics
    """
    if layer.objects is None:
        return {}

    neighbors = {}
    neighbor_counts = []

    for idx, obj in layer.objects.iterrows():
        touching = layer.objects[layer.objects.index != idx].intersects(obj.geometry)
        neighbor_ids = layer.objects[touching].index.tolist()

        neighbors[idx] = neighbor_ids
        neighbor_counts.append(len(neighbor_ids))

    layer.objects["neighbor_count"] = neighbor_counts

    stats = {
        "neighbor_count": {
            "mean": np.mean(neighbor_counts),
            "min": np.min(neighbor_counts),
            "max": np.max(neighbor_counts),
            "std": np.std(neighbor_counts),
        }
    }

    return stats
