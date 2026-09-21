"""Basic statistics for layers in NickySpatial."""

import numpy as np


def attach_basic_stats(layer, column, prefix=None):
    """Attach basic statistics for a column to a layer.

    Parameters:
    -----------
    layer : Layer
        Layer to attach statistics to
    column : str
        Column to calculate statistics for
    prefix : str, optional
        Prefix for result names

    Returns:
    --------
    stats : dict
        Dictionary with calculated statistics
    """
    if layer.objects is None or column not in layer.objects.columns:
        raise ValueError(f"Column '{column}' not found in layer objects")

    prefix = f"{prefix}_" if prefix else ""

    values = layer.objects[column]
    stats = {
        f"{prefix}min": values.min(),
        f"{prefix}max": values.max(),
        f"{prefix}mean": values.mean(),
        f"{prefix}median": values.median(),
        f"{prefix}std": values.std(),
        f"{prefix}sum": values.sum(),
        f"{prefix}count": len(values),
    }

    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        stats[f"{prefix}percentile_{p}"] = np.percentile(values, p)

    return stats


def attach_count(layer, class_column="classification", class_value=None):
    """Count objects in a layer, optionally filtered by class.

    Parameters:
    -----------
    layer : Layer
        Layer to count objects in
    class_column : str
        Column containing class values
    class_value : str, optional
        Class value to filter by

    Returns:
    --------
    count : int
        Number of objects
    """
    if layer.objects is None:
        return 0

    if class_value is not None and class_column in layer.objects.columns:
        count = layer.objects[layer.objects[class_column] == class_value].shape[0]
    else:
        count = layer.objects.shape[0]

    return count


def attach_class_distribution(layer, class_column="classification"):
    """Calculate the distribution of classes in a layer.

    Parameters:
    -----------
    layer : Layer
        Layer to analyze
    class_column : str
        Column containing class values

    Returns:
    --------
    distribution : dict
        Dictionary with class counts and percentages
    """
    if layer.objects is None or class_column not in layer.objects.columns:
        return {}

    class_counts = layer.objects[class_column].value_counts()
    total_count = len(layer.objects)
    class_percentages = (class_counts / total_count * 100).round(2)

    distribution = {
        "counts": class_counts.to_dict(),
        "percentages": class_percentages.to_dict(),
        "total": total_count,
    }

    return distribution
