# -*- coding: utf-8 -*-
"""Implements segmentation algorithms to partition images into meaningful region objects."""

import warnings

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import Polygon
from skimage import segmentation

from .layer import Layer


class SlicSegmentation:
    """Implementation of Multiresolution segmentation algorithm."""

    def __init__(self, scale=15, compactness=0.6):
        """Initialize SLIC with scale (segment size) and compactness (shape regularity)."""
        self.scale = scale
        self.compactness = compactness

    def execute(self, image_data, transform, crs, layer_manager=None, layer_name=None):
        """Apply SLIC segmentation: normalize bands → stack → segment → vectorize."""
        num_bands, height, width = image_data.shape

        normalized_bands = []
        for i in range(num_bands):
            band = image_data[i]
            if band.max() == band.min():
                normalized_bands.append(np.zeros_like(band))
                continue
            norm_band = (band - band.min()) / (band.max() - band.min())
            normalized_bands.append(norm_band)

        multichannel_image = np.stack(normalized_bands, axis=-1)
        n_segments = int(width * height / (self.scale * self.scale))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments = segmentation.slic(
                multichannel_image,
                n_segments=n_segments,
                compactness=self.compactness,
                sigma=1.0,
                start_label=1,
                channel_axis=-1,
            )

        if not layer_name:
            layer_name = f"Segmentation_scale{self.scale}_comp{self.compactness}"

        layer = Layer(name=layer_name, layer_type="segmentation")
        layer.raster = segments
        layer.transform = transform
        layer.crs = crs
        layer.metadata = {
            "scale": self.scale,
            "compactness": self.compactness,
            "n_segments": n_segments,
            "num_segments_actual": len(np.unique(segments)),
        }

        segment_objects = self._create_segment_objects(segments, transform, crs)
        layer.objects = segment_objects

        bands = [f"band_{i + 1}" for i in range(num_bands)]
        self._calculate_statistics(layer, image_data, bands)

        if layer_manager:
            layer_manager.add_layer(layer)

        return layer

    def _create_segment_objects(self, segments, transform, crs):
        segment_ids = np.unique(segments)
        geometries = []
        properties = []

        for segment_id in segment_ids:
            mask = segments == segment_id

            if not np.any(mask):
                continue

            shapes = rasterio.features.shapes(mask.astype(np.int16), mask=mask, transform=transform)

            segment_polygons = []
            for geom, val in shapes:
                if val == 1:
                    try:
                        polygon = Polygon(geom["coordinates"][0])
                        if polygon.is_valid:
                            segment_polygons.append(polygon)
                    except Exception:
                        continue

            if not segment_polygons:
                continue

            largest_polygon = max(segment_polygons, key=lambda p: p.area)

            area_pixels = np.sum(mask)

            pixel_width = abs(transform.a)
            pixel_height = abs(transform.e)
            area_units = area_pixels * pixel_width * pixel_height

            prop = {
                "segment_id": int(segment_id),
                "area_pixels": int(area_pixels),
                "area_units": float(area_units),
            }

            geometries.append(largest_polygon)
            properties.append(prop)

        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
        return gdf

    def _calculate_statistics(self, layer, image_data, bands):
        """Calculate statistics for segments based on image data.

        Parameters:
        -----------
        layer : Layer
            Layer containing segments
        image_data : numpy.ndarray
            Array with raster data values (bands, height, width)
        bands : list of str
            Names of the bands
        """
        segments = layer.raster
        segment_objects = layer.objects

        segment_ids = segment_objects["segment_id"].values

        for i, band_name in enumerate(bands):
            if i >= image_data.shape[0]:
                break

            band_data = image_data[i]

            for segment_id in segment_ids:
                mask = segments == segment_id

                if segment_id not in segment_objects["segment_id"].values:
                    continue

                segment_pixels = band_data[mask]

                if len(segment_pixels) == 0:
                    continue

                mean_val = float(np.mean(segment_pixels))
                std_val = float(np.std(segment_pixels))
                min_val = float(np.min(segment_pixels))
                max_val = float(np.max(segment_pixels))
                median_val = float(np.median(segment_pixels))

                idx = segment_objects.index[segment_objects["segment_id"] == segment_id].tolist()[0]
                segment_objects.at[idx, f"{band_name}_mean"] = mean_val
                segment_objects.at[idx, f"{band_name}_std"] = std_val
                segment_objects.at[idx, f"{band_name}_min"] = min_val
                segment_objects.at[idx, f"{band_name}_max"] = max_val
                segment_objects.at[idx, f"{band_name}_median"] = median_val
