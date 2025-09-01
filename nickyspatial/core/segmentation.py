# -*- coding: utf-8 -*-
"""Implements segmentation algorithms to partition images into meaningful region objects.

The functions here might apply clustering or region-growing techniques, aiding object-based remote sensing analysis.
This module includes the SlicSegmentation class, which implements a bottom-up region-growing algorithm
Algorithms:
- SlicSegmentation: Bottom-up region-growing algorithm
- FelzenszwalbSegmentation: Graph-based segmentation
- WatershedSegmentation: Topographic watershed algorithm
- RegularGridSegmentation: Simple grid-based segmentation
"""

import time
import warnings

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import Polygon
from skimage import filters, segmentation, util

from .layer import Layer


class BaseSegmentation:
    """Base class for segmentation algorithms."""

    def __init__(self):
        """Initialize the base segmentation class."""
        pass

    def _validate_inputs(self, image_data, transform, crs, raster_path=None):
        """Validate common inputs across all segmentation algorithms."""
        # If raster_path is provided, other params are optional (will be auto-extracted)
        if raster_path is None:
            if image_data is None:
                raise ValueError("Either image_data or raster_path must be provided")
            if transform is None:
                raise ValueError("transform cannot be None when image_data is provided")
            if crs is None:
                raise ValueError("crs cannot be None when image_data is provided")

        if image_data is not None:
            if len(image_data.shape) != 3:
                raise ValueError("image_data must be 3D array (bands, height, width)")
            if image_data.size == 0:
                raise ValueError("image_data cannot be empty")

    def _prepare_inputs(self, image_data=None, transform=None, crs=None, raster_path=None, target_crs=None):
        """Prepare and validate inputs for segmentation algorithms.

        Handles automatic raster loading and CRS reprojection.

        Returns:
        --------
        tuple : (image_data, transform, crs)
            Processed inputs ready for segmentation
        """
        if raster_path is not None:
            from ..io.raster import read_raster

            image_data, transform, crs = read_raster(raster_path)

        if target_crs is not None and target_crs != crs:
            from rasterio.warp import reproject, Resampling, calculate_default_transform
            from rasterio.crs import CRS
            from rasterio.transform import array_bounds

            if isinstance(target_crs, str):
                target_crs = CRS.from_string(target_crs)

            new_transform, new_width, new_height = calculate_default_transform(
                crs,
                target_crs,
                image_data.shape[2],
                image_data.shape[1],
                *array_bounds(image_data.shape[1], image_data.shape[2], transform),
            )

            reprojected_data = np.zeros((image_data.shape[0], new_height, new_width), dtype=image_data.dtype)

            for band_idx in range(image_data.shape[0]):
                reproject(
                    source=image_data[band_idx],
                    destination=reprojected_data[band_idx],
                    src_transform=transform,
                    src_crs=crs,
                    dst_transform=new_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

            image_data = reprojected_data
            transform = new_transform
            crs = target_crs

        self._validate_inputs(image_data, transform, crs, raster_path)

        return image_data, transform, crs

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

    def _normalize_bands(self, image_data):
        """Normalize image bands to [0, 1] range.

        Parameters:
        -----------
        image_data : numpy.ndarray
            Array with raster data values (bands, height, width)

        Returns:
        --------
        numpy.ndarray
            Normalized multichannel image (height, width, channels)
        """
        num_bands, height, width = image_data.shape
        normalized_bands = []

        for i in range(num_bands):
            band = image_data[i]

            if band.max() == band.min():
                normalized_bands.append(np.zeros_like(band))
                continue

            norm_band = (band - band.min()) / (band.max() - band.min())
            normalized_bands.append(norm_band)

        return np.stack(normalized_bands, axis=-1)


class SlicSegmentation(BaseSegmentation):
    """Implementation of Multiresolution segmentation algorithm.

    This algorithm segments an image using a bottom-up region-growing approach
    that optimizes the homogeneity of pixel values within segments while
    considering shape compactness.
    """

    def __init__(self, scale=15, compactness=0.6, **kwargs):
        """Initialize the segmentation algorithm.

        Parameters:
        -----------
        scale : float
            Scale parameter that influences the size of the segments.
            Higher values create larger segments.
        shape : float, range [0, 1]
            Weight of shape criterion vs. color criterion.
            Higher values give more weight to shape.
        compactness : float, range [0, 1]
            Weight of compactness criterion vs. smoothness criterion.
            Higher values create more compact segments.
        **kwargs : dict
            Additional parameters passed to skimage.segmentation.slic()
        """
        super().__init__()
        self.scale = scale
        self.compactness = compactness
        self.slic_kwargs = kwargs

    def execute(
        self, image_data=None, transform=None, crs=None, raster_path=None, target_crs=None, layer_manager=None, layer_name=None
    ):
        """Perform segmentation and create a layer with the results.

        Parameters:
        -----------
        image_data : numpy.ndarray, optional
            Array with raster data values (bands, height, width).
            If not provided, must specify raster_path.
        transform : affine.Affine, optional
            Affine transformation for the raster.
            If not provided, will be extracted from raster_path.
        crs : rasterio.crs.CRS, optional
            Coordinate reference system.
            If not provided, will be extracted from raster_path.
        raster_path : str, optional
            Path to raster file. If provided, image_data, transform, and crs
            will be extracted automatically.
        target_crs : str or rasterio.crs.CRS, optional
            Target CRS for the output. If different from input CRS,
            automatic reprojection will be performed.
        layer_manager : LayerManager, optional
            Layer manager to add the result layer to
        layer_name : str, optional
            Name for the result layer

        Returns:
        --------
        layer : Layer
            Layer containing the segmentation results
        """
        start_time = time.time()

        image_data, transform, crs = self._prepare_inputs(image_data, transform, crs, raster_path, target_crs)
        num_bands, height, width = image_data.shape

        multichannel_image = self._normalize_bands(image_data)

        n_segments = int(width * height / (self.scale * self.scale))
        print(f"Number of segments: {n_segments}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments = segmentation.slic(
                multichannel_image,
                n_segments=n_segments,
                compactness=self.compactness,
                sigma=1.0,
                start_label=1,
                channel_axis=-1,
                **self.slic_kwargs,
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
            "execution_time_seconds": round(time.time() - start_time, 3),
        }

        if self.slic_kwargs:
            layer.metadata["kwargs"] = self.slic_kwargs

        segment_objects = self._create_segment_objects(segments, transform, crs)
        layer.objects = segment_objects

        bands = [f"band_{i + 1}" for i in range(num_bands)]
        self._calculate_statistics(layer, image_data, bands)

        if layer_manager:
            layer_manager.add_layer(layer)

        return layer


class FelzenszwalbSegmentation(BaseSegmentation):
    """Implementation of Felzenszwalb's efficient graph-based segmentation.

    This algorithm builds a graph of pixel similarities and uses a minimum
    spanning tree approach to segment the image into regions of similar
    characteristics.
    """

    def __init__(self, scale=100, sigma=0.5, min_size=50, **kwargs):
        """Initialize the Felzenszwalb segmentation algorithm.

        Parameters:
        -----------
        scale : float
            Free parameter that influences the size of the segments.
            Higher values create larger segments.
        sigma : float
            Width (standard deviation) of Gaussian kernel for pre-processing. Higher values
            give more smoothing.
        min_size : int
            Minimum component size. Smaller components are merged with
            neighboring larger components.
        **kwargs : dict
            Additional parameters passed to skimage.segmentation.felzenszwalb()
        """
        super().__init__()
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.fz_kwargs = kwargs

    def execute(
        self, image_data=None, transform=None, crs=None, raster_path=None, target_crs=None, layer_manager=None, layer_name=None
    ):
        """Perform Felzenszwalb segmentation and create a layer with the results."""
        start_time = time.time()

        image_data, transform, crs = self._prepare_inputs(image_data, transform, crs, raster_path, target_crs)
        num_bands, height, width = image_data.shape
        multichannel_image = self._normalize_bands(image_data)

        print(f"Felzenszwalb - Processing {height}x{width} image with {num_bands} bands")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments = segmentation.felzenszwalb(
                multichannel_image, scale=self.scale, sigma=self.sigma, min_size=self.min_size, channel_axis=-1, **self.fz_kwargs
            )

        segments = segments + 1

        if not layer_name:
            layer_name = f"Felzenszwalb_scale{self.scale}_sigma{self.sigma}"

        layer = Layer(name=layer_name, layer_type="segmentation")
        layer.raster = segments
        layer.transform = transform
        layer.crs = crs
        layer.metadata = {
            "algorithm": "Felzenszwalb",
            "scale": self.scale,
            "sigma": self.sigma,
            "min_size": self.min_size,
            "num_segments_actual": len(np.unique(segments)),
            "execution_time_seconds": round(time.time() - start_time, 3),
        }

        if self.fz_kwargs:
            layer.metadata["kwargs"] = self.fz_kwargs

        segment_objects = self._create_segment_objects(segments, transform, crs)
        layer.objects = segment_objects

        bands = [f"band_{i + 1}" for i in range(num_bands)]
        self._calculate_statistics(layer, image_data, bands)

        if layer_manager:
            layer_manager.add_layer(layer)

        return layer


class WatershedSegmentation(BaseSegmentation):
    """Implementation of watershed segmentation algorithm using regular grid seeding.

    The watershed algorithm treats the image as a topographic surface where
    pixel intensities represent elevation. It finds watershed lines that
    separate different catchment basins, effectively segmenting the image
    into distinct regions.
    """

    def __init__(self, n_points=468, compactness=0, watershed_line=False, preprocessing="sobel", mask=None, **kwargs):
        """Initialize the watershed segmentation algorithm.

        Parameters:
        -----------
        n_points : int
            Number of seed points to generate using regular grid.
            Higher values create more segments.
        compactness : float, optional
            Use compact watershed with given compactness parameter.
            Higher values result in more regularly-shaped watershed basins.
        watershed_line : bool, optional
            If watershed_line is True, a one-pixel wide line separates the regions
            obtained by the watershed algorithm. The line has the label 0.
        preprocessing : str, optional
            Method for preprocessing the image before watershed segmentation.
            Options: 'sobel', 'prewitt', 'scharr'
        mask : ndarray of bools or 0s and 1s, optional
            Array of same shape as image. Only points at which mask == True
            will be labeled.
        **kwargs : dict
            Additional parameters passed to watershed function
        """
        super().__init__()
        self.n_points = n_points
        self.compactness = compactness
        self.watershed_line = watershed_line
        self.preprocessing = preprocessing
        self.mask = mask
        self.watershed_kwargs = kwargs

    def _preprocess_image(self, image):
        """Apply edge detection preprocessing to the image before watershed segmentation."""
        if len(image.shape) == 3:
            image = np.mean(image, axis=-1)

        if self.preprocessing == "sobel":
            processed = filters.sobel(image)
        elif self.preprocessing == "prewitt":
            processed = filters.prewitt(image)
        elif self.preprocessing == "scharr":
            processed = filters.scharr(image)
        else:
            raise ValueError(f"Unknown preprocessing method: {self.preprocessing}. Available options: 'sobel', 'prewitt', 'scharr'")

        return processed

    def _generate_seeds(self, image_shape, n_points):
        """Generate seed points using regular grid approach."""
        grid = util.regular_grid(image_shape, n_points=n_points)

        seeds = np.zeros(image_shape, dtype=int)
        seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

        return seeds

    def execute(
        self, image_data=None, transform=None, crs=None, raster_path=None, target_crs=None, layer_manager=None, layer_name=None
    ):
        """Perform watershed segmentation and create a layer with the results."""
        start_time = time.time()

        image_data, transform, crs = self._prepare_inputs(image_data, transform, crs, raster_path, target_crs)
        num_bands, height, width = image_data.shape

        multichannel_image = self._normalize_bands(image_data)

        print(f"Watershed - Processing {height}x{width} image with {num_bands} bands")
        print(f"Using {self.n_points} seed points with {self.preprocessing} edge detection")

        edges = self._preprocess_image(multichannel_image)

        seeds = self._generate_seeds((height, width), self.n_points)
        actual_seeds = len(np.unique(seeds)) - 1
        print(f"Generated {actual_seeds} seed points")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments = segmentation.watershed(
                edges,
                markers=seeds,
                mask=self.mask,
                compactness=self.compactness,
                watershed_line=self.watershed_line,
                **self.watershed_kwargs,
            )

        # Handle watershed lines if present
        if self.watershed_line and 0 in np.unique(segments):
            print("Watershed lines preserved (label 0)")
        elif np.min(segments) == 0:
            # Relabel to start from 1 if no watershed lines
            segments = segments + 1

        if not layer_name:
            layer_name = f"Watershed_{self.preprocessing}_n{self.n_points}_comp{self.compactness}"

        # Create layer
        layer = Layer(name=layer_name, layer_type="segmentation")
        layer.raster = segments
        layer.transform = transform
        layer.crs = crs
        layer.metadata = {
            "algorithm": "Watershed",
            "preprocessing": self.preprocessing,
            "n_points": self.n_points,
            "compactness": self.compactness,
            "watershed_line": self.watershed_line,
            "num_seeds": actual_seeds,
            "num_segments_actual": len(np.unique(segments)),
            "execution_time_seconds": round(time.time() - start_time, 3),
        }

        if self.watershed_kwargs:
            layer.metadata["kwargs"] = self.watershed_kwargs

        # Create segment objects
        segment_objects = self._create_segment_objects(segments, transform, crs)
        layer.objects = segment_objects

        # Calculate statistics
        bands = [f"band_{i + 1}" for i in range(num_bands)]
        self._calculate_statistics(layer, image_data, bands)

        if layer_manager:
            layer_manager.add_layer(layer)

        return layer


class RegularGridSegmentation(BaseSegmentation):
    """Implementation of regular grid segmentation algorithm.

    This algorithm divides the image into regular rectangular segments
    of specified dimensions, creating a uniform grid pattern across
    the entire image.
    """

    def __init__(self, grid_size=(10, 10), overlap=0, boundary_handling="pad"):
        """Initialize the regular grid segmentation algorithm.

        Parameters:
        -----------
        grid_size : tuple of int
            Size of each grid cell (height, width) in pixels.
            For example, (10, 10) creates 10x10 pixel squares.
        overlap : int, optional
            Number of pixels to overlap between adjacent segments.
            Default is 0 (no overlap).
        boundary_handling : str, optional
            How to handle boundary segments that don't fit exactly:
            - 'pad': Pad the image to fit complete grid cells
            - 'truncate': Allow partial segments at boundaries
            - 'stretch': Stretch boundary segments to fill remaining space
        """
        super().__init__()
        self.grid_size = grid_size
        self.overlap = overlap
        self.boundary_handling = boundary_handling

    def _calculate_grid_dimensions(self, image_shape):
        """Calculate grid dimensions and segment layout."""
        height, width = image_shape
        grid_h, grid_w = self.grid_size

        # Calculate effective grid size considering overlap
        effective_h = grid_h - self.overlap
        effective_w = grid_w - self.overlap

        if self.boundary_handling == "pad":
            # Calculate number of complete segments needed
            n_rows = int(np.ceil(height / effective_h))
            n_cols = int(np.ceil(width / effective_w))

            # Calculate padded dimensions
            padded_height = n_rows * effective_h + self.overlap
            padded_width = n_cols * effective_w + self.overlap

            pad_h = max(0, padded_height - height)
            pad_w = max(0, padded_width - width)

        elif self.boundary_handling == "truncate":
            # Allow partial segments
            n_rows = int(np.ceil(height / effective_h))
            n_cols = int(np.ceil(width / effective_w))

            padded_height = height
            padded_width = width
            pad_h = pad_w = 0

        else:  # stretch
            # Calculate based on fitting segments exactly
            n_rows = max(1, int(height / effective_h))
            n_cols = max(1, int(width / effective_w))

            padded_height = height
            padded_width = width
            pad_h = pad_w = 0

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "padded_height": padded_height,
            "padded_width": padded_width,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "effective_h": effective_h,
            "effective_w": effective_w,
        }

    def _create_grid_segments(self, image_shape, grid_info):
        """Create the segmentation array with regular grid."""
        height, width = image_shape
        segments = np.zeros((height, width), dtype=np.int32)

        segment_id = 1

        for row in range(grid_info["n_rows"]):
            for col in range(grid_info["n_cols"]):
                # Calculate segment boundaries
                if self.boundary_handling == "stretch" and (row == grid_info["n_rows"] - 1 or col == grid_info["n_cols"] - 1):
                    # Stretch last row/column to image boundary
                    start_row = row * grid_info["effective_h"]
                    start_col = col * grid_info["effective_w"]

                    if row == grid_info["n_rows"] - 1:
                        end_row = height
                    else:
                        end_row = min(start_row + self.grid_size[0], height)

                    if col == grid_info["n_cols"] - 1:
                        end_col = width
                    else:
                        end_col = min(start_col + self.grid_size[1], width)
                else:
                    # Regular grid cell
                    start_row = row * grid_info["effective_h"]
                    start_col = col * grid_info["effective_w"]
                    end_row = min(start_row + self.grid_size[0], height)
                    end_col = min(start_col + self.grid_size[1], width)

                # Skip if segment is too small (in truncate mode)
                if end_row <= start_row or end_col <= start_col:
                    continue

                # Assign segment ID
                segments[start_row:end_row, start_col:end_col] = segment_id
                segment_id += 1

        return segments

    def execute(
        self, image_data=None, transform=None, crs=None, raster_path=None, target_crs=None, layer_manager=None, layer_name=None
    ):
        """Perform regular grid segmentation and create a layer with the results."""
        start_time = time.time()

        image_data, transform, crs = self._prepare_inputs(image_data, transform, crs, raster_path, target_crs)
        num_bands, height, width = image_data.shape

        print(f"RegularGrid - Processing {height}x{width} image with {num_bands} bands")
        print(f"Grid size: {self.grid_size}, Overlap: {self.overlap}, Boundary: {self.boundary_handling}")

        grid_info = self._calculate_grid_dimensions((height, width))

        print(f"Creating {grid_info['n_rows']}x{grid_info['n_cols']} = {grid_info['n_rows'] * grid_info['n_cols']} segments")

        if self.boundary_handling == "pad" and (grid_info["pad_h"] > 0 or grid_info["pad_w"] > 0):
            padded_image = np.pad(image_data, ((0, 0), (0, grid_info["pad_h"]), (0, grid_info["pad_w"])), mode="reflect")
            working_shape = (grid_info["padded_height"], grid_info["padded_width"])
            print(f"Padded image to {working_shape} (added {grid_info['pad_h']}x{grid_info['pad_w']} pixels)")
        else:
            padded_image = image_data
            working_shape = (height, width)

        segments = self._create_grid_segments(working_shape, grid_info)

        # If we padded, crop back to original size
        if self.boundary_handling == "pad" and (grid_info["pad_h"] > 0 or grid_info["pad_w"] > 0):
            segments = segments[:height, :width]
            # Also crop the padded image data for statistics calculation
            padded_image = padded_image[:, :height, :width]

        # Ensure we're using the original image data for statistics
        working_image = image_data

        if not layer_name:
            layer_name = f"RegularGrid_{self.grid_size[0]}x{self.grid_size[1]}_overlap{self.overlap}"

        # Create layer
        layer = Layer(name=layer_name, layer_type="segmentation")
        layer.raster = segments
        layer.transform = transform
        layer.crs = crs
        layer.metadata = {
            "algorithm": "RegularGrid",
            "grid_size": self.grid_size,
            "overlap": self.overlap,
            "boundary_handling": self.boundary_handling,
            "n_rows": grid_info["n_rows"],
            "n_cols": grid_info["n_cols"],
            "num_segments_actual": len(np.unique(segments[segments > 0])),
            "image_shape": (height, width),
            "execution_time_seconds": round(time.time() - start_time, 3),
        }

        # Create segment objects
        segment_objects = self._create_segment_objects(segments, transform, crs)
        layer.objects = segment_objects

        # Calculate statistics
        bands = [f"band_{i + 1}" for i in range(num_bands)]
        self._calculate_statistics(layer, working_image, bands)

        if layer_manager:
            layer_manager.add_layer(layer)

        return layer
