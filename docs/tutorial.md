# Getting Started

This guide will help you get started with NickySpatial for object-based image analysis.

## Installation

Install NickySpatial using pip:

```bash
pip install nickyspatial
```

For deep learning support, install with CNN dependencies:

```bash
pip install nickyspatial[cnn]
```

## Basic Workflow

NickySpatial follows a typical object-based image analysis workflow:

1. Load raster data
2. Perform segmentation
3. Calculate statistics
4. Apply classification
5. Visualize results

## Quick Example

```python
import nickyspatial as ns

# 1. Load raster data
layer = ns.read_raster("satellite_image.tif")

# 2. Perform segmentation
segmenter = ns.SlicSegmentation(scale=20, compactness=0.5)
segmented_layer = segmenter.execute(layer.raster, layer.transform, layer.crs)

# 3. Calculate statistics
ns.attach_basic_stats(segmented_layer, "mean_intensity")
ns.attach_area_stats(segmented_layer)
ns.attach_spectral_indices(segmented_layer, indices=["ndvi"])

# 4. Visualize
ns.plot_layer_interactive(segmented_layer, "ndvi")
```

## Core Concepts

### Layers
Layers are the fundamental data structure in NickySpatial, containing:
- Spatial objects (segments) as geometries
- Attribute data for each object
- Raster representation
- Spatial reference information

### Segmentation
Segmentation divides an image into meaningful objects using algorithms like SLIC.

### Statistics
Various statistical measures can be calculated for each object:
- Basic statistics (mean, std, min, max)
- Spatial statistics (area, perimeter, shape metrics)
- Spectral statistics (indices like NDVI, NDWI)

### Classification
Objects can be classified using:
- Rule-based approaches
- Machine learning algorithms
- Deep learning models

## Next Steps

- Explore the [Features](features.md) page for comprehensive functionality
- Check out detailed [Examples](examples/01_basic_segmentation_workflow.ipynb)
- Review the [API Reference](reference.md) for complete documentation
