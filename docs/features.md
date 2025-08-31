# Features Overview

NickySpatial provides object-based image analysis capabilities for remote sensing applications.

## Image Segmentation

### SLIC Segmentation
Transform pixel-based images into meaningful objects using the SLIC algorithm.

```python
import nickyspatial as ns

layer = ns.read_raster("satellite_image.tif")
segmenter = ns.SlicSegmentation(scale=20, compactness=0.5)
segmented_layer = segmenter.execute(layer.raster, layer.transform, layer.crs)
```

Parameters:
- `scale`: Controls the size of segments (higher values create larger segments)
- `compactness`: Balance between spatial and spectral similarity
- `sigma`: Gaussian smoothing parameter

## Statistical Analysis

### Basic Statistics
Calculate fundamental statistical measures for each object.

```python
ns.attach_basic_stats(layer, column="mean_intensity")
ns.attach_count(layer, class_column="classification")
ns.attach_class_distribution(layer)
```

Available metrics: mean, median, mode, standard deviation, variance, min, max values, object counts and distributions.

### Spatial Statistics
Analyze geometric and spatial properties of objects.

```python
ns.attach_area_stats(layer)
ns.attach_shape_metrics(layer)
ns.attach_neighbor_stats(layer)
```

Spatial metrics: area, perimeter, shape complexity, neighbor relationships, spatial distribution patterns.

### Spectral Statistics
Calculate spectral indices and band-specific statistics.

```python
ns.attach_spectral_indices(layer, indices=["ndvi", "ndwi", "evi"])
ns.attach_ndvi(layer, red_band=3, nir_band=4)
```

Spectral indices: NDVI, NDWI, EVI, custom band ratios.

## Classification Methods

### Traditional Machine Learning
Use scikit-learn algorithms for object classification.

```python
classifier = ns.SupervisedClassifier(
    name="RF Classification", 
    classifier_type="Random Forest", 
    classifier_params={"n_estimators": 100}
)
classification_layer, accuracy, feature_importances = classifier.execute(
    layer, 
    samples=samples, 
    layer_name="classified_layer",
    features=["mean_intensity", "area_units", "ndvi"]
)
```

Supported algorithms: Random Forest, SVM, K-Nearest Neighbors.

### Deep Learning
Leverage TensorFlow/Keras for advanced classification.

```python
cnn_classifier = ns.SupervisedClassifierDL(
    name="CNN Classification",
    classifier_type="cnn",
    classifier_params={"epochs": 50, "batch_size": 32}
)
classification_layer, model_history, eval_result, count_dict, invalid_patches = cnn_classifier.execute(
    layer,
    samples=samples,
    image_data=image_data,
    layer_name="cnn_classified"
)
```

Features: Convolutional Neural Networks, custom model architectures, patch-based training, GPU acceleration.

## Rule-Based Classification

### Rule Engine
Apply custom logic for object classification based on multiple criteria.

```python
ruleset = ns.RuleSet(name="Land_Cover")
ruleset.add_rule(name="water", condition="ndwi > 0.3 & area_units > 500")
ruleset.add_rule(name="vegetation", condition="ndvi > 0.4 & mean_green < 100")
ruleset.add_rule(name="urban", condition="mean_intensity > 150 & compactness < 0.8")
classified_layer = ruleset.execute(layer, result_field="classification")
```

### Spatial Rules
Define relationships between objects.

```python
enclosed_rules = ns.EnclosedByRuleSet()
touched_rules = ns.TouchedByRuleSet()
merge_rules = ns.MergeRuleSet()

spatial_result = enclosed_rules.execute(
    layer, 
    class_column_name="classification",
    class_value_a="building", 
    class_value_b="road",
    new_class_name="enclosed_building"
)
```

Rule types: attribute-based rules, spatial relationship rules, hierarchical classification, conditional logic.

## Filtering and Processing

### Spatial Filters
Refine object boundaries and merge similar objects.

```python
ns.merge_small_segments(layer, min_size=50)
ns.smooth_boundaries(layer, iterations=3)
ns.select_by_area(layer, min_area=100, max_area=10000)
```

### Spectral Filters
Enhance spectral characteristics and apply band operations.

```python
ns.enhance_contrast(layer, method="histogram_equalization")
ns.spectral_filter(layer, bands=[1, 2, 3], operation="mean")
```

## Visualization

### Interactive Maps
Create dynamic visualizations with multiple backends.

```python
ns.plot_layer_interactive(layer, color_column="classification")
ns.plot_layer_interactive_plotly(layer, "ndvi")
ns.plot_comparison(layer1, layer2, split_view=True)
```

### Statistical Charts
Visualize data distributions and relationships.

```python
ns.plot_histogram(layer, "area_units")
ns.plot_statistics(layer, metrics=["mean", "std", "area"])
ns.plot_training_history(cnn_classifier.history)
```

Visualization options: Folium-based interactive maps, Plotly interactive charts, Matplotlib static plots, classification comparisons.

## Data I/O

### Raster Operations
Work with various raster formats and coordinate systems.

```python
layer = ns.read_raster("multispectral.tif")
ns.write_raster(processed_layer, "results.tif")
raster_output = ns.layer_to_raster(segmented_layer)
```

### Vector Operations
Handle vector data and conversion between formats.

```python
vector_layer = ns.read_vector("boundaries.shp")
ns.write_vector(classified_layer, "classification.geojson")
vector_result = ns.layer_to_vector(layer)
```

Supported formats:
- Raster: GeoTIFF, JPEG2000, NetCDF, HDF
- Vector: Shapefile, GeoJSON, KML, GeoPackage

## Layer Management

### Layer Class
Central data structure for managing spatial objects.

```python
layer_manager = ns.LayerManager()
layer_manager.add_layer("segments", segmented_layer)
layer_manager.add_layer("classified", classified_layer)

print(layer.objects.head())
print(layer.statistics)
print(layer.metadata)
```

### Data Integration
Combine multiple data sources and coordinate systems.

```python
combined_layer = ns.Layer.from_multiple_sources([
    "optical_data.tif",
    "radar_data.tif",
    "elevation.tif"
])
```

## Performance Features

- Memory efficient: optimized for large raster datasets
- Parallel processing: multi-core support for computationally intensive operations
- Lazy loading: load data on-demand to manage memory usage
- Caching: intelligent caching for repeated operations

## Integration

NickySpatial integrates with the Python geospatial ecosystem:

- GeoPandas: vector data manipulation
- Rasterio: raster data I/O
- Scikit-image: image processing algorithms
- Scikit-learn: machine learning algorithms
- TensorFlow/Keras: deep learning capabilities
- Folium: interactive web mapping
- Plotly: interactive visualizations
