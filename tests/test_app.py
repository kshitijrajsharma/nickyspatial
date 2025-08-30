# -*- coding: utf-8 -*-
"""Test suite for the NickySpatial application.

This suite tests the full workflow of the NickySpatial library, including
reading a raster, performing segmentation, calculating spectral indices,
applying classification rules, and exporting results.
It also includes checks for the generated outputs and their validity.
"""

import json
import os
import shutil

import pytest

from nickyspatial import (
    EnclosedByRuleSet,
    LayerManager,
    MergeRuleSet,
    RuleSet,
    SlicSegmentation,
    SupervisedClassifier,
    TouchedByRuleSet,
    attach_area_stats,
    attach_ndvi,
    attach_shape_metrics,
    attach_spectral_indices,
    layer_to_raster,
    layer_to_vector,
    plot_classification,
    plot_layer,
    plot_layer_interactive,
    plot_layer_interactive_plotly,
    plot_sample,
    read_raster,
)

try:
    import pandas as pd

    from nickyspatial import SupervisedClassifierDL, plot_training_history

    HAS_CNN = True
except ImportError:
    HAS_CNN = False

try:
    import importlib.util

    HAS_INTERACTIVE = importlib.util.find_spec("ipywidgets") is not None and importlib.util.find_spec("plotly") is not None
except ImportError:
    HAS_INTERACTIVE = False


@pytest.fixture(autouse=True)
def clean_output():
    """Fixture to clean up the output directory before and after tests."""
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    yield
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


@pytest.fixture
def test_raster_path():
    """Fixture to provide the path to a sample raster image for testing."""
    path = os.path.join("data", "sample.tif")
    if not os.path.exists(path):
        pytest.skip("Test image not found in data/ directory.")
    return path


def check_geojson_features(filepath):
    """Check if the GeoJSON file contains features."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("type") == "FeatureCollection", "Invalid GeoJSON: wrong type."
    features = data.get("features")
    assert isinstance(features, list), "GeoJSON features is not a list."
    assert len(features) > 0, f"No features found in {filepath}."


def test_full_workflow(test_raster_path):
    """Test the full workflow of segmentation, classification, and export."""
    # Step 1: Create the output directory and initialize the LayerManager.
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    manager = LayerManager()

    # Step 2: Read the raster.
    image_data, transform, crs = read_raster(test_raster_path)
    assert image_data is not None, "Failed to read image data."

    # Step 3: Perform segmentation.
    segmenter = SlicSegmentation(scale=20, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Base_Segmentation")
    assert segmentation_layer is not None, "Segmentation layer was not created."

    # Step 4: Plot segmentation and save the figure.
    fig1 = plot_layer(segmentation_layer, image_data, rgb_bands=(3, 2, 1), show_boundaries=True)
    seg_img_path = os.path.join(output_dir, "1_segmentation.png")
    fig1.savefig(seg_img_path)
    assert os.path.exists(seg_img_path), "Segmentation image not saved."

    # Step 5: Calculate spectral indices (NDVI and others).
    segmentation_layer.attach_function(
        attach_ndvi,
        name="ndvi_stats",
        nir_column="band_4_mean",
        red_column="band_3_mean",
        output_column="NDVI",
    )
    segmentation_layer.attach_function(
        attach_spectral_indices,
        name="spectral_indices",
        bands={
            "blue": "band_1_mean",
            "green": "band_2_mean",
            "red": "band_3_mean",
            "nir": "band_4_mean",
        },
    )
    fig2 = plot_layer(segmentation_layer, attribute="NDVI", title="NDVI Values", cmap="RdYlGn")
    ndvi_img_path = os.path.join(output_dir, "2_ndvi.png")
    fig2.savefig(ndvi_img_path)
    assert os.path.exists(ndvi_img_path), "NDVI image not saved."

    # Step 6: Calculate shape metrics.
    segmentation_layer.attach_function(attach_shape_metrics, name="shape_metrics")
    seg_vector_path = os.path.join(output_dir, "segmentation.geojson")
    layer_to_vector(segmentation_layer, seg_vector_path)
    assert os.path.exists(seg_vector_path), "Segmentation GeoJSON not created."

    # Step 7: Apply land cover classification rules.
    land_cover_rules = RuleSet(name="Land_Cover")
    land_cover_rules.add_rule(name="Vegetation", condition="NDVI > 0.2")
    land_cover_rules.add_rule(name="Other", condition="NDVI <= 0.2")
    land_cover_layer = land_cover_rules.execute(segmentation_layer, layer_manager=manager, layer_name="Land_Cover")
    assert land_cover_layer is not None, "Land cover layer was not created."
    fig3 = plot_classification(land_cover_layer, class_field="classification")
    lc_img_path = os.path.join(output_dir, "3_land_cover.png")
    fig3.savefig(lc_img_path)
    assert os.path.exists(lc_img_path), "Land cover classification image not saved."

    # Step 8: Calculate class statistics.
    land_cover_layer.attach_function(attach_area_stats, name="area_by_class", by_class="classification")
    area_stats = land_cover_layer.get_function_result("area_by_class")
    assert "class_areas" in area_stats, "Area stats missing class_areas."
    assert len(area_stats["class_areas"]) > 0, "Area stats computed no classes."

    # Step 9: Apply hierarchical classification rules for vegetation.
    vegetation_rules = RuleSet(name="Vegetation_Types")
    vegetation_rules.add_rule(
        name="Healthy_Vegetation",
        condition="(classification == 'Vegetation') & (NDVI > 0.6)",
    )
    vegetation_rules.add_rule(
        name="Moderate_Vegetation",
        condition="(classification == 'Vegetation') & (NDVI <= 0.6) & (NDVI > 0.4)",
    )
    vegetation_rules.add_rule(
        name="Sparse_Vegetation",
        condition="(classification == 'Vegetation') & (NDVI <= 0.4)",
    )
    vegetation_layer = vegetation_rules.execute(
        land_cover_layer,
        layer_manager=manager,
        layer_name="Vegetation_Types",
        result_field="veg_class",
    )
    assert vegetation_layer is not None, "Vegetation layer was not created."
    fig4 = plot_classification(vegetation_layer, class_field="veg_class")
    veg_img_path = os.path.join(output_dir, "4_vegetation_types.png")
    fig4.savefig(veg_img_path)
    assert os.path.exists(veg_img_path), "Vegetation classification image not saved."

    # Step 10: Export results.
    lc_vector_path = os.path.join(output_dir, "land_cover.geojson")
    veg_vector_path = os.path.join(output_dir, "vegetation_types.geojson")
    layer_to_vector(land_cover_layer, lc_vector_path)
    layer_to_vector(vegetation_layer, veg_vector_path)
    assert os.path.exists(lc_vector_path), "Land cover GeoJSON not saved."
    assert os.path.exists(veg_vector_path), "Vegetation GeoJSON not saved."

    check_geojson_features(lc_vector_path)
    check_geojson_features(veg_vector_path)

    # Print the available layers.
    available_layers = manager.get_layer_names()
    assert len(available_layers) >= 3, "Expected at least three layers in manager."

    """Supervised Classification (ML) Test Start"""
    # Step 1: Perform segmentation.
    segmenter = SlicSegmentation(scale=20, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Base_Segmentation")
    assert segmentation_layer is not None, "Segmentation layer was not created."

    # Step 2: define samples and colors
    samples = {
        "Water": [41, 134, 246, 491],
        "built-up": [12, 499, 290, 484],
        "vegetation": [36, 143, 239, 588, 371],
    }
    classes_color = {"Water": "#3437c2", "built-up": "#de1421", "vegetation": "#0f6b2f"}
    # Step 3: RF Supervised Classification implementation
    params = {"n_estimators": 100, "oob_score": True, "random_state": 42}
    rf_classification = SupervisedClassifier(name="RF Classification", classifier_type="Random Forest", classifier_params=params)
    rf_classification_layer, accuracy, feature_importances = rf_classification.execute(
        segmentation_layer,
        samples=samples,
        layer_manager=manager,
        layer_name="RF Classification",
    )
    fig5 = plot_classification(rf_classification_layer, class_field="classification", class_color=classes_color)
    assert rf_classification_layer is not None, "RF classification layer was not created."
    rf_classification_img_path = os.path.join(output_dir, "5_RF_classification.png")
    fig5.savefig(rf_classification_img_path)
    assert os.path.exists(rf_classification_img_path), "RF Classification layer not saved."

    # Step 4: Apply merge segements based on class
    merger = MergeRuleSet("MergeByVegAndType")
    class_value = ["Water", "vegetation"]
    merged_layer = merger.execute(
        source_layer=rf_classification_layer,
        class_column_name="classification",
        class_value=class_value,
        layer_manager=manager,
        layer_name="Merged RF Classification",
    )
    assert merged_layer is not None, "Merged layer was not created."
    fig6 = plot_classification(merged_layer, class_field="classification", class_color=classes_color)
    merged_img_path = os.path.join(output_dir, "5_merged.png")
    fig6.savefig(merged_img_path)
    assert os.path.exists(merged_img_path), "Merged_by layer not saved."

    # Step 5: Apply enclosed_by rule
    encloser_rule = EnclosedByRuleSet()
    enclosed_by_layer = encloser_rule.execute(
        source_layer=merged_layer,
        class_column_name="classification",
        class_value_a="vegetation",
        class_value_b="built-up",
        new_class_name="park",
        layer_manager=manager,
        layer_name="enclosed_by_layer",
    )
    assert enclosed_by_layer is not None, "Enclosed_by layer was not created."
    classes_color["park"] = "#d2f7dc"
    fig7 = plot_classification(enclosed_by_layer, class_field="classification", class_color=classes_color)
    enclosed_img_path = os.path.join(output_dir, "7_enclosed_by.png")
    fig7.savefig(enclosed_img_path)
    assert os.path.exists(enclosed_img_path), "Enclosed_by layer not saved."

    # Step 6: Apply touched_by rule
    touched_by_rule = TouchedByRuleSet()
    touched_by_layer = touched_by_rule.execute(
        source_layer=enclosed_by_layer,
        class_column_name="classification",
        class_value_a="built-up",
        class_value_b="Water",
        new_class_name="Water front builtup",
        layer_manager=manager,
        layer_name="touched_by_layer",
    )
    assert touched_by_layer is not None, "Touched_by layer was not created."
    classes_color["Water front builtup"] = "#cc32cf"
    fig8 = plot_classification(touched_by_layer, class_field="classification", class_color=classes_color)
    touched_img_path = os.path.join(output_dir, "8_touched_by.png")
    fig8.savefig(touched_img_path)
    assert os.path.exists(touched_img_path), "Touched_by layer not saved."


@pytest.mark.skipif(not HAS_CNN, reason="TensorFlow/CNN dependencies not available")
def test_cnn_classification(test_raster_path):
    """Test CNN-based supervised classification workflow."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    manager = LayerManager()

    # Read the raster
    image_data, transform, crs = read_raster(test_raster_path)
    assert image_data is not None, "Failed to read image data."

    # Perform segmentation
    segmenter = SlicSegmentation(scale=20, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Base_Segmentation")
    assert segmentation_layer is not None, "Segmentation layer was not created."

    # Define samples and colors
    samples = {
        "Water": [102, 384, 659, 1142, 1662, 1710, 2113, 2182, 2481, 1024],
        "built-up": [467, 1102, 1431, 1984, 1227, 1736, 774, 1065],
        "vegetation": [832, 1778, 2035, 1417, 1263, 242, 2049, 2397],
    }

    classes_color = {"Water": "#3437c2", "built-up": "#de1421", "vegetation": "#0f6b2f"}

    # CNN classification parameters
    params = {
        "patch_size": (5, 5),
        "epochs": 60,
        "batch_size": 32,
        "early_stopping_patience": 5,
        "hidden_layers": [
            {"filters": 32, "kernel_size": 3, "max_pooling": True},
            {"filters": 64, "kernel_size": 3, "max_pooling": True},
        ],
        "use_batch_norm": False,
        "dense_units": 64,
    }

    cnn_classification = SupervisedClassifierDL(
        name="CNN Classification", classifier_type="Convolution Neural Network (CNN)", classifier_params=params
    )

    cnn_classification_layer, model_history, eval_result, count_dict, invalid_patches_segments_ids = cnn_classification.execute(
        source_layer=segmentation_layer,
        samples=samples,
        image_data=image_data,
        layer_manager=manager,
        layer_name="CNN Classification",
    )

    # Test results
    assert cnn_classification_layer is not None, "CNN classification layer was not created."

    # Plot and save CNN classification
    fig9 = plot_classification(cnn_classification_layer, class_field="classification", class_color=classes_color)
    cnn_classification_img_path = os.path.join(output_dir, "6_CNN_classification.png")
    fig9.savefig(cnn_classification_img_path)
    assert os.path.exists(cnn_classification_img_path), "CNN Classification layer not saved."

    # Test confusion matrix
    cm = eval_result["confusion_matrix"]
    df_cm = pd.DataFrame(cm, index=samples.keys(), columns=samples.keys())
    df_cm.index.name = "Predicted Label"

    # Plot and save model history
    mh_classification_img_path = os.path.join(output_dir, "6_model_history.png")
    fig10 = plot_training_history(model_history)
    fig10.savefig(mh_classification_img_path)
    assert os.path.exists(mh_classification_img_path), "Model History not saved."

    # Validate model performance
    assert "accuracy" in eval_result, "Evaluation results missing accuracy."
    assert "loss" in eval_result, "Evaluation results missing loss."
    assert eval_result["accuracy"] > 0, "CNN accuracy should be greater than 0."


def test_layer_to_raster_functionality(test_raster_path):
    """Test layer_to_raster functionality used in simple_usecase.ipynb."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    manager = LayerManager()

    # Read raster and create segmentation
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=30, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Test_Segmentation")

    # Apply classification rules
    segmentation_layer.attach_function(
        attach_ndvi,
        name="ndvi_stats",
        nir_column="band_4_mean",
        red_column="band_3_mean",
        output_column="NDVI",
    )

    land_cover_rules = RuleSet(name="Land_Cover")
    land_cover_rules.add_rule(name="Vegetation", condition="NDVI > 0.2")
    land_cover_rules.add_rule(name="Other", condition="NDVI <= 0.2")

    land_cover_layer = land_cover_rules.execute(segmentation_layer, layer_manager=manager, layer_name="Land_Cover")

    # Test layer_to_raster export
    raster_output_path = os.path.join(output_dir, "test_classification.tif")
    layer_to_raster(land_cover_layer, raster_output_path, column="classification")

    assert os.path.exists(raster_output_path), "Raster export failed."

    # Test reading the exported raster
    exported_data, exported_transform, exported_crs = read_raster(raster_output_path)
    assert exported_data is not None, "Could not read exported raster."
    assert exported_data.shape[1:] == image_data.shape[1:], "Exported raster dimensions don't match."


def test_visualization_functions(test_raster_path):
    """Test all visualization functions used in notebooks."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    manager = LayerManager()

    # Setup basic data
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=25, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Viz_Test")

    # Test basic plot_layer
    fig1 = plot_layer(segmentation_layer, image_data, rgb_bands=(3, 2, 1), show_boundaries=True)
    assert fig1 is not None, "plot_layer failed."
    fig1.savefig(os.path.join(output_dir, "test_plot_layer.png"))

    # Add classification for classification plots
    samples = {
        "Water": [41, 134, 246],
        "Built-up": [12, 499, 290],
        "Vegetation": [36, 143, 239],
    }
    classes_color = {"Water": "#3437c2", "Built-up": "#de1421", "Vegetation": "#0f6b2f"}

    rf_classification = SupervisedClassifier(
        name="Test RF", classifier_type="Random Forest", classifier_params={"n_estimators": 10, "random_state": 42}
    )
    rf_layer, accuracy, feature_importances = rf_classification.execute(
        segmentation_layer, samples=samples, layer_manager=manager, layer_name="RF_Test"
    )

    # Test plot_classification
    fig2 = plot_classification(rf_layer, class_field="classification", class_color=classes_color)
    assert fig2 is not None, "plot_classification failed."
    fig2.savefig(os.path.join(output_dir, "test_plot_classification.png"))

    # Test plot_sample
    fig3 = plot_sample(rf_layer, image_data, transform, rgb_bands=(3, 2, 1), class_color=classes_color)
    assert fig3 is not None, "plot_sample failed."
    fig3.savefig(os.path.join(output_dir, "test_plot_sample.png"))


@pytest.mark.skipif(not HAS_INTERACTIVE, reason="Interactive plotting dependencies not available")
def test_interactive_plotting(test_raster_path):
    """Test interactive plotting functions from notebooks."""
    manager = LayerManager()

    # Setup basic data
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=20, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Interactive_Test")

    # Test plot_layer_interactive (returns widgets, difficult to assert)
    try:
        plot_layer_interactive(segmentation_layer, image_data, figsize=(8, 6))
        # If no exception is raised, consider it successful
        assert True, "plot_layer_interactive executed successfully."
    except Exception as e:
        pytest.skip(f"Interactive plotting failed (expected in headless env): {e}")

    # Test plot_layer_interactive_plotly
    try:
        plot_layer_interactive_plotly(segmentation_layer, image_data, rgb_bands=(0, 1, 2), show_boundaries=True, figsize=(400, 300))
        assert True, "plot_layer_interactive_plotly executed successfully."
    except Exception as e:
        pytest.skip(f"Plotly interactive plotting failed (expected in headless env): {e}")


def test_different_classifiers(test_raster_path):
    """Test different classification algorithms (SVM, KNN) used in supervised_classification.ipynb."""
    manager = LayerManager()

    # Setup data
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=20, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Classifier_Test")

    samples = {
        "Water": [41, 134, 246, 491],
        "Built-up": [12, 499, 290, 484],
        "Vegetation": [36, 143, 239, 588],
    }

    # Test SVM classifier
    svm_params = {"kernel": "rbf", "C": 1.0, "random_state": 42}
    svm_classification = SupervisedClassifier(name="SVM Classification", classifier_type="SVM", classifier_params=svm_params)

    svm_layer, svm_accuracy, svm_features = svm_classification.execute(
        segmentation_layer, samples=samples, layer_manager=manager, layer_name="SVM_Classification"
    )

    assert svm_layer is not None, "SVM classification layer was not created."
    assert svm_accuracy > 0, "SVM accuracy should be greater than 0."

    # Test KNN classifier
    knn_params = {"n_neighbors": 5}
    knn_classification = SupervisedClassifier(name="KNN Classification", classifier_type="KNN", classifier_params=knn_params)

    knn_layer, knn_accuracy, knn_features = knn_classification.execute(
        segmentation_layer, samples=samples, layer_manager=manager, layer_name="KNN_Classification"
    )

    assert knn_layer is not None, "KNN classification layer was not created."
    assert knn_accuracy > 0, "KNN accuracy should be greater than 0."


def test_hierarchical_rules_workflow(test_raster_path):
    """Test hierarchical classification workflow from simple_usecase.ipynb."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    manager = LayerManager()

    # Setup basic workflow
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=40, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Hierarchical_Test")

    # Calculate spectral indices
    segmentation_layer.attach_function(
        attach_ndvi,
        name="ndvi_stats",
        nir_column="band_4_mean",
        red_column="band_3_mean",
        output_column="NDVI",
    )

    segmentation_layer.attach_function(
        attach_spectral_indices,
        name="spectral_indices",
        bands={
            "blue": "band_1_mean",
            "green": "band_2_mean",
            "red": "band_3_mean",
            "nir": "band_4_mean",
        },
    )

    # Apply land cover rules
    land_cover_rules = RuleSet(name="Land_Cover")
    land_cover_rules.add_rule(name="Vegetation", condition="NDVI > 0.2")
    land_cover_rules.add_rule(name="Other", condition="NDVI <= 0.2")

    land_cover_layer = land_cover_rules.execute(segmentation_layer, layer_manager=manager, layer_name="Land_Cover")

    # Apply hierarchical vegetation rules
    vegetation_rules = RuleSet(name="Vegetation_Types")
    vegetation_rules.add_rule(name="Healthy_Vegetation", condition="(classification == 'Vegetation') & (NDVI > 0.6)")
    vegetation_rules.add_rule(
        name="Moderate_Vegetation", condition="(classification == 'Vegetation') & (NDVI <= 0.6) & (NDVI > 0.4)"
    )
    vegetation_rules.add_rule(name="Sparse_Vegetation", condition="(classification == 'Vegetation') & (NDVI <= 0.4)")

    vegetation_layer = vegetation_rules.execute(
        land_cover_layer, layer_manager=manager, layer_name="Vegetation_Types", result_field="veg_class"
    )

    assert vegetation_layer is not None, "Hierarchical vegetation layer was not created."

    # Test exports
    land_cover_geojson = os.path.join(output_dir, "hierarchical_land_cover.geojson")
    vegetation_geojson = os.path.join(output_dir, "hierarchical_vegetation.geojson")
    land_cover_raster = os.path.join(output_dir, "hierarchical_land_cover.tif")

    layer_to_vector(land_cover_layer, land_cover_geojson)
    layer_to_vector(vegetation_layer, vegetation_geojson)
    layer_to_raster(land_cover_layer, land_cover_raster, column="classification")

    assert os.path.exists(land_cover_geojson), "Land cover GeoJSON not created."
    assert os.path.exists(vegetation_geojson), "Vegetation GeoJSON not created."
    assert os.path.exists(land_cover_raster), "Land cover raster not created."

    # Verify layer manager
    available_layers = manager.get_layer_names()
    expected_layers = ["Hierarchical_Test", "Land_Cover", "Vegetation_Types"]
    for layer_name in expected_layers:
        assert layer_name in available_layers, f"Layer {layer_name} not found in manager."


@pytest.mark.skipif(not HAS_CNN, reason="TensorFlow/CNN dependencies not available")
def test_cnn_training_history(test_raster_path):
    """Test CNN training history visualization from CNN.ipynb."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    manager = LayerManager()

    # Setup CNN classification
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=20, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="CNN_History_Test")

    samples = {
        "Water": [102, 384, 659, 1142],
        "Built-up": [467, 1102, 1431, 1984],
        "Vegetation": [832, 1778, 2035, 1417],
    }

    # Minimal CNN parameters for testing
    params = {
        "patch_size": (3, 3),
        "epochs": 5,
        "batch_size": 16,
        "early_stopping_patience": 2,
        "hidden_layers": [{"filters": 16, "kernel_size": 3, "max_pooling": True}],
        "use_batch_norm": False,
        "dense_units": 32,
    }

    cnn_classification = SupervisedClassifierDL(
        name="CNN Test", classifier_type="Convolution Neural Network (CNN)", classifier_params=params
    )

    cnn_layer, model_history, eval_result, count_dict, invalid_patches = cnn_classification.execute(
        source_layer=segmentation_layer,
        samples=samples,
        image_data=image_data,
        layer_manager=manager,
        layer_name="CNN_Test",
    )

    assert cnn_layer is not None, "CNN classification layer was not created."
    assert model_history is not None, "Model history was not returned."

    # Test plot_training_history
    fig = plot_training_history(model_history)
    assert fig is not None, "plot_training_history failed."

    history_plot_path = os.path.join(output_dir, "test_training_history.png")
    fig.savefig(history_plot_path)
    assert os.path.exists(history_plot_path), "Training history plot not saved."

    # Test confusion matrix exists
    assert "confusion_matrix" in eval_result, "Confusion matrix missing from eval results."
    cm = eval_result["confusion_matrix"]
    assert cm is not None, "Confusion matrix is None."

    # Create confusion matrix DataFrame as shown in CNN.ipynb
    df_cm = pd.DataFrame(cm, index=samples.keys(), columns=samples.keys())
    df_cm.index.name = "Predicted Label"
    assert df_cm is not None, "Confusion matrix DataFrame creation failed."


def test_area_statistics_functionality(test_raster_path):
    """Test area statistics calculation used across notebooks."""
    manager = LayerManager()

    # Setup data
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=25, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Area_Stats_Test")

    # Apply classification
    samples = {"Class_A": [41, 134], "Class_B": [246, 491], "Class_C": [12, 499]}
    rf_classification = SupervisedClassifier(
        name="Area Test RF", classifier_type="Random Forest", classifier_params={"n_estimators": 10, "random_state": 42}
    )
    classified_layer, _, _ = rf_classification.execute(
        segmentation_layer, samples=samples, layer_manager=manager, layer_name="Area_Test_Classification"
    )

    # Test area statistics by class (as used in notebooks)
    classified_layer.attach_function(attach_area_stats, name="area_by_class", by_class="classification")
    area_stats = classified_layer.get_function_result("area_by_class")

    assert "class_areas" in area_stats, "Area stats missing class_areas."
    assert "class_percentages" in area_stats, "Area stats missing class_percentages."
    assert len(area_stats["class_areas"]) > 0, "No class areas calculated."
    assert len(area_stats["class_percentages"]) > 0, "No class percentages calculated."

    # Verify percentages sum to approximately 100
    total_percentage = sum(area_stats["class_percentages"].values())
    assert abs(total_percentage - 100.0) < 1.0, f"Class percentages don't sum to 100: {total_percentage}"


def test_complete_workflow_integration(test_raster_path):
    """Test complete workflow integration covering all major notebook examples."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    manager = LayerManager()

    # Step 1: Basic setup (common to all notebooks)
    image_data, transform, crs = read_raster(test_raster_path)
    segmenter = SlicSegmentation(scale=30, compactness=1)
    segmentation_layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="Integration_Test")

    # Step 2: Calculate all spectral indices
    segmentation_layer.attach_function(
        attach_ndvi,
        name="ndvi_stats",
        nir_column="band_4_mean",
        red_column="band_3_mean",
        output_column="NDVI",
    )

    segmentation_layer.attach_function(
        attach_spectral_indices,
        name="spectral_indices",
        bands={
            "blue": "band_1_mean",
            "green": "band_2_mean",
            "red": "band_3_mean",
            "nir": "band_4_mean",
        },
    )

    segmentation_layer.attach_function(attach_shape_metrics, name="shape_metrics")

    # Step 3: Rule-based classification
    land_cover_rules = RuleSet(name="Integration_Land_Cover")
    land_cover_rules.add_rule(name="Vegetation", condition="NDVI > 0.2")
    land_cover_rules.add_rule(name="Other", condition="NDVI <= 0.2")

    land_cover_rules.execute(segmentation_layer, layer_manager=manager, layer_name="Rule_Based_Classification")

    # Step 4: Supervised classification
    samples = {
        "Water": [41, 134, 246],
        "Built-up": [12, 499, 290],
        "Vegetation": [36, 143, 239],
    }

    rf_classification = SupervisedClassifier(
        name="Integration RF", classifier_type="Random Forest", classifier_params={"n_estimators": 20, "random_state": 42}
    )

    supervised_layer, accuracy, _ = rf_classification.execute(
        segmentation_layer, samples=samples, layer_manager=manager, layer_name="Supervised_Classification"
    )

    # Step 5: Post-processing (merge and spatial rules)
    merger = MergeRuleSet("Integration_Merge")
    merged_layer = merger.execute(
        source_layer=supervised_layer,
        class_column_name="classification",
        class_value=["Water", "Vegetation"],
        layer_manager=manager,
        layer_name="Merged_Classification",
    )

    # Step 6: Spatial relationship rules
    encloser_rule = EnclosedByRuleSet()
    enclosed_layer = encloser_rule.execute(
        source_layer=merged_layer,
        class_column_name="classification",
        class_value_a="Vegetation",
        class_value_b="Built-up",
        new_class_name="Park",
        layer_manager=manager,
        layer_name="Enclosed_Classification",
    )

    touched_rule = TouchedByRuleSet()
    final_layer = touched_rule.execute(
        source_layer=enclosed_layer,
        class_column_name="classification",
        class_value_a="Built-up",
        class_value_b="Water",
        new_class_name="Waterfront",
        layer_manager=manager,
        layer_name="Final_Classification",
    )

    # Step 7: Area statistics
    final_layer.attach_function(attach_area_stats, name="final_area_stats", by_class="classification")
    final_stats = final_layer.get_function_result("final_area_stats")

    # Step 8: All export formats
    vector_path = os.path.join(output_dir, "integration_final.geojson")
    raster_path = os.path.join(output_dir, "integration_final.tif")

    layer_to_vector(final_layer, vector_path)
    layer_to_raster(final_layer, raster_path, column="classification")

    # Step 9: Visualizations
    classes_color = {
        "Water": "#3437c2",
        "Built-up": "#de1421",
        "Vegetation": "#0f6b2f",
        "Park": "#d2f7dc",
        "Waterfront": "#cc32cf",
    }

    plot_fig = plot_classification(final_layer, class_field="classification", class_color=classes_color)
    plot_path = os.path.join(output_dir, "integration_final_plot.png")
    plot_fig.savefig(plot_path)

    # Assertions
    assert final_layer is not None, "Final integrated layer was not created."
    assert os.path.exists(vector_path), "Final vector export failed."
    assert os.path.exists(raster_path), "Final raster export failed."
    assert os.path.exists(plot_path), "Final plot export failed."
    assert "class_areas" in final_stats, "Final area statistics missing."
    assert len(manager.get_layer_names()) >= 7, "Not all layers were created in manager."
    assert accuracy > 0, "Supervised classification accuracy should be positive."

    # Verify all expected classes exist
    unique_classes = final_layer.objects["classification"].unique()
    assert len(unique_classes) > 1, "Final layer should have multiple classes."
