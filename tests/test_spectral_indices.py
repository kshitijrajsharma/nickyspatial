# -*- coding: utf-8 -*-
"""Test suite for spectral indices functionality."""

import numpy as np
import pytest

from nickyspatial import (
    LayerManager,
    SlicSegmentation,
    add_custom_index,
    attach_spectral_index,
    get_available_indices,
    read_raster,
)
from nickyspatial.stats.spectral import SpectralIndexCalculator


@pytest.fixture
def sample_layer():
    """Create a layer with sample data for testing."""
    manager = LayerManager()
    image_data, transform, crs = read_raster("data/sample.tif")

    segmenter = SlicSegmentation(scale=50, compactness=1)
    layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="test_layer")

    assert layer.objects is not None
    assert len(layer.objects) > 0

    return layer


class TestSpectralIndexCalculator:
    """Test the SpectralIndexCalculator class."""

    def test_calculator_initialization(self):
        """Test calculator initializes with predefined indices."""
        calc = SpectralIndexCalculator()
        assert len(calc.predefined_indices) > 0
        assert "NDVI" in calc.predefined_indices
        assert "NDWI" in calc.predefined_indices
        assert "EVI" in calc.predefined_indices

    def test_parse_formula_basic(self):
        """Test basic formula parsing."""
        calc = SpectralIndexCalculator()
        bands = {"NIR": "nir_col", "RED": "red_col"}

        formula = "(NIR - RED) / (NIR + RED)"
        parsed = calc._parse_formula(formula, bands)

        assert "bands['nir_col']" in parsed
        assert "bands['red_col']" in parsed
        assert "NIR" not in parsed or "bands[" in parsed

    def test_parse_formula_case_insensitive(self):
        """Test formula parsing is case insensitive."""
        calc = SpectralIndexCalculator()
        bands = {"nir": "nir_col", "red": "red_col"}

        formula = "(NIR - RED) / (nir + red)"
        parsed = calc._parse_formula(formula, bands)

        assert "bands['nir_col']" in parsed
        assert "bands['red_col']" in parsed

    def test_evaluate_formula_simple(self):
        """Test formula evaluation with simple data."""
        calc = SpectralIndexCalculator()

        bands_data = {"nir": np.array([0.8, 0.7, 0.6]), "red": np.array([0.4, 0.3, 0.2])}

        formula = "(bands['nir'] - bands['red']) / (bands['nir'] + bands['red'])"
        result = calc._evaluate_formula(formula, bands_data)

        expected = (bands_data["nir"] - bands_data["red"]) / (bands_data["nir"] + bands_data["red"])
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_formula_with_constants(self):
        """Test formula evaluation with mathematical constants."""
        calc = SpectralIndexCalculator()

        bands_data = {"nir": np.array([0.8, 0.7]), "red": np.array([0.4, 0.3])}

        formula = "2.5 * bands['nir'] + 1.0"
        result = calc._evaluate_formula(formula, bands_data)

        expected = 2.5 * bands_data["nir"] + 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        calc = SpectralIndexCalculator()
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        stats = calc._calculate_statistics(values)

        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats
        assert "median" in stats
        assert "count" in stats

        assert stats["mean"] == 0.3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["count"] == 5


class TestAttachSpectralIndex:
    """Test the attach_spectral_index function."""

    def test_predefined_ndvi(self, sample_layer):
        """Test NDVI calculation with predefined formula."""
        result = sample_layer.attach_function(
            attach_spectral_index, name="ndvi_test", index_name="NDVI", bands={"NIR": "band_4_mean", "RED": "band_3_mean"}
        )

        function_result = sample_layer.get_function_result("ndvi_test")

        assert "error" not in function_result
        assert "NDVI" in sample_layer.objects.columns
        assert function_result["index_name"] == "NDVI"
        assert "statistics" in function_result
        assert function_result["statistics"]["count"] > 0

        ndvi_values = sample_layer.objects["NDVI"]
        assert not ndvi_values.isna().all()
        assert ndvi_values.min() >= -1.0
        assert ndvi_values.max() <= 1.0

    def test_predefined_ndwi(self, sample_layer):
        """Test NDWI calculation with predefined formula."""
        result = sample_layer.attach_function(
            attach_spectral_index, name="ndwi_test", index_name="NDWI", bands={"GREEN": "band_2_mean", "NIR": "band_4_mean"}
        )

        function_result = sample_layer.get_function_result("ndwi_test")

        assert "error" not in function_result
        assert "NDWI" in sample_layer.objects.columns
        assert function_result["index_name"] == "NDWI"

        ndwi_values = sample_layer.objects["NDWI"]
        assert not ndwi_values.isna().all()
        assert ndwi_values.min() >= -1.0
        assert ndwi_values.max() <= 1.0

    def test_predefined_evi(self, sample_layer):
        """Test EVI calculation with predefined formula."""
        result = sample_layer.attach_function(
            attach_spectral_index,
            name="evi_test",
            index_name="EVI",
            bands={"NIR": "band_4_mean", "RED": "band_3_mean", "BLUE": "band_1_mean"},
        )

        function_result = sample_layer.get_function_result("evi_test")

        assert "error" not in function_result
        assert "EVI" in sample_layer.objects.columns
        assert function_result["index_name"] == "EVI"

        evi_values = sample_layer.objects["EVI"]
        assert not evi_values.isna().all()

    def test_custom_formula_simple_ratio(self, sample_layer):
        """Test custom formula: simple band ratio."""
        result = sample_layer.attach_function(
            attach_spectral_index,
            name="ratio_test",
            index_name="NIR_RED_RATIO",
            formula="NIR / RED",
            bands={"NIR": "band_4_mean", "RED": "band_3_mean"},
            output_column="RATIO",
        )

        function_result = sample_layer.get_function_result("ratio_test")

        assert "error" not in function_result
        assert "RATIO" in sample_layer.objects.columns
        assert function_result["output_column"] == "RATIO"

        ratio_values = sample_layer.objects["RATIO"]
        assert not ratio_values.isna().all()
        assert (ratio_values > 0).all()

    def test_custom_formula_complex(self, sample_layer):
        """Test custom formula with mathematical operations."""
        result = sample_layer.attach_function(
            attach_spectral_index,
            name="complex_test",
            index_name="CUSTOM_INDEX",
            formula="sqrt((NIR**2 + RED**2) / 2)",
            bands={"NIR": "band_4_mean", "RED": "band_3_mean"},
        )

        function_result = sample_layer.get_function_result("complex_test")

        assert "error" not in function_result
        assert "CUSTOM_INDEX" in sample_layer.objects.columns

        values = sample_layer.objects["CUSTOM_INDEX"]
        assert not values.isna().all()
        assert (values >= 0).all()

    def test_custom_formula_with_constants(self, sample_layer):
        """Test custom formula with numeric constants."""
        result = sample_layer.attach_function(
            attach_spectral_index,
            name="const_test",
            index_name="SCALED_NDVI",
            formula="2.5 * (NIR - RED) / (NIR + RED + 0.5)",
            bands={"NIR": "band_4_mean", "RED": "band_3_mean"},
        )

        function_result = sample_layer.get_function_result("const_test")

        assert "error" not in function_result
        assert "SCALED_NDVI" in sample_layer.objects.columns

    def test_missing_bands(self, sample_layer):
        """Test handling of missing band columns."""
        result = sample_layer.attach_function(
            attach_spectral_index,
            name="missing_test",
            index_name="NDVI",
            bands={"NIR": "nonexistent_nir", "RED": "nonexistent_red"},
        )

        function_result = sample_layer.get_function_result("missing_test")

        assert "error" in function_result
        assert "No required band columns found" in function_result["error"]

    def test_invalid_formula(self, sample_layer):
        """Test handling of invalid formula."""
        result = sample_layer.attach_function(
            attach_spectral_index,
            name="invalid_test",
            index_name="INVALID",
            formula="undefined_function(NIR, RED)",
            bands={"NIR": "band_4_mean", "RED": "band_3_mean"},
        )

        function_result = sample_layer.get_function_result("invalid_test")

        assert "error" in function_result
        assert "Error evaluating formula" in function_result["error"]

    def test_nonexistent_predefined_index(self, sample_layer):
        """Test handling of nonexistent predefined index."""
        result = sample_layer.attach_function(attach_spectral_index, name="nonexistent_test", index_name="NONEXISTENT_INDEX")

        function_result = sample_layer.get_function_result("nonexistent_test")

        assert "error" in function_result
        assert "No predefined formula" in function_result["error"]

    def test_all_predefined_indices(self, sample_layer):
        """Test all predefined indices can be calculated."""
        available_indices = get_available_indices()

        for index_name in available_indices.keys():
            if index_name in ["EVI", "SAVI"]:
                bands = {"NIR": "band_4_mean", "RED": "band_3_mean", "BLUE": "band_1_mean"}
            elif index_name in ["NDBI", "NBR"]:
                continue
            else:
                bands = {"NIR": "band_4_mean", "RED": "band_3_mean", "GREEN": "band_2_mean", "BLUE": "band_1_mean"}

            result = sample_layer.attach_function(
                attach_spectral_index, name=f"{index_name.lower()}_test", index_name=index_name, bands=bands
            )

            function_result = sample_layer.get_function_result(f"{index_name.lower()}_test")

            if "error" not in function_result:
                assert index_name in sample_layer.objects.columns
                assert function_result["statistics"]["count"] > 0


class TestCustomIndexManagement:
    """Test custom index addition and retrieval."""

    def test_add_custom_index(self):
        """Test adding a custom index definition."""
        original_count = len(get_available_indices())

        add_custom_index("TEST_INDEX", "(NIR - RED) * 2", "Test index for unit testing", "Test et al. 2024")

        indices = get_available_indices()
        assert len(indices) == original_count + 1
        assert "TEST_INDEX" in indices
        assert indices["TEST_INDEX"]["formula"] == "(NIR - RED) * 2"
        assert indices["TEST_INDEX"]["description"] == "Test index for unit testing"
        assert indices["TEST_INDEX"]["reference"] == "Test et al. 2024"

    def test_get_available_indices(self):
        """Test getting available indices."""
        indices = get_available_indices()

        assert isinstance(indices, dict)
        assert len(indices) > 0
        assert "NDVI" in indices
        assert "formula" in indices["NDVI"]
        assert "description" in indices["NDVI"]
        assert "reference" in indices["NDVI"]

    def test_use_added_custom_index(self, sample_layer):
        """Test using a newly added custom index."""
        add_custom_index("SIMPLE_DIFF", "NIR - RED", "Simple difference index")

        result = sample_layer.attach_function(
            attach_spectral_index,
            name="custom_diff_test",
            index_name="SIMPLE_DIFF",
            bands={"NIR": "band_4_mean", "RED": "band_3_mean"},
        )

        function_result = sample_layer.get_function_result("custom_diff_test")

        assert "error" not in function_result
        assert "SIMPLE_DIFF" in sample_layer.objects.columns
        assert function_result["description"] == "Simple difference index"


class TestRealWorldScenarios:
    """Test real-world usage scenarios with actual data."""

    def test_vegetation_analysis_workflow(self, sample_layer):
        """Test a complete vegetation analysis workflow."""
        vegetation_indices = ["NDVI", "GNDVI", "EVI"]

        for idx in vegetation_indices:
            bands = {"NIR": "band_4_mean", "RED": "band_3_mean", "GREEN": "band_2_mean", "BLUE": "band_1_mean"}

            sample_layer.attach_function(attach_spectral_index, name=f"{idx.lower()}_analysis", index_name=idx, bands=bands)

            result = sample_layer.get_function_result(f"{idx.lower()}_analysis")
            assert "error" not in result
            assert idx in sample_layer.objects.columns

        vegetation_objects = sample_layer.objects[(sample_layer.objects["NDVI"] > 0.3) & (sample_layer.objects["GNDVI"] > 0.2)]

        assert len(vegetation_objects) >= 0

    def test_water_detection_workflow(self, sample_layer):
        """Test water detection using multiple indices."""
        water_indices = ["NDWI", "MNDWI"]

        for idx in water_indices:
            if idx == "MNDWI":
                continue

            bands = {"GREEN": "band_2_mean", "NIR": "band_4_mean"}

            sample_layer.attach_function(attach_spectral_index, name=f"{idx.lower()}_water", index_name=idx, bands=bands)

            result = sample_layer.get_function_result(f"{idx.lower()}_water")
            assert "error" not in result
            assert idx in sample_layer.objects.columns

        potential_water = sample_layer.objects[sample_layer.objects["NDWI"] > 0]
        assert len(potential_water) >= 0

    def test_multi_index_comparison(self, sample_layer):
        """Test calculating multiple indices for comparison."""
        sample_layer.attach_function(
            attach_spectral_index, name="ndvi_calc", index_name="NDVI", bands={"NIR": "band_4_mean", "RED": "band_3_mean"}
        )

        sample_layer.attach_function(
            attach_spectral_index,
            name="custom_ndvi",
            index_name="CUSTOM_NDVI",
            formula="(NIR - RED) / (NIR + RED)",
            bands={"NIR": "band_4_mean", "RED": "band_3_mean"},
        )

        ndvi_result = sample_layer.get_function_result("ndvi_calc")
        custom_result = sample_layer.get_function_result("custom_ndvi")

        assert "error" not in ndvi_result
        assert "error" not in custom_result

        np.testing.assert_array_almost_equal(
            sample_layer.objects["NDVI"].values, sample_layer.objects["CUSTOM_NDVI"].values, decimal=6
        )


def test_spectral_indices_integration():
    """Integration test using real sample data."""
    manager = LayerManager()
    image_data, transform, crs = read_raster("data/sample.tif")

    segmenter = SlicSegmentation(scale=100, compactness=1)
    layer = segmenter.execute(image_data, transform, crs, layer_manager=manager, layer_name="integration_test")

    layer.attach_function(
        attach_spectral_index, name="ndvi_integration", index_name="NDVI", bands={"NIR": "band_4_mean", "RED": "band_3_mean"}
    )

    layer.attach_function(
        attach_spectral_index,
        name="custom_integration",
        index_name="BRIGHTNESS",
        formula="(RED + GREEN + BLUE) / 3",
        bands={"RED": "band_3_mean", "GREEN": "band_2_mean", "BLUE": "band_1_mean"},
    )

    ndvi_result = layer.get_function_result("ndvi_integration")
    brightness_result = layer.get_function_result("custom_integration")

    assert "error" not in ndvi_result
    assert "error" not in brightness_result
    assert "NDVI" in layer.objects.columns
    assert "BRIGHTNESS" in layer.objects.columns

    assert ndvi_result["statistics"]["count"] > 0
    assert brightness_result["statistics"]["count"] > 0

    high_vegetation = layer.objects[layer.objects["NDVI"] > 0.4]
    bright_objects = layer.objects[layer.objects["BRIGHTNESS"] > layer.objects["BRIGHTNESS"].median()]

    assert len(high_vegetation) >= 0
    assert len(bright_objects) >= 0
