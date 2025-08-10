# -*- coding: utf-8 -*-
"""Streamlit Webapp.

Frontend for the demo of nickyspatial library
"""

import datetime
import os
import tempfile
from tempfile import NamedTemporaryFile

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from folium.raster_layers import ImageOverlay
from PIL import Image
from pyproj import Transformer
from skimage.segmentation import mark_boundaries
from streamlit_folium import st_folium

import streamlit as st
from nickyspatial import (
    EnclosedByRuleSet,
    LayerManager,
    MergeRuleSet,
    RuleSet,
    SlicSegmentation,
    SupervisedClassifier,
    SupervisedClassifierDL,
    TouchedByRuleSet,
    attach_area_stats,
    attach_ndvi,
    attach_shape_metrics,
    attach_spectral_indices,
    layer_to_raster,
    layer_to_vector,
    plot_classification,
    plot_layer,
    read_raster,
    plot_training_history
)

st.set_page_config(page_title="nickyspatial - Remote Sensing Analysis", page_icon="#", layout="wide")


@st.cache_resource
def get_temp_dir():
    """Get a temporary directory for storing output files."""
    return tempfile.mkdtemp()


def initialize_session_state():
    """Initialize session state variables."""
    if "manager" not in st.session_state:
        st.session_state.manager = LayerManager()
    if "image_data" not in st.session_state:
        st.session_state.image_data = None
    if "transform" not in st.session_state:
        st.session_state.transform = None
    if "crs" not in st.session_state:
        st.session_state.crs = None
    if "layers" not in st.session_state:
        st.session_state.layers = {}
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = get_temp_dir()
    if "rule_sets" not in st.session_state:
        st.session_state.rule_sets = {}
    if "active_ruleset" not in st.session_state:
        st.session_state.active_ruleset = None
    if "band_mappings" not in st.session_state:
        st.session_state.band_mappings = {}
    if "available_attributes" not in st.session_state:
        st.session_state.available_attributes = set()
    if "classes" not in st.session_state:
        st.session_state.classes = {}
    if "active_segmentation_layer_name" not in st.session_state:
        st.session_state.active_segmentation_layer_name = {}
    if "processes" not in st.session_state:
        st.session_state.processes = []
    if "delete_index" not in st.session_state:
        st.session_state.delete_index = None
    if "expanders" not in st.session_state:
        st.session_state.expanders = {}
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = None


def load_raster(file_path):
    """Load raster data and initialize session state variables."""
    try:
        with st.spinner("Reading raster data..."):
            image_data, transform, crs = read_raster(file_path)
            st.session_state.image_data = image_data
            st.session_state.transform = transform
            st.session_state.crs = crs
            st.session_state.manager = LayerManager()
            # st.session_state.layers = {}
            # st.session_state.rule_sets = {}
            print(st.session_state.layers, "hvchdvhc")

            num_bands = image_data.shape[0]
            default_mappings = {}
            if num_bands >= 4:
                default_mappings = {"blue": "band_1_mean", "green": "band_2_mean", "red": "band_3_mean", "nir": "band_4_mean"}
            elif num_bands == 3:
                default_mappings = {"blue": "band_1_mean", "green": "band_2_mean", "red": "band_3_mean"}

            st.session_state.band_mappings = default_mappings
            update_available_attributes()

        return True
    except Exception as e:
        st.error(f"Error loading raster data: {str(e)}")
        return False


def get_layer_attributes(layer_name):
    """Get available attributes from the specified layer."""
    if layer_name and layer_name in st.session_state.layers:
        layer = st.session_state.layers[layer_name]
        if layer and hasattr(layer, "objects") and hasattr(layer.objects, "columns"):
            return [col for col in layer.objects.columns if col not in ["geometry"]]
    return []


def update_available_attributes():
    """Update the available attributes based on the current layers."""
    attributes = set()
    for _layer_name, layer in st.session_state.layers.items():
        if layer and hasattr(layer, "objects") and hasattr(layer.objects, "columns"):
            for col in layer.objects.columns:
                if col not in ["geometry"]:
                    attributes.add(col)
    st.session_state.available_attributes = attributes


def perform_segmentation(image_data, transform, crs, scale_param, compactness_param, segmentation_name):
    """Perform segmentation on the image data."""
    try:
        with st.spinner("Performing segmentation..."):
            segmenter = SlicSegmentation(
                scale=scale_param,
                compactness=compactness_param,
            )

            segmentation_layer = segmenter.execute(
                image_data,
                transform,
                crs,
                layer_manager=st.session_state.manager,
                layer_name=segmentation_name,
            )

            st.session_state.layers[segmentation_name] = segmentation_layer
            update_available_attributes()
            return segmentation_layer
    except Exception as e:
        st.error(f"Error during segmentation: {str(e)}")
        return None


def perform_supervised_classification(layer, selected_classifier, classifier_params, classification_name, features):
    """Perform segmentation on the image data."""
    try:
        with st.spinner("Performing supervised classification..."):
            samples = {}
            for key in list(st.session_state.classes.keys()):
                samples[key] = st.session_state.classes[key]["sample_ids"]

            classifier = SupervisedClassifier(
                name="RF Classification", classifier_type=selected_classifier, classifier_params=classifier_params
            )

            classification_layer, accuracy, feature_importances = classifier.execute(
                layer, samples=samples, layer_manager=st.session_state.manager, layer_name=classification_name, features=features
            )

            st.session_state.layers[classification_name] = classification_layer
            update_available_attributes()
            return classification_layer, accuracy, feature_importances
    except Exception as e:
        st.error(f"Error during supervised classification: {str(e)}")
        return None
    
def perform_supervised_classification_DL(layer,image_data, selected_classifier, classifier_params, classification_name):
    """Perform segmentation on the image data."""
    try:
        with st.spinner("Performing supervised classification..."):
            samples = {}
            for key in list(st.session_state.classes.keys()):
                samples[key] = st.session_state.classes[key]["sample_ids"]

            classifier = SupervisedClassifierDL(
                name="CNN Classification", classifier_type=selected_classifier, classifier_params=classifier_params
            )

            classification_layer, model_history, eval_resul= classifier.execute(
                layer, samples=samples,image_data=image_data, layer_manager=st.session_state.manager, layer_name=classification_name
            )

            st.session_state.layers[classification_name] = classification_layer
            update_available_attributes()
            return classification_layer, model_history, eval_resul
    except Exception as e:
        st.error(f"Error during supervised classification: {str(e)}")
        return None


def perform_merge_region(layer, class_column_name, class_value, layer_name):
    """Perform segmentation on the image data."""
    try:
        with st.spinner("Performing merge regions..."):
            merger = MergeRuleSet("MergeByVegAndType")
            merged_layer = merger.execute(
                source_layer=layer,
                class_column_name=class_column_name,
                class_value=class_value,
                layer_manager=st.session_state.manager,
                layer_name=layer_name,
            )

            st.session_state.layers[layer_name] = merged_layer
            update_available_attributes()
            return merged_layer
    except Exception as e:
        st.error(f"Error during merged_layer: {str(e)}")
        return None


def perform_enclosed_by(layer, class_column_name, class_value_a, class_value_b, new_class_name, layer_name):
    """Perform segmentation on the image data."""
    try:
        with st.spinner("Performing merge regions..."):
            encloser = EnclosedByRuleSet()
            enclosed_by_layer = encloser.execute(
                source_layer=layer,
                class_column_name=class_column_name,
                class_value_a=class_value_a,
                class_value_b=class_value_b,
                new_class_name=new_class_name,
                layer_manager=st.session_state.manager,
                layer_name=layer_name,
            )
            st.session_state.layers[layer_name] = enclosed_by_layer
            update_available_attributes()
            return enclosed_by_layer
    except Exception as e:
        st.error(f"Error during enclosed_by_layer: {str(e)}")
        return None


def perform_touched_by(layer, class_column_name, class_value_a, class_value_b, new_class_name, layer_name):
    """Perform segmentation on the image data."""
    try:
        with st.spinner("Performing merge regions..."):
            touched_by_rule = TouchedByRuleSet()
            touched_by_layer = touched_by_rule.execute(
                source_layer=layer,
                class_column_name=class_column_name,
                class_value_a=class_value_a,
                class_value_b=class_value_b,
                new_class_name=new_class_name,
                layer_manager=st.session_state.manager,
                layer_name=layer_name,
            )
            st.session_state.layers[layer_name] = touched_by_layer
            update_available_attributes()
            return touched_by_layer
    except Exception as e:
        st.error(f"Error during touched_by_layer: {str(e)}")
        return None


def calculate_ndvi(layer, nir_column, red_column, output_column="NDVI"):
    """Calculate NDVI for the specified layer."""
    try:
        with st.spinner("Calculating NDVI..."):
            layer.attach_function(
                attach_ndvi,
                name="ndvi_stats",
                nir_column=nir_column,
                red_column=red_column,
                output_column=output_column,
            )
            update_available_attributes()
        return True
    except Exception as e:
        st.error(f"Error calculating NDVI: {str(e)}")
        return False


def calculate_spectral_indices(layer, band_mappings):
    """Calculate spectral indices for the specified layer."""
    try:
        with st.spinner("Calculating spectral indices..."):
            layer.attach_function(attach_spectral_indices, name="spectral_indices", bands=band_mappings)
            update_available_attributes()
        return True
    except Exception as e:
        st.error(f"Error calculating spectral indices: {str(e)}")
        return False


def calculate_shape_metrics(layer):
    """Calculate shape metrics for the specified layer."""
    try:
        with st.spinner("Calculating shape metrics..."):
            layer.attach_function(attach_shape_metrics, name="shape_metrics")
            update_available_attributes()
        return True
    except Exception as e:
        st.error(f"Error calculating shape metrics: {str(e)}")
        return False


def apply_rule_set(ruleset, input_layer, output_layer_name, result_field):
    """Apply the specified rule set to the input layer."""
    try:
        with st.spinner(f"Applying {ruleset.name} rules..."):
            result_layer = ruleset.execute(
                input_layer, layer_manager=st.session_state.manager, layer_name=output_layer_name, result_field=result_field
            )

            st.session_state.layers[output_layer_name] = result_layer
            update_available_attributes()
            return result_layer
    except Exception as e:
        st.error(f"Error applying rule set: {str(e)}")
        return None


def calculate_area_stats(layer, class_field):
    """Calculate area statistics for the specified layer."""
    try:
        with st.spinner("Calculating area statistics..."):
            layer.attach_function(attach_area_stats, name="area_by_class", by_class=class_field)
            return layer.get_function_result("area_by_class")
    except Exception as e:
        st.error(f"Error calculating area statistics: {str(e)}")
        return None


def export_vector(layer, export_filepath):
    """Export the specified layer as a GeoJSON file."""
    try:
        with st.spinner("Exporting as GeoJSON..."):
            layer_to_vector(layer, export_filepath)
            return True
    except Exception as e:
        st.error(f"Error exporting vector: {str(e)}")
        return False


def export_raster(layer, export_filepath, column):
    """Export the specified layer as a GeoTIFF file."""
    try:
        with st.spinner("Exporting as GeoTIFF..."):
            layer_to_raster(layer, export_filepath, column=column)
            return True
    except Exception as e:
        st.error(f"Error exporting raster: {str(e)}")
        return False


def create_example_rule_sets():
    """Create example rule sets for demonstration."""
    try:
        with st.spinner("Creating example rule sets..."):
            land_cover_rules = RuleSet(name="Land_Cover")
            land_cover_rules.add_rule(name="Vegetation", condition="NDVI > 0.2")
            land_cover_rules.add_rule(name="Other", condition="NDVI <= 0.2")

            vegetation_rules = RuleSet(name="Vegetation_Types")
            vegetation_rules.add_rule(name="Healthy_Vegetation", condition="(classification == 'Vegetation') & (NDVI > 0.6)")
            vegetation_rules.add_rule(
                name="Moderate_Vegetation", condition="(classification == 'Vegetation') & (NDVI <= 0.6) & (NDVI > 0.4)"
            )
            vegetation_rules.add_rule(name="Sparse_Vegetation", condition="(classification == 'Vegetation') & (NDVI <= 0.4)")

            return {"Land_Cover": land_cover_rules, "Vegetation_Types": vegetation_rules}
    except Exception as e:
        st.error(f"Error creating example rule sets: {str(e)}")
        return {}


def render_segmentation(index):
    """Render the segmentation tab for image segmentation and feature calculation."""
    try:
        st.markdown("## Image Segmentation")
        st.write("Configure segmentation parameters and run the algorithm")
        process_data = st.session_state.processes[index]
        if "params" not in process_data:
            process_data["params"] = {}

        col1, col2, col3 = st.columns(3)

        with col1:
            segmentation_name = st.text_input("Segmentation Layer Name", "Base_Segmentation", key=f"seg_name_{index}")
        with col2:
            scale_param = st.slider(
                "Scale Parameter",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Controls the size of segments. Higher values create larger segments.",
                key=f"scale_param_{index}",
            )
        with col3:
            compactness_param = st.slider(
                "Compactness",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Controls the compactness of segments. Higher values create more compact segments.",
                key=f"compactness_param_{index}",
            )

        # st.subheader("Configure Band Mappings")
        st.markdown("#### Configure Band Mappings")
        st.write("Set up mappings for spectral bands to use in indices and analysis")

        # Get raw bands from the image data
        raw_bands = [f"band_{i + 1}" for i in range(st.session_state.image_data.shape[0])]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.session_state.band_mappings["blue"] = st.selectbox(
                "Blue band mapping",
                raw_bands,
                index=raw_bands.index(st.session_state.band_mappings.get("blue", raw_bands[0]).replace("_mean", ""))
                if "blue" in st.session_state.band_mappings
                else 0,
                key=f"blue_band_{index}",
            )
        with col2:
            st.session_state.band_mappings["green"] = st.selectbox(
                "Green band mapping",
                raw_bands,
                index=raw_bands.index(
                    st.session_state.band_mappings.get("green", raw_bands[min(1, len(raw_bands) - 1)]).replace("_mean", "")
                )
                if "green" in st.session_state.band_mappings
                else min(1, len(raw_bands) - 1),
                key=f"green_band_{index}",
            )
        with col3:
            st.session_state.band_mappings["red"] = st.selectbox(
                "Red band mapping",
                raw_bands,
                index=raw_bands.index(
                    st.session_state.band_mappings.get("red", raw_bands[min(2, len(raw_bands) - 1)]).replace("_mean", "")
                )
                if "red" in st.session_state.band_mappings
                else min(2, len(raw_bands) - 1),
                key=f"red_band_{index}",
            )
        with col4:
            if len(raw_bands) > 3:
                st.session_state.band_mappings["nir"] = st.selectbox(
                    "NIR band mapping",
                    raw_bands,
                    index=raw_bands.index(
                        st.session_state.band_mappings.get("nir", raw_bands[min(3, len(raw_bands) - 1)]).replace("_mean", "")
                    )
                    if "nir" in st.session_state.band_mappings
                    else min(3, len(raw_bands) - 1),
                    key=f"nir_band_{index}",
                )
                # process_data["params"]["segmentation_name"] = segmentation_name

        for key in st.session_state.band_mappings:
            if not st.session_state.band_mappings[key].endswith("_mean"):
                st.session_state.band_mappings[key] = f"{st.session_state.band_mappings[key]}_mean"

        segmentation_button = st.button("Run Segmentation", key=f"run_seg_{index}")

        if segmentation_button:
            if segmentation_name in list(st.session_state.layers.keys()):
                st.error("Layer name already exists")
                st.stop()
            segmentation_layer = perform_segmentation(
                st.session_state.image_data,
                st.session_state.transform,
                st.session_state.crs,
                scale_param,
                compactness_param,
                segmentation_name,
            )

            # st.write(segmentation_layer)

            if segmentation_layer:
                # print(type(segmentation_layer),"type seg layer")
                fig = plot_layer(
                    segmentation_layer,
                    st.session_state.image_data,
                    rgb_bands=(3, 2, 1),  # Adjusted for 0-indexed bands
                    show_boundaries=True,
                )
                process_data["output_fig"] = fig
                # process_data["layer_type"] = "segmentation"
            st.success(f"Segmentation '{segmentation_name}' completed successfully!")

        if "output_fig" in process_data:
            st.pyplot(process_data["output_fig"])
            # st.pyplot(fig)
    except Exception as e:
        st.error(f"error: {str(e)}")


def render_calculate_features(index):
    """Render the UI for calculating features based on selected segmentation layer and band attributes."""
    try:
        st.markdown("#### Calculate Features")

        if not st.session_state.layers:
            st.warning("No segmentation layers available. Run segmentation first.")
        else:
            segmentation_for_features = st.selectbox(
                "Select segmentation layer for feature calculation:",
                options=list(st.session_state.layers.keys()),
                key=f"feature_{index}",
            )
            if segmentation_for_features:
                selected_layer = st.session_state.layers[segmentation_for_features]

                # Get attributes available from the selected layer
                layer_attributes = get_layer_attributes(segmentation_for_features)

                # Find band attributes in the layer
                band_attributes = [attr for attr in layer_attributes if attr.startswith("band_") and attr.endswith("_mean")]

                feature_options = st.multiselect(
                    "Select features to calculate:",
                    ["NDVI", "Spectral Indices", "Shape Metrics"],
                    default=[],
                    key=f"feat_options_{index}",
                )

                if "NDVI" in feature_options:
                    st.markdown("#### NDVI Configuration")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        nir_column = st.selectbox(
                            "NIR column:",
                            band_attributes,
                            index=band_attributes.index(st.session_state.band_mappings.get("nir", "band_4_mean"))
                            if "nir" in st.session_state.band_mappings and st.session_state.band_mappings["nir"] in band_attributes
                            else 0,
                            key=f"nir_band1_{index}",
                        )
                    with col2:
                        red_column = st.selectbox(
                            "RED column:",
                            band_attributes,
                            index=band_attributes.index(st.session_state.band_mappings.get("red", "band_3_mean"))
                            if "red" in st.session_state.band_mappings and st.session_state.band_mappings["red"] in band_attributes
                            else 0,
                            key=f"red_band2_{index}",
                        )
                    with col3:
                        ndvi_output = st.text_input("Output column name:", "NDVI", key=f"ndvi_output_{index}")

                if st.button("Calculate Selected Features", key=f"calc_feat_{index}"):
                    features_calculated = False

                    if "NDVI" in feature_options:
                        if calculate_ndvi(selected_layer, nir_column, red_column, ndvi_output):
                            fig = plot_layer(selected_layer, attribute=ndvi_output, title=f"{ndvi_output} Values", cmap="RdYlGn")
                            st.pyplot(fig)
                            st.success(f"{ndvi_output} calculation completed!")
                            features_calculated = True
                            layer_attributes = get_layer_attributes(segmentation_for_features)

                    if "Spectral Indices" in feature_options:
                        indices_band_mappings = {
                            "blue": st.session_state.band_mappings.get("blue", "band_1_mean"),
                            "green": st.session_state.band_mappings.get("green", "band_2_mean"),
                            "red": st.session_state.band_mappings.get("red", "band_3_mean"),
                            "nir": st.session_state.band_mappings.get("nir", "band_4_mean")
                            if "nir" in st.session_state.band_mappings
                            else "band_4_mean",
                        }

                        if calculate_spectral_indices(selected_layer, indices_band_mappings):
                            st.success("Spectral indices calculation completed!")
                            features_calculated = True

                            layer_attributes = get_layer_attributes(segmentation_for_features)
                            # TODO: ndvi_output need to be handled properly
                            indices = [
                                col
                                for col in layer_attributes
                                if col not in ["geometry", "segment_id", ndvi_output] and not col.startswith("band_")
                            ]
                            if indices:
                                st.write("Available spectral indices:")
                                st.write(", ".join(indices))

                                selected_index = st.selectbox("Visualize index:", indices)
                                if selected_index:
                                    fig = plot_layer(selected_layer, attribute=selected_index, title=f"{selected_index} Values")
                                    st.pyplot(fig)

                    if "Shape Metrics" in feature_options:
                        if calculate_shape_metrics(selected_layer):
                            st.success("Shape metrics calculation completed!")
                            features_calculated = True

                            layer_attributes = get_layer_attributes(segmentation_for_features)

                            metrics = [
                                col
                                for col in layer_attributes
                                if col not in ["geometry", "segment_id"] and not col.startswith("band_") and col != ndvi_output
                            ]
                            if metrics:
                                st.write("Available shape metrics:")
                                st.write(", ".join(metrics))

                    if features_calculated:
                        st.success("All selected features calculated successfully!")
    except Exception as e:
        st.error(f"error: {str(e)}")


def render_merge_regions(index):
    """Render merge regions algorithm."""
    try:
        process_data = st.session_state.processes[index]
        if "params" not in process_data:
            process_data["params"] = {}
        col1, col2 = st.columns(2)

        if not st.session_state.layers.keys():
            st.error("Error: No layers available")
            return

        with col1:
            input_layer = st.selectbox(
                "Select input layer:",
                options=list(st.session_state.layers.keys()),
                key=f"input_layer_{index}",
                index=list(st.session_state.layers.keys()).index(
                    process_data["params"].get("input_layer", list(st.session_state.layers.keys())[0])
                ),
            )
            process_data["params"]["input_layer"] = input_layer
            layer = st.session_state.layers[input_layer]
            layer_objects = layer.objects
            try:
                value_option_list = list(layer_objects["classification"].unique())
            except Exception:
                st.error("Layer is invalid")
                st.stop()

            attr_option_list = list(layer_objects.columns)
            class_column_name = st.selectbox(
                "Select Column",
                options=attr_option_list,
                key=f"attr_column_{index}",
                index=attr_option_list.index(process_data["params"].get("class_column_name", attr_option_list[0])),
            )
            process_data["params"]["class_column_name"] = class_column_name

            value_option_list = list(layer_objects[class_column_name].unique())
            value_option_list.insert(0, "All")

        with col2:
            layer_name = st.text_input(
                "Layer Name", value=process_data["params"].get("layer_name", "Merged Regions"), key=f"layer_name_{index}"
            )
            process_data["params"]["layer_name"] = layer_name

            # Ensure class_value is a list of valid options from value_option_list
            default_values = process_data["params"].get("class_value", ["All"])
            valid_default_values = [value for value in default_values if value in value_option_list]
            class_value = st.multiselect(
                "Select Values", options=value_option_list, default=valid_default_values, key=f"class_value_{index}"
            )
            if "All" in class_value:
                class_value = [item for item in value_option_list if item != "All"]

            process_data["params"]["class_value"] = class_value
        execute_button = st.button("Execute", key=f"execute_merge_{index}")  # key=f"execute_merge_{index}
        if execute_button:
            if layer_name in list(st.session_state.layers.keys()):
                st.error("Layer name already exists")
                st.stop()
            merged_layer = perform_merge_region(layer, class_column_name, class_value, layer_name)
            if merged_layer:
                class_color = {}
                for key in list(st.session_state.classes.keys()):
                    class_color[key] = st.session_state.classes[key]["color"]

                fig = plot_classification(merged_layer, class_field="classification", class_color=class_color)
                process_data["output_fig"] = fig
                # process_data["layer_type"] = "classification"

        if "output_fig" in process_data:
            st.pyplot(process_data["output_fig"])
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_enclosed_by_class(index):
    """Render elclosed_by_class algorithm."""
    try:
        process_data = st.session_state.processes[index]
        if "params" not in process_data:
            process_data["params"] = {}
        col1, col2 = st.columns(2)

        with col1:
            input_layer = st.selectbox(
                "Select input layer:",
                options=list(st.session_state.layers.keys()),
                key=f"input_layer_{index}",
                index=list(st.session_state.layers.keys()).index(
                    process_data["params"].get("input_layer", list(st.session_state.layers.keys())[0])
                ),
            )
            process_data["params"]["input_layer"] = input_layer
            layer = st.session_state.layers[input_layer]
            layer_objects = layer.objects
            try:
                value_option_list = list(layer_objects["classification"].unique())
            except Exception:
                st.error("Layer is invalid")
                st.stop()
        with col2:
            layer_name = st.text_input(
                "Layer Name", value=process_data["params"].get("layer_name", "Enclosed by"), key=f"layer_name_{index}"
            )
            process_data["params"]["layer_name"] = layer_name

        cola, colb, colc, cold = st.columns(4)

        with cola:
            # Ensure class_value is a list of valid options from value_option_list
            class_value_a = process_data["params"].get("class_value", 0)
            if class_value_a in value_option_list:
                class_value_index = value_option_list.index(class_value_a)
            else:
                class_value_index = 0

            # valid_default_values = [value for value in default_values if value in value_option_list]
            class_value = st.selectbox(
                "Select class", options=value_option_list, index=class_value_index, key=f"class_value_{index}"
            )
            process_data["params"]["class_value"] = class_value
        with colb:
            class_value_b = process_data["params"].get("enclosing_class_value", 0)
            if class_value_b in value_option_list:
                class_value_b_index = value_option_list.index(class_value_b)
            else:
                class_value_b_index = 0

            enclosing_class_value = st.selectbox(
                "Select enclosing class", options=value_option_list, index=class_value_b_index, key=f"enclosing_class_value_{index}"
            )
            process_data["params"]["enclosing_class_value"] = enclosing_class_value
        with colc:
            new_class_name = st.text_input(
                "New Class Name", value=process_data["params"].get("new_class_name", "new_class"), key=f"new_class_{index}"
            )

            process_data["params"]["new_class_name"] = new_class_name
        with cold:
            st.markdown("<div style='font-size: 14px;  margin-bottom: 4px;'>Color</div>", unsafe_allow_html=True)

            new_class_color_1 = st.color_picker(
                "choose color", "#000000", label_visibility="collapsed", key=f"new_class_color_{index}"
            )

        st.session_state.classes[new_class_name] = {"color": new_class_color_1, "sample_ids": []}

        execute_button = st.button("Execute", key=f"execute_enclosed_by_{index}")  # key=f"execute_enclosed_by_{index}"
        if execute_button:
            if layer_name in list(st.session_state.layers.keys()):
                st.error("Layer name already exists")
                st.stop()
            enclosed_by_layer = perform_enclosed_by(
                layer,
                class_column_name="classification",
                class_value_a=class_value,
                class_value_b=enclosing_class_value,
                new_class_name=new_class_name,
                layer_name=layer_name,
            )
            if enclosed_by_layer:
                class_color = {}
                for key in list(st.session_state.classes.keys()):
                    class_color[key] = st.session_state.classes[key]["color"]

                fig = plot_classification(enclosed_by_layer, class_field="classification", class_color=class_color)
                process_data["output_fig"] = fig
                # process_data["layer_type"] = "classification"

        if "output_fig" in process_data:
            st.pyplot(process_data["output_fig"])
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_touched_by_class(index):
    """Render elclosed_by_class algorithm."""
    try:
        process_data = st.session_state.processes[index]
        if "params" not in process_data:
            process_data["params"] = {}
        col1, col2 = st.columns(2)

        with col1:
            input_layer = st.selectbox(
                "Select input layer:",
                options=list(st.session_state.layers.keys()),
                key=f"input_layer_{index}",
                index=list(st.session_state.layers.keys()).index(
                    process_data["params"].get("input_layer", list(st.session_state.layers.keys())[0])
                ),
            )
            process_data["params"]["input_layer"] = input_layer
            layer = st.session_state.layers[input_layer]
            layer_objects = layer.objects
            try:
                value_option_list = list(layer_objects["classification"].unique())
            except Exception:
                st.error("Layer is invalid")
                st.stop()
        with col2:
            layer_name = st.text_input(
                "Layer Name", value=process_data["params"].get("layer_name", "Touched by"), key=f"layer_name_{index}"
            )
            process_data["params"]["layer_name"] = layer_name

        cola, colb, colc, cold = st.columns(4)

        with cola:
            # Ensure class_value is a list of valid options from value_option_list
            class_value_a = process_data["params"].get("class_value", 0)
            if class_value_a in value_option_list:
                class_value_index = value_option_list.index(class_value_a)
            else:
                class_value_index = 0

            # valid_default_values = [value for value in default_values if value in value_option_list]
            class_value = st.selectbox(
                "Select class", options=value_option_list, index=class_value_index, key=f"class_value_{index}"
            )
            process_data["params"]["class_value"] = class_value
        with colb:
            class_value_b = process_data["params"].get("enclosing_class_value", 0)
            if class_value_b in value_option_list:
                class_value_b_index = value_option_list.index(class_value_b)
            else:
                class_value_b_index = 0

            enclosing_class_value = st.selectbox(
                "Select enclosing class", options=value_option_list, index=class_value_b_index, key=f"enclosing_class_value_{index}"
            )
            process_data["params"]["enclosing_class_value"] = enclosing_class_value
        with colc:
            new_class_name = st.text_input(
                "New Class Name", value=process_data["params"].get("new_class_name", "new_class"), key=f"new_class_{index}"
            )

            process_data["params"]["new_class_name"] = new_class_name
        with cold:
            st.markdown("<div style='font-size: 14px;  margin-bottom: 4px;'>Color</div>", unsafe_allow_html=True)

            new_class_color_1 = st.color_picker(
                "choose color", "#000000", label_visibility="collapsed", key=f"new_class_color_{index}"
            )

        st.session_state.classes[new_class_name] = {"color": new_class_color_1, "sample_ids": []}

        execute_button = st.button("Execute", key=f"execute_touched_by_{index}")  # key=f"execute_touched_by_{index}"
        if execute_button:
            if layer_name in list(st.session_state.layers.keys()):
                st.error("Layer name already exists")
                st.stop()
            touched_by_layer = perform_touched_by(
                layer,
                class_column_name="classification",
                class_value_a=class_value,
                class_value_b=enclosing_class_value,
                new_class_name=new_class_name,
                layer_name=layer_name,
            )
            if touched_by_layer:
                class_color = {}
                for key in list(st.session_state.classes.keys()):
                    class_color[key] = st.session_state.classes[key]["color"]

                fig = plot_classification(touched_by_layer, class_field="classification", class_color=class_color)
                process_data["output_fig"] = fig
                # process_data["layer_type"] = "classification"

        if "output_fig" in process_data:
            st.pyplot(process_data["output_fig"])
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_select_samples(index):
    """Render select samples window that allows to create the class and select the samples interactively."""
    try:
        st.markdown("### Sample Collection")

        # Sidebar for class management
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("#### ➕ Add New Class")
            col2a, col2b = st.columns([2, 1])  # Adjust ratio as needed

            with col2a:
                new_class = st.text_input("Class Name", key=f"new_class_input_{index}")

            with col2b:
                st.markdown("<div style='font-size: 14px;  margin-bottom: 4px;'>Color</div>", unsafe_allow_html=True)

                new_class_color = st.color_picker(
                    "choose color", "#000000", label_visibility="collapsed", key=f"new_class_color_{index}"
                )

            if st.button("Add Class", key=f"add_class_{index}"):
                if new_class and new_class not in st.session_state.classes:
                    st.session_state.classes[new_class] = {"color": new_class_color, "sample_ids": []}
                    st.success(f"Class '{new_class}' added!")
                elif new_class in st.session_state.classes:
                    st.warning("Class already exists.")
                else:
                    st.error("Please enter a class name.")

            st.markdown("##### Classes")
            class_names = list(st.session_state.classes.keys())
            for idx, class_name in enumerate(class_names):
                # for class_name, class_info in st.session_state.classes.items():

                # color = class_info["color"]
                color = st.session_state.classes[class_name]["color"]
                col1a, col2b, col3c = st.columns([0.8, 0.20, 0.20])

                with col1a:
                    # st.markdown(f"<div style='padding-top: 5px;'>{class_name}</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='display:inline-flex;align-items:center;margin-bottom:0px;'>"
                        f"<div style='width:25px;height:25px;background:{color};border:1px solid black;margin-right:8px;'></div>"
                        f"{class_name}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with col2b:
                    if st.button("✏️", key=f"edit_{class_name}"):
                        st.session_state["edit_index"] = idx
                        st.session_state["edit_mode"] = True

                with col3c:
                    if st.button("🗑️", key=f"delete_{class_name}"):
                        del st.session_state.classes[class_name]
                        st.rerun()
            if st.session_state.edit_mode and st.session_state.edit_index is not None:
                edit_idx = st.session_state.edit_index
                class_name = class_names[edit_idx]
                current_color = st.session_state.classes[class_name]["color"]

                st.markdown("#### Edit Class")
                colb1, colb2 = st.columns([0.7, 0.3])
                with colb1:
                    new_name = st.text_input("Class name", value=class_name, key="edit_name")
                with colb2:
                    new_color = st.color_picker("Color", value=current_color, key="edit_color")

                cola1, cola2 = st.columns([0.3, 0.5])
                with cola1:
                    if st.button("Update", key=f"save_changes_{index}"):
                        # Apply changes
                        st.session_state.classes[new_name] = {
                            "color": new_color,
                            "sample_ids": st.session_state.classes[class_name]["sample_ids"],
                        }
                        if new_name != class_name:
                            del st.session_state.classes[class_name]
                        st.session_state.edit_mode = False
                        st.session_state.edit_index = None
                        st.rerun()
                with cola2:
                    if st.button("Cancel", key=f"cancel_{index}"):
                        st.session_state.edit_mode = False
                        st.session_state.edit_index = None

            selected_class = st.radio("Select Class", list(st.session_state.classes.keys()), key=f"class_radio_{index}")

        with col2:
            st.markdown(
                """
                <style>
                    .element-container iframe {
                        width: 100% !important;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Click Segments on Interactive Map")

            # st.markdown("#### Load Example Rule Sets")

            if st.button("Load Training Sample Example", key=f"load_sample_{index}"):
                st.session_state.classes = {
                    "Water": {"color": "#3437c2", "sample_ids": [102, 384, 659, 1142, 1662, 1710, 2113, 2182, 2481, 1024]},
                    "Builtup": {"color": "#de1421", "sample_ids": [467, 1102, 1431, 1984, 1227, 1736, 774, 1065]},
                    "Vegetation": {"color": "#0f6b2f", "sample_ids": [832, 1778, 2035, 1417, 1263, 242, 2049, 2397]},
                }

            # Select the input layer
            layers_keys = list(st.session_state.layers.keys())
            if not layers_keys:
                st.warning("No segmentation layers available in session state.")

            col2a, col2b = st.columns([0.5, 0.5])
            with col2a:
                input_layer = st.selectbox(
                    "Select segmentation layer:", options=list(st.session_state.layers.keys()), key=f"select_box_{index}"
                )
            st.session_state.active_segmentation_layer_name = input_layer
            layer = st.session_state.layers[input_layer]
            segments = layer.raster
            transform = layer.transform
            crs = layer.crs

            image_data = st.session_state.image_data

            # rgb_bands = (3, 2, 1)  # Adjust as needed

            # Create RGB base image (grayscale fallback)
            if image_data.shape[0] >= 3:
                # Get raw bands from the image data
                raw_bands = [f"band_{i + 1}" for i in range(image_data.shape[0])]

                col1, col2, col3 = st.columns(3)

                with col1:
                    red_band_index = st.selectbox(
                        "Red band",
                        raw_bands,
                        index=0,
                        key=f"red_map_{index}",
                    )

                with col2:
                    green_band_index = st.selectbox(
                        "Green band",
                        raw_bands,
                        index=1,
                        key=f"green_map_{index}",
                    )
                with col3:
                    blue_band_index = st.selectbox(
                        "Blue band",
                        raw_bands,
                        index=2,
                        key=f"blue_map_{index}",
                    )
                col11, col12, _ = st.columns(3)
                with col11:
                    show_boundaries = st.checkbox("Show segment boundaries", value=True)
                with col12:
                    show_samples = st.checkbox("Show samples", value=True)

                r_index = raw_bands.index(red_band_index)
                g_index = raw_bands.index(green_band_index)
                b_index = raw_bands.index(blue_band_index)
                r = image_data[r_index]
                g = image_data[g_index]
                b = image_data[b_index]

                r_norm = np.clip((r - r.min()) / (r.max() - r.min() + 1e-10), 0, 1)
                g_norm = np.clip((g - g.min()) / (g.max() - g.min() + 1e-10), 0, 1)
                b_norm = np.clip((b - b.min()) / (b.max() - b.min() + 1e-10), 0, 1)

                base_img = np.stack([r_norm, g_norm, b_norm], axis=2)
            else:
                gray = image_data[0]
                gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
                base_img = np.stack([gray_norm] * 3, axis=2)

            # Assign colors to labeled segments
            segment_colored_img = (base_img * 255).astype(np.uint8).copy()

            # Assign colors to labeled segments
            for _class_name, class_data in st.session_state.classes.items():
                color = class_data["color"]
                if color.startswith("#"):
                    color_rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
                else:
                    color_rgb = color  # Assume it's already an RGB triplet
                for seg_id in class_data["sample_ids"]:
                    mask = segments == seg_id
                    for c in range(3):  # RGB channels
                        segment_colored_img[:, :, c][mask] = color_rgb[c]

            if show_boundaries:
                if show_samples:
                    final_img = segment_colored_img
                else:
                    final_img = base_img
                overlay_img = mark_boundaries(final_img, segments, color=(1, 1, 0), mode="thick")
                overlay_uint8 = (overlay_img * 255).astype(np.uint8)
            else:
                if show_samples:
                    overlay_uint8 = segment_colored_img
                else:
                    overlay_uint8 = (base_img * 255).astype(np.uint8)

            # Save overlay to a temporary file
            tmp_path = NamedTemporaryFile(suffix=".png", delete=False).name
            Image.fromarray(overlay_uint8).save(tmp_path)
            # Calculate map bounds in EPSG:4326
            height, width = segments.shape
            top_left_utm = rasterio.transform.xy(transform, 0, 0, offset="ul")
            bottom_right_utm = rasterio.transform.xy(transform, height, width, offset="lr")
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            top_left_latlon = transformer.transform(*top_left_utm)
            bottom_right_latlon = transformer.transform(*bottom_right_utm)
            bounds = [[bottom_right_latlon[1], bottom_right_latlon[0]], [top_left_latlon[1], top_left_latlon[0]]]

            # Create the map
            center_lat = (top_left_latlon[1] + bottom_right_latlon[1]) / 2
            center_lon = (top_left_latlon[0] + bottom_right_latlon[0]) / 2
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15)
            ImageOverlay(
                name="Colored Segments", image=tmp_path, bounds=bounds, opacity=0.8, interactive=True, cross_origin=False
            ).add_to(fmap)
            folium.LayerControl().add_to(fmap)

            # st.write(st.session_state.classes)

            # Display the map and handle click events
            click_info = st_folium(fmap, height=600, width=1000, key=f"map_folium_{index}")

            if show_boundaries:
                # Process click events
                if click_info and click_info.get("last_clicked"):
                    lat = click_info["last_clicked"]["lat"]
                    lon = click_info["last_clicked"]["lng"]
                    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
                    col, row = ~transform * (x, y)
                    col, row = int(col), int(row)
                    if 0 <= row < segments.shape[0] and 0 <= col < segments.shape[1]:
                        seg_id = int(segments[row, col])
                        found = False
                        for _class_name, class_data in st.session_state.classes.items():
                            if seg_id in class_data["sample_ids"]:
                                found = True
                                break
                        if found and seg_id in st.session_state.classes[selected_class]["sample_ids"]:
                            st.session_state.classes[selected_class]["sample_ids"].remove(seg_id)
                            st.success(f"Segment ID: {seg_id} at ({col}, {row}) removed from {selected_class}.")
                        elif not found and seg_id not in st.session_state.classes[selected_class]["sample_ids"]:
                            st.session_state.classes[selected_class]["sample_ids"].append(seg_id)
                            st.success(f"Segment ID: {seg_id} at ({col}, {row}) added to {selected_class}.")
                        st.write(st.session_state.classes)
                        st.rerun()
    except Exception as e:
        st.error(f"error: {str(e)}")


def render_supervised_classification(index):
    """Render the classification window for applying supervised classification to segmentation layers."""
    try:
        process_data = st.session_state.processes[index]
        if "params" not in process_data:
            process_data["params"] = {}
        st.markdown("### Configure Training Classifier")
        if "classes" in st.session_state and st.session_state.classes:
            col1, col2, col3 = st.columns(3)
            with col1:
                layer_options = list(st.session_state.layers.keys())
                seg_layer_name = st.selectbox(
                    "Select segmentation layer for classification:",
                    options=layer_options,
                    index=layer_options.index(st.session_state.active_segmentation_layer_name),
                    key=f"seg_layer_{index}",
                )
            with col2:
                classifier_list = ["Random Forest"]
                selected_classifier = st.selectbox(
                    "Choose a classifier", options=classifier_list, index=0, key=f"select_classifier_{index}"
                )
            with col3:
                classification_name = st.text_input("Layer Name", "Supervised_Classification", key=f"classification_name_{index}")
            col1a, col1b, col1c = st.columns(3)
            if selected_classifier == "Random Forest":
                with col1a:
                    n_estimators = st.number_input(
                        "Number of Trees", min_value=10, max_value=1000, value=100, step=10, key=f"no_of_trees_{index}"
                    )
                with col1b:
                    boolean_options = ["True", "False"]
                    oob_score = st.selectbox("Use Out-of-Bag Score", options=boolean_options, key=f"oob_score_{index}")
                with col1c:
                    random_state = st.number_input(
                        "Random Seed", min_value=0, max_value=2**32 - 1, value=42, step=1, key=f"no_of_seed_{index}"
                    )
            features_options = (
                st.session_state.layers[seg_layer_name]
                .objects.drop(columns=["segment_id", "classification", "geometry"], errors="ignore")
                .columns
            )
            features = st.multiselect(
                "Select features ",
                features_options,
                # default=None,
                key=f"classifier_feat_{index}",
            )
            apply_button = st.button("Execute", key=f"execute_classification_{index}")
            if apply_button:
                classifier_params = {"n_estimators": n_estimators, "oob_score": bool(oob_score), "random_state": random_state}
                # seg_layer_name = st.session_state.active_segmentation_layer_name
                layer = st.session_state.layers[seg_layer_name]
                if classification_name in list(st.session_state.layers.keys()):
                    st.error("Layer name already exists")
                    st.stop()
                classification_layer, accuracy, feature_importances = perform_supervised_classification(
                    layer, selected_classifier, classifier_params, classification_name, features
                )
                if classification_layer:
                    class_color = {}
                    for key in list(st.session_state.classes.keys()):
                        class_color[key] = st.session_state.classes[key]["color"]
                    fig = plot_classification(classification_layer, class_field="classification", class_color=class_color)
                    process_data["output_fig"] = fig
                    process_data["accuracy"] = accuracy
                    process_data["feature_importances"] = feature_importances
                    # process_data["layer_type"] = "classification"
                    # st.session_state.classification_fig = fig  # Store the figure in session state
            if "accuracy" in process_data:
                st.write(f"OOB Score: {process_data['accuracy']}")
            if "output_fig" in process_data:
                st.pyplot(process_data["output_fig"])
            if "feature_importances" in process_data:
                fi = process_data["feature_importances"]
                fig, ax = plt.subplots(figsize=(6, 4))
                fi.plot.bar(ax=ax)
                ax.set_title("Importances")
                ax.set_xlabel("Features")
                ax.set_ylabel("Importance (%)")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.error("Error: Sample data are not created")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_supervised_classification_deeplearning(index):
    """Render the classification window for applying deep learning based supervised classification to segmentation layers."""
    try:
        process_data = st.session_state.processes[index]
        if "params" not in process_data:
            process_data["params"] = {}
        st.markdown("### Configure Model Training")
        if "classes" in st.session_state and st.session_state.classes:
            col1, col2, col3 = st.columns(3)
            with col1:
                layer_options = list(st.session_state.layers.keys())
                seg_layer_name = st.selectbox(
                    "Select segmentation layer for classification:",
                    options=layer_options,
                    index=layer_options.index(st.session_state.active_segmentation_layer_name),
                    key=f"seg_layer_{index}",
                )
            with col2:
                classifier_list = ["Convolution Neural Network (CNN)"]
                selected_classifier = st.selectbox(
                    "Choose a classifier", options=classifier_list, index=0, key=f"select_dl_classifier_{index}"
                )
            with col3:
                classification_name = st.text_input("Layer Name", "Supervised_Classification", key=f"dl_classification_name_{index}")
            col1a, col1b, col1c = st.columns(3)
            if selected_classifier == "Convolution Neural Network (CNN)":
                with col1a:
                    patch_size_list = [(5,5),(8,8),(16,16)]
                    patch_size = st.selectbox(
                    "Choose a patch size", options=patch_size_list, index=0, key=f"select_dl_patch_size_{index}"
                )
                    # patch_size = st.number_input(
                    #     "Patch Size", min_value=10, max_value=500, value=20, step=10, key=f"patch_size_{index}"
                    # )
                with col1b:
                   batch_size = st.number_input(
                        "Batch Size", min_value=8, max_value=256, value=16, step=16, key=f"batch_size_{index}"
                    )
                with col1c:
                    epochs = st.number_input(
                        "Epochs", min_value=10, max_value=1000, value=30, step=20, key=f"epoch_{index}"
                    )
            # features_options = (
            #     st.session_state.layers[seg_layer_name]
            #     .objects.drop(columns=["segment_id", "classification", "geometry"], errors="ignore")
            #     .columns
            # )
            # features = st.multiselect(
            #     "Select features ",
            #     features_options,
            #     # default=None,
            #     key=f"classifier_feat_{index}",
            # )

            apply_button = st.button("Execute", key=f"execute_dl_classification_{index}")
            if apply_button:
                classifier_params = {"patch_size": patch_size, "batch_size": batch_size, "epochs": epochs}
                # seg_layer_name = st.session_state.active_segmentation_layer_name

                layer = st.session_state.layers[seg_layer_name]
                image_data= st.session_state.image_data
                if classification_name in list(st.session_state.layers.keys()):
                    st.error("Layer name already exists")
                    st.stop()
                classification_layer, model_history, eval_resul = perform_supervised_classification_DL(
                    layer, image_data, selected_classifier, classifier_params, classification_name
                )
                if classification_layer:
                    class_color = {}
                    for key in list(st.session_state.classes.keys()):
                        class_color[key] = st.session_state.classes[key]["color"]
                    fig = plot_classification(classification_layer, class_field="classification", class_color=class_color)
                    # training_history_plt=plot_training_history(model_history)
                    process_data["model_history"] = model_history
                    process_data["output_fig"] = fig
                    # process_data["training_history_plt"] = training_history_plt
                    process_data["eval_result"] = eval_resul
            if "model_history" in process_data:
                training_history_plt=plot_training_history(process_data["model_history"])
                st.markdown("#### Model Training Accuracy")
                st.write("training_history_plt")
                st.pyplot(training_history_plt)
            if "eval_result" in process_data:
                st.markdown("#### Model Evaluation")
                st.write(f"Test Accuracy: {process_data['eval_result']['accuracy']:.3f}")
                cm=process_data["eval_result"]['confusion_matrix']
                class_names = st.session_state.classes.keys()
                df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
                df_cm.index.name = 'Predicted Label'
                st.markdown("#### Confusion Matrix")
                st.table(df_cm)
            if "output_fig" in process_data:
                st.pyplot(process_data["output_fig"])
        else:
            st.error("Error: Sample data are not created")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_rule_based_classification(index):
    """Render the rule based classification."""
    try:
        st.markdown("#### Apply Rule Sets")

        if not st.session_state.rule_sets:
            st.info("No rule sets defined yet. Go to Rule Builder tab to create rule sets.")
        else:
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                selected_ruleset = st.selectbox(
                    "Select rule set to apply:", options=list(st.session_state.rule_sets.keys()), key=f"select_ruleset_{index}"
                )

            with col2:
                input_layer = st.selectbox(
                    "Select input layer:", options=list(st.session_state.layers.keys()), key=f"input_layer_{index}"
                )

            with col3:
                result_field = st.text_input("Result field name:", "classification", key=f"result_field_{index}")

            if selected_ruleset and input_layer:
                col1a, _ = st.columns([2, 3])
                with col1a:
                    output_layer_name = st.text_input(
                        "Output layer name:", f"{selected_ruleset}_results", key=f"out_layer_name_{index}"
                    )

                if st.button("Apply Rule Set", key=f"apply_ruleset_{index}"):
                    ruleset = st.session_state.rule_sets[selected_ruleset]
                    input_layer_obj = st.session_state.layers[input_layer]
                    if output_layer_name in list(st.session_state.layers.keys()):
                        st.error("Output layer name already exists")
                        st.stop()
                    result_layer = apply_rule_set(ruleset, input_layer_obj, output_layer_name, result_field)

                    if result_layer:
                        fig = plot_classification(result_layer, class_field=result_field)
                        st.pyplot(fig)

                        area_stats = calculate_area_stats(result_layer, result_field)

                        if area_stats:
                            st.markdown("#### Area Statistics")
                            stats_data = []
                            for class_name, area in area_stats.get("class_areas", {}).items():
                                percentage = area_stats.get("class_percentages", {}).get(class_name, 0)
                                stats_data.append(
                                    {"Class": class_name, "Area (sq. units)": f"{area:.2f}", "Percentage": f"{percentage:.1f}%"}
                                )

                            st.table(stats_data)

                        st.success(f"Rule set '{selected_ruleset}' applied successfully to create layer '{output_layer_name}'!")

        st.markdown("#### Load Example Rule Sets")

        if st.button("Load Vegetation Classification Example", key=f"load_example_{index}"):
            has_ndvi = False
            for _layer_name, layer in st.session_state.layers.items():
                if "NDVI" in layer.objects.columns:
                    has_ndvi = True
                    break

            if not has_ndvi:
                st.warning("NDVI not calculated for any layer. Please calculate NDVI first in the Segmentation tab.")
            else:
                example_rule_sets = create_example_rule_sets()

                if example_rule_sets:
                    st.session_state.rule_sets.update(example_rule_sets)
                    st.success("Example rule sets loaded successfully!")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_rule_builder(index):
    """Render the rule builder tab for creating and managing classification rule sets."""
    try:
        st.header("Rule Builder")
        st.write("Create and manage classification rule sets and rules")

        st.markdown("#### Create Rule Set")

        new_ruleset_name = st.text_input("New Rule Set Name:", "", key=f"ruleset_name_{index}")
        if new_ruleset_name and st.button("Create New Rule Set"):
            if new_ruleset_name in st.session_state.rule_sets:
                st.warning(f"Rule set '{new_ruleset_name}' already exists.")
            else:
                with st.spinner("Creating new rule set..."):
                    st.session_state.rule_sets[new_ruleset_name] = RuleSet(name=new_ruleset_name)
                    st.session_state.active_ruleset = new_ruleset_name
                    st.success(f"Rule set '{new_ruleset_name}' created successfully!")

        st.markdown("#### Manage Rules")

        if not st.session_state.rule_sets:
            st.info("No rule sets created yet. Create a rule set first.")
        else:
            ruleset_selection = st.selectbox(
                "Select rule set to manage:", options=list(st.session_state.rule_sets.keys()), key=f"ruleset_select_{index}"
            )

            if ruleset_selection:
                st.session_state.active_ruleset = ruleset_selection
                active_ruleset = st.session_state.rule_sets[ruleset_selection]

                st.write(f"Rules in '{ruleset_selection}':")

                rules = active_ruleset.get_rules()
                if not rules:
                    st.info("No rules in this rule set yet.")
                else:
                    rule_data = []
                    for i, (name, condition) in enumerate(rules):
                        rule_data.append({"#": i + 1, "Name": name, "Condition": condition})

                    st.table(pd.DataFrame(rule_data))

                st.markdown("#### Add New Rule")

                if not st.session_state.available_attributes:
                    st.warning("No layers with attributes available. Create a segmentation layer with features first.")
                else:
                    st.write("Available attributes for rules:")
                    st.write(", ".join(sorted(st.session_state.available_attributes)))

                    col1, col2 = st.columns(2)

                    with col1:
                        rule_name = st.text_input("Rule Name:", "", key=f"rule_name_{index}")

                    st.markdown("#### Rule Condition Builder")

                    if "condition_builder" not in st.session_state:
                        st.session_state.condition_builder = []

                    if st.button("Add Condition Component", key=f"rule_add_{index}"):
                        st.session_state.condition_builder.append({"attribute": "", "operator": ">", "value": "", "connector": "&"})

                    condition_parts = []

                    for i, _condition in enumerate(st.session_state.condition_builder):
                        st.markdown("#### Condition Component {i + 1}")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            attribute = st.selectbox(
                                f"Attribute {i + 1}", options=sorted(st.session_state.available_attributes), key=f"attr_{index}_{i}"
                            )
                            st.session_state.condition_builder[i]["attribute"] = attribute

                        with col2:
                            operator = st.selectbox(
                                f"Operator {i + 1}", options=[">", ">=", "<", "<=", "==", "!="], key=f"op_{index}_{i}"
                            )
                            st.session_state.condition_builder[i]["operator"] = operator

                        with col3:
                            if operator == "==" or operator == "!=":
                                value = st.text_input(f"Value {i + 1} (use quotes for text)", key=f"val_{index}_{i}")
                            else:
                                value = st.number_input(f"Value {i + 1}", key=f"val_{index}_{i}", format="%.2f")
                            st.session_state.condition_builder[i]["value"] = value

                        condition_part = f"{attribute} {operator} {value}"
                        condition_parts.append(condition_part)

                        if i < len(st.session_state.condition_builder) - 1:
                            connector = st.selectbox("Connect with", options=["&", "|"], key=f"conn_{index}_{i}")
                            st.session_state.condition_builder[i]["connector"] = connector

                    if condition_parts:
                        final_condition = ""
                        for i, part in enumerate(condition_parts):
                            final_condition += part
                            if i < len(condition_parts) - 1:
                                final_condition += f" {st.session_state.condition_builder[i]['connector']} "

                        st.markdown("#### Final Condition:")
                        st.code(final_condition)

                        manual_condition = st.text_area("Or manually edit condition:", final_condition)

                        if manual_condition and st.button("Add Rule to Set", key=f"rule_apply_{index}"):
                            if not rule_name:
                                st.warning("Rule Name is required.")
                            else:
                                with st.spinner("Adding rule..."):
                                    active_ruleset.add_rule(name=rule_name, condition=manual_condition)
                                    st.session_state.condition_builder = []
                                    st.success(f"Rule '{rule_name}' added to rule set '{ruleset_selection}'!")
                                    st.rerun()

                    if st.button("Clear Condition Builder", key=f"rule_clear_{index}"):
                        st.session_state.condition_builder = []
                        st.rerun()

                st.markdown("#### Delete Rule Set")
                if st.button("Delete Current Rule Set", key=f"delete_ruleset_{index}"):
                    with st.spinner("Deleting rule set..."):
                        if st.session_state.active_ruleset in st.session_state.rule_sets:
                            del st.session_state.rule_sets[st.session_state.active_ruleset]
                            if st.session_state.rule_sets:
                                st.session_state.active_ruleset = list(st.session_state.rule_sets.keys())[0]
                            else:
                                st.session_state.active_ruleset = None
                            st.success(f"Rule set '{ruleset_selection}' deleted.")
                            st.rerun()
    except Exception as e:
        st.error(f"error: {str(e)}")


def render_process_tab():
    """Render the process tab for running available algorithms."""
    try:
        operation_list = [
            "",
            "Segmentation",
            "Add features",
            "Create Rule",
            "Rule-based Classification",
            "Select Sample Data",
            "Supervised Classification (ML)",
            "Supervised Classification (DL)",
            "Merge Region",
            "Find Enclosed by Class",
            "Touched_by",
        ]
        if "expanders" not in st.session_state:
            st.session_state.expanders = {}

        add_process_button = st.button("➕ Add process", key="add_process")
        if "processes" not in st.session_state:
            st.session_state.processes = []
        if add_process_button:
            st.session_state.processes.append(
                {
                    "id": len(st.session_state.processes),
                    "type": "",  # Default process type
                }
            )

        for i, process in enumerate(st.session_state.processes):
            key = f"expander_{i}"
            # Default: True (expanded) if not already tracked
            if key not in st.session_state.expanders:
                st.session_state.expanders[key] = True

            with st.expander(f"Process {i + 1}: {process['type']}", expanded=st.session_state.expanders[key]):
                col1, col2 = st.columns(2)
                with col1:
                    selected_operation = st.selectbox(
                        "Select  Operation",
                        operation_list,
                        index=operation_list.index(process["type"]) if process["type"] in operation_list else 0,
                        key=f"operation_{i}",
                    )
                st.session_state.processes[i]["type"] = selected_operation
                if selected_operation == "Segmentation":
                    render_segmentation(i)
                elif selected_operation == "Add features":
                    render_calculate_features(i)
                elif selected_operation == "Select Sample Data":
                    render_select_samples(i)
                elif selected_operation == "Rule-based Classification":
                    render_rule_based_classification(i)
                elif selected_operation == "Create Rule":
                    render_rule_builder(i)
                elif selected_operation == "Supervised Classification (ML)":
                    render_supervised_classification(i)
                elif selected_operation == "Supervised Classification (DL)":
                    render_supervised_classification_deeplearning(i)
                elif selected_operation == "Merge Region":
                    render_merge_regions(i)
                elif selected_operation == "Find Enclosed by Class":
                    render_enclosed_by_class(i)
                elif selected_operation == "Touched_by":
                    render_touched_by_class(i)

            if st.button("🗑️ Delete", key=f"delete_{i}"):
                st.session_state.delete_index = i
                st.rerun()

        # add_process_button = st.button("➕ Add process", key="add_process")
        # if "processes" not in st.session_state:
        #     st.session_state.processes = []
        # if add_process_button:
        #     st.session_state.processes.append(
        #         {
        #             "id": len(st.session_state.processes),
        #             "type": "",  # Default process type
        #         }
        #     )

        # st.write(st.session_state.processes)

        if st.session_state.delete_index is not None:
            del st.session_state.processes[st.session_state.delete_index]
            st.session_state.delete_index = None
            st.rerun()
    except Exception as e:
        st.error(f"error: {str(e)}")


def render_layer_manager_tab():
    """Render the layer manager tab for managing segmentation and classification layers."""
    st.header("Layer Manager")
    st.write("View, inspect, and manage layers")

    st.markdown("#### Available Layers")

    if not st.session_state.layers:
        st.info("No layers created yet.")
    else:
        layer_list = st.session_state.manager.get_layer_names()

        for i, layer_name in enumerate(layer_list):
            st.markdown(f"**{i + 1}. {layer_name}**")

            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                if st.button("View Info", key=f"info_{layer_name}_{i}"):
                    layer = st.session_state.layers.get(layer_name)
                    with st.spinner("Loading layer information..."):
                        if layer and hasattr(layer, "objects"):
                            st.write(f"Layer: {layer_name}")
                            st.write(f"Number of features: {len(layer.objects)}")
                            st.write("Attributes:")
                            attribute_list = [col for col in layer.objects.columns if col != "geometry"]
                            st.write(", ".join(attribute_list))

            with col2:
                if st.button("Visualize", key=f"vis_{layer_name}_{i}"):
                    layer = st.session_state.layers.get(layer_name)
                    with st.spinner("Generating visualization..."):
                        if layer:
                            for field in ["classification", "veg_class"]:
                                if field in layer.objects.columns:
                                    fig = plot_classification(layer, class_field=field)
                                    st.pyplot(fig)
                                    break
                            else:
                                fig = plot_layer(
                                    layer,
                                    st.session_state.image_data,
                                    rgb_bands=(2, 1, 0),  # Adjusted for 0-indexed bands
                                    show_boundaries=True,
                                )
                                st.pyplot(fig)

            with col3:
                if st.button("Export Options", key=f"export_{layer_name}_{i}"):
                    layer = st.session_state.layers.get(layer_name)
                    if layer:
                        st.session_state.export_layer = layer_name
                        st.info(f"Layer '{layer_name}' added to export queue. Go to Results tab to export.")

            with col4:
                if st.button("❌", key=f"del_{layer_name}_{i}"):
                    if layer_name in st.session_state.layers:
                        with st.spinner(f"Deleting layer '{layer_name}'..."):
                            del st.session_state.layers[layer_name]
                            if layer_name in layer_list:
                                st.session_state.manager.remove_layer(layer_name)
                            st.success(f"Layer '{layer_name}' deleted!")
                            update_available_attributes()
                            st.rerun()
            st.markdown("---")


def render_results_tab():
    """Render the results tab for exporting segmentation and classification results."""
    st.header("Results and Export")

    if not st.session_state.layers:
        st.warning("No results to display yet. Please run segmentation and classification first.")
    else:
        st.write("Export your results as vector (GeoJSON) or raster (GeoTIFF) files.")

        st.markdown("#### Available Layers")
        for i, layer_name in enumerate(st.session_state.manager.get_layer_names()):
            st.write(f"  {i + 1}. {layer_name}")

        st.markdown("#### Export Options")

        export_layer_name = st.selectbox(
            "Select Layer to Export",
            options=list(st.session_state.layers.keys()),
            format_func=lambda x: x.capitalize().replace("_", " "),
        )

        if export_layer_name:
            export_path = st.text_input("Export Filename Base (without extension):", export_layer_name.lower())

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export as Vector (GeoJSON)"):
                    export_filepath = os.path.join(st.session_state.output_dir, f"{export_path}.geojson")
                    success = export_vector(st.session_state.layers[export_layer_name], export_filepath)

                    if success:
                        with open(export_filepath, "rb") as f:
                            st.download_button(
                                label="Download GeoJSON", data=f, file_name=f"{export_path}.geojson", mime="application/geo+json"
                            )

            with col2:
                column_options = get_layer_attributes(export_layer_name)

                if column_options:
                    raster_column = st.selectbox("Select attribute for raster export:", column_options)

                    if st.button("Export as Raster (GeoTIFF)"):
                        export_filepath = os.path.join(st.session_state.output_dir, f"{export_path}.tif")
                        success = export_raster(st.session_state.layers[export_layer_name], export_filepath, raster_column)

                        if success:
                            with open(export_filepath, "rb") as f:
                                st.download_button(
                                    label="Download GeoTIFF", data=f, file_name=f"{export_path}.tif", mime="image/tiff"
                                )

            st.markdown("#### Project Export")

            if st.button("Export Project Metadata"):
                with st.spinner("Creating project metadata..."):
                    metadata = {
                        "project_name": "nickyspatial Project",
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "layers": list(st.session_state.manager.get_layer_names()),
                        "rule_sets": {},
                    }

                    for ruleset_name, ruleset in st.session_state.rule_sets.items():
                        metadata["rule_sets"][ruleset_name] = [
                            {"name": name, "condition": condition} for name, condition in ruleset.get_rules()
                        ]

                    import json

                    metadata_filepath = os.path.join(st.session_state.output_dir, "project_metadata.json")
                    with open(metadata_filepath, "w") as f:
                        json.dump(metadata, f, indent=4)

                    with open(metadata_filepath, "rb") as f:
                        st.download_button(
                            label="Download Project Metadata", data=f, file_name="project_metadata.json", mime="application/json"
                        )


def render_welcome_screen():
    """Render the welcome screen for the application."""
    st.write("""
    ## Welcome to the nickyspatial Interactive Web App

    This application allows you to perform object-based image analysis on remote sensing data with complete flexibility
    to configure all parameters, rules, and layers.

    To get started:
    1. Use the sidebar to upload your own raster data or use the sample data
    2. Configure segmentation parameters and create segments
    3. Calculate features like NDVI, spectral indices, or shape metrics
    4. Create rule sets and classification rules in the Rule Builder
    5. Apply your rules to classify the image
    6. Visualize and export your results

    ### Features:
    - Fully configurable image segmentation parameters
    - Dynamic creation of custom rule sets and rules
    - Flexible band mapping for spectral indices
    - Multiple layer management and visualization
    - Export results in vector and raster formats
    """)


def render_app_info():
    """Render the application information in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About nickyspatial**

    An open-source object-based image analysis library for remote sensing.

    [GitHub Repository](https://github.com/kshitijrajsharma/nickyspatial)
    """)


def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()

    st.title("nickyspatial - Object-Based Image Analysis")
    st.markdown("An interactive tool for remote sensing analysis and classification")

    st.sidebar.header("Configuration")

    st.sidebar.subheader("1. Load Data")
    data_option = st.sidebar.radio("Choose data source:", ("Upload your own", "Use sample data"))

    if data_option == "Upload your own":
        uploaded_file = st.sidebar.file_uploader("Upload a raster file (GeoTIFF format)", type=["tif", "tiff"])
        if uploaded_file is not None:
            temp_file = os.path.join(st.session_state.output_dir, "uploaded_file.tif")
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            _load_success = load_raster(temp_file)
    else:
        sample_data_path = st.sidebar.text_input(
            "Path to sample data:", "https://github.com/kshitijrajsharma/nickyspatial/raw/refs/heads/master/data/sample.tif"
        )
        if st.sidebar.button("Load Sample Data"):
            _load_success = load_raster(sample_data_path)

    if st.session_state.image_data is not None:
        st.sidebar.success("Image loaded successfully")
        st.sidebar.info(f"Image dimensions: {st.session_state.image_data.shape}")
        if st.session_state.crs:
            st.sidebar.info(f"CRS: {st.session_state.crs}")

        tabs = st.tabs(["Process", "Layer Manager", "Results"])

        # tabs = st.tabs(["Segmentation", "Classification", "Rule Builder", "Layer Manager", "Results"])

        # with tabs[0]:
        #     render_segmentation_tab()

        # with tabs[1]:
        #     render_classification_tab()

        # with tabs[2]:
        #     render_rule_builder_tab()
        with tabs[0]:
            render_process_tab()

        with tabs[1]:
            render_layer_manager_tab()

        with tabs[2]:
            render_results_tab()
    else:
        render_welcome_screen()

    render_app_info()


if __name__ == "__main__":
    main()
