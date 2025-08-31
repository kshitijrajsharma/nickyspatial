# -*- coding: utf-8 -*-
"""Testing Working Document!

Just a workspace document to test the functionality of the library.
"""

import os

from nickyspatial import (
    LayerManager,
    RuleSet,
    SlicSegmentation,
    attach_area_stats,
    attach_ndvi,
    attach_shape_metrics,
    attach_spectral_indices,
    layer_to_raster,
    layer_to_vector,
    plot_classification,
    plot_layer,
    read_raster,
)


def run_example(raster_path=None):
    """Run Example."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    manager = LayerManager()

    if raster_path and os.path.exists(raster_path):
        print(f"Reading raster data from {raster_path}...")
        image_data, transform, crs = read_raster(raster_path)
        print(f"Image dimensions: {image_data.shape}")
        print(f"Coordinate system: {crs}")
    else:
        raise ValueError(f"Raster file not found at {raster_path}. Please provide a valid raster file.")
    print("\nPerforming segmentation...")
    segmenter = SlicSegmentation(
        scale=40,
        compactness=1,
    )

    segmentation_layer = segmenter.execute(
        image_data,
        transform,
        crs,
        layer_manager=manager,
        layer_name="Base_Segmentation",
    )
    print(segmentation_layer)

    fig1 = plot_layer(segmentation_layer, image_data, rgb_bands=(3, 2, 1), show_boundaries=True)
    fig1.savefig(os.path.join(output_dir, "1_segmentation.png"))

    print("\nCalculating spectral indices...")

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
    fig2.savefig(os.path.join(output_dir, "2_ndvi.png"))

    print("\nCalculating shape metrics...")

    segmentation_layer.attach_function(attach_shape_metrics, name="shape_metrics")

    layer_to_vector(segmentation_layer, os.path.join(output_dir, "segmentation.geojson"))

    print("\nApplying land cover classification rules...")

    land_cover_rules = RuleSet(name="Land_Cover")

    land_cover_rules.add_rule(name="Vegetation", condition="NDVI > 0.2")
    land_cover_rules.add_rule(name="Other", condition="NDVI <= 0.2")

    land_cover_layer = land_cover_rules.execute(segmentation_layer, layer_manager=manager, layer_name="Land_Cover")

    fig3 = plot_classification(land_cover_layer, class_field="classification")
    fig3.savefig(os.path.join(output_dir, "3_land_cover.png"))

    print("\nCalculating class statistics...")

    land_cover_layer.attach_function(attach_area_stats, name="area_by_class", by_class="classification")

    area_stats = land_cover_layer.get_function_result("area_by_class")

    print("\nLand cover area statistics:")
    for class_name, area in area_stats.get("class_areas", {}).items():
        percentage = area_stats.get("class_percentages", {}).get(class_name, 0)
        print(f"  {class_name}: {area:.2f} sq. units ({percentage:.1f}%)")

    print("\nApplying hierarchical classification rules...")

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

    fig4 = plot_classification(vegetation_layer, class_field="veg_class")
    fig4.savefig(os.path.join(output_dir, "4_vegetation_types.png"))

    print("\nExporting results...")

    layer_to_vector(land_cover_layer, os.path.join(output_dir, "land_cover.geojson"))
    layer_to_vector(vegetation_layer, os.path.join(output_dir, "vegetation_types.geojson"))

    layer_to_raster(
        land_cover_layer,
        os.path.join(output_dir, "land_cover.tif"),
        column="classification",
    )

    print(f"\nResults saved to {output_dir}")
    print("Available layers:")
    for i, layer_name in enumerate(manager.get_layer_names()):
        print(f"  {i + 1}. {layer_name}")


if __name__ == "__main__":
    run_example("data/sample.tif")
