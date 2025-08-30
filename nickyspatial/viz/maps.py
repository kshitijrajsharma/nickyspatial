# -*- coding: utf-8 -*-
"""Functions to create maps and visualize layers."""

import random

import ipywidgets as widgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from matplotlib.colors import ListedColormap
from skimage.segmentation import mark_boundaries


def plot_layer(
    layer,
    image_data=None,
    attribute=None,
    title=None,
    rgb_bands=(2, 1, 0),
    figsize=(12, 10),
    cmap="viridis",
    show_boundaries=False,
):
    """Plot a layer, optionally with an attribute or image backdrop."""
    fig, ax = plt.subplots(figsize=figsize)

    if title:
        ax.set_title(title)
    elif attribute:
        ax.set_title(f"{attribute} by Segment")
    else:
        ax.set_title("Layer Visualization")

    if image_data is not None:
        num_bands = image_data.shape[0]
        if num_bands >= 3 and max(rgb_bands) < num_bands:
            r = image_data[rgb_bands[0]]
            g = image_data[rgb_bands[1]]
            b = image_data[rgb_bands[2]]

            r_norm = np.clip((r - r.min()) / (r.max() - r.min() + 1e-10), 0, 1)
            g_norm = np.clip((g - g.min()) / (g.max() - g.min() + 1e-10), 0, 1)
            b_norm = np.clip((b - b.min()) / (b.max() - b.min() + 1e-10), 0, 1)

            rgb = np.stack([r_norm, g_norm, b_norm], axis=2)

            ax.imshow(rgb)
        else:
            gray = image_data[0]
            gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
            ax.imshow(gray_norm, cmap="gray")

    if attribute and attribute in layer.objects.columns:
        layer.objects.plot(
            column=attribute,
            cmap=cmap,
            ax=ax,
            legend=True,
            alpha=0.7 if image_data is not None else 1.0,
        )

    if show_boundaries and layer.raster is not None:
        from skimage.segmentation import mark_boundaries

        if image_data is not None:
            if "num_bands" in locals() and num_bands >= 3:
                base_img = rgb
            else:
                gray = image_data[0]
                gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
                base_img = np.stack([gray_norm, gray_norm, gray_norm], axis=2)

            bounded = mark_boundaries(base_img, layer.raster, color=(1, 1, 0), mode="thick")

            if attribute is None:
                ax.imshow(bounded)
        else:
            ax.imshow(
                mark_boundaries(
                    np.zeros((layer.raster.shape[0], layer.raster.shape[1], 3)),
                    layer.raster,
                    color=(1, 1, 0),
                    mode="thick",
                )
            )

    ax.grid(alpha=0.3)
    return fig


def plot_layer_interactive(layer, image_data=None, figsize=(10, 8)):
    """Interactive plot of a layer with widgets and working click."""
    """
    %matplotlib widget
    plot_layer_interactive(layer=segmentation_layer,image_data=image_data,figsize=(10,8))
    Not supported in google collab
    """
    title_widget = widgets.Text(value="Layer Visualization", description="Title:")

    rgb_band_max = image_data.shape[0] - 1 if image_data is not None else 2

    red_band_widget = widgets.Select(
        options=list(range(rgb_band_max + 1)), value=0 if rgb_band_max >= 2 else 0, description="Red Band:"
    )
    green_band_widget = widgets.Select(
        options=list(range(rgb_band_max + 1)), value=1 if rgb_band_max >= 2 else 0, description="Green Band:"
    )
    blue_band_widget = widgets.Select(
        options=list(range(rgb_band_max + 1)), value=2 if rgb_band_max >= 2 else 0, description="Blue Band:"
    )

    show_boundaries_widget = widgets.Checkbox(value=True, description="Show Boundaries")

    fig, ax = plt.subplots(figsize=figsize)
    out_fig = widgets.Output()

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x_pix, y_pix = int(event.xdata), int(event.ydata)
        if (0 <= x_pix < layer.raster.shape[1]) and (0 <= y_pix < layer.raster.shape[0]):
            segment_id = layer.raster[y_pix, x_pix]
            msg = f"Clicked at (x={x_pix}, y={y_pix}) → Segment ID: {segment_id}"
            title_widget.value = msg
            ax.set_title(msg)
            fig.canvas.draw_idle()
        else:
            msg = "Clicked outside raster bounds"
            title_widget.value = msg
            ax.set_title(msg)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)

    def update_plot(red_band, green_band, blue_band, show_boundaries):
        ax.clear()

        if image_data is not None:
            num_bands = image_data.shape[0]
            if red_band >= 0 and green_band >= 0 and blue_band >= 0:
                r = image_data[red_band].astype(float)
                g = image_data[green_band].astype(float)
                b = image_data[blue_band].astype(float)

                r_norm = np.clip((r - r.min()) / (r.max() - r.min() + 1e-10), 0, 1)
                g_norm = np.clip((g - g.min()) / (g.max() - g.min() + 1e-10), 0, 1)
                b_norm = np.clip((b - b.min()) / (b.max() - b.min() + 1e-10), 0, 1)

                rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
                ax.imshow(rgb)
            else:
                gray = image_data[0]
                gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
                ax.imshow(gray_norm, cmap="gray")

        if show_boundaries and layer.raster is not None:
            if image_data is not None:
                if num_bands >= 3:
                    base_img = rgb
                else:
                    gray = image_data[0]
                    gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
                    base_img = np.stack([gray_norm, gray_norm, gray_norm], axis=2)

                bounded = mark_boundaries(base_img, layer.raster, color=(1, 1, 0), mode="thick")
                ax.imshow(bounded)
            else:
                ax.imshow(
                    mark_boundaries(
                        np.zeros((layer.raster.shape[0], layer.raster.shape[1], 3)),
                        layer.raster,
                        color=(1, 1, 0),
                        mode="thick",
                    )
                )

        ax.grid(alpha=0.3)
        fig.canvas.draw_idle()

    ui = widgets.VBox(
        [
            red_band_widget,
            green_band_widget,
            blue_band_widget,
            show_boundaries_widget,
        ]
    )

    controls = widgets.interactive_output(
        update_plot,
        {
            "red_band": red_band_widget,
            "green_band": green_band_widget,
            "blue_band": blue_band_widget,
            "show_boundaries": show_boundaries_widget,
        },
    )

    display(ui, out_fig, controls)


def plot_classification(layer, class_field="classification", figsize=(12, 10), legend=True, class_color=None):
    """Plot classified segments with different colors for each class."""
    fig, ax = plt.subplots(figsize=figsize)
    if not class_color:
        class_color = {}

    if class_field not in layer.objects.columns:
        raise ValueError(f"Class field '{class_field}' not found in layer objects")

    class_values = [v for v in layer.objects[class_field].unique() if v is not None]

    base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(class_values), 1)))

    colors_list = []
    for idx, class_value in enumerate(class_values):
        if class_color and class_value in list(class_color.keys()):
            color_hex = class_color[class_value]
        else:
            if idx < len(base_colors):
                rgb = base_colors[idx][:3]
                color_hex = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            else:
                color_hex = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            class_color[class_value] = color_hex

        rgb_tuple = tuple(int(color_hex[i : i + 2], 16) / 255 for i in (1, 3, 5))
        colors_list.append(rgb_tuple)

    cmap = ListedColormap(colors_list)

    class_map = {value: i for i, value in enumerate(class_values)}
    layer.objects["_class_id"] = layer.objects[class_field].map(class_map)

    layer.objects.plot(
        column="_class_id",
        cmap=cmap,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    if legend and len(class_values) > 0:
        patches = [mpatches.Patch(color=class_color[value], label=value) for value in class_values]
        ax.legend(handles=patches, loc="upper right", title=class_field)

    ax.set_title("Classification Map")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    if "_class_id" in layer.objects.columns:
        layer.objects = layer.objects.drop(columns=["_class_id"])

    return fig


def plot_comparison(
    before_layer,
    after_layer,
    attribute=None,
    class_field=None,
    figsize=(16, 8),
    title=None,
):
    """Plot before and after views of layers for comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if title:
        fig.suptitle(title)

    if attribute and attribute in before_layer.objects.columns:
        before_layer.objects.plot(column=attribute, ax=ax1, legend=True)
        ax1.set_title(f"Before: {attribute}")
    elif class_field and class_field in before_layer.objects.columns:
        class_values = [v for v in before_layer.objects[class_field].unique() if v is not None]
        num_classes = len(class_values)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_classes, 1)))
        cmap = ListedColormap(colors)
        class_map = {value: i for i, value in enumerate(class_values)}
        before_layer.objects["_class_id"] = before_layer.objects[class_field].map(class_map)

        before_layer.objects.plot(
            column="_class_id",
            cmap=cmap,
            ax=ax1,
            edgecolor="black",
            linewidth=0.5,
            legend=False,
        )

        patches = [mpatches.Patch(color=colors[i], label=value) for i, value in enumerate(class_values)]
        ax1.legend(handles=patches, loc="upper right", title=class_field)
        ax1.set_title(f"Before: {class_field}")
    else:
        before_layer.objects.plot(ax=ax1)
        ax1.set_title("Before")

    if attribute and attribute in after_layer.objects.columns:
        after_layer.objects.plot(column=attribute, ax=ax2, legend=True)
        ax2.set_title(f"After: {attribute}")
    elif class_field and class_field in after_layer.objects.columns:
        class_values = [v for v in after_layer.objects[class_field].unique() if v is not None]
        num_classes = len(class_values)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_classes, 1)))
        cmap = ListedColormap(colors)
        class_map = {value: i for i, value in enumerate(class_values)}
        after_layer.objects["_class_id"] = after_layer.objects[class_field].map(class_map)

        after_layer.objects.plot(
            column="_class_id",
            cmap=cmap,
            ax=ax2,
            edgecolor="black",
            linewidth=0.5,
            legend=False,
        )

        patches = [mpatches.Patch(color=colors[i], label=value) for i, value in enumerate(class_values)]
        ax2.legend(handles=patches, loc="upper right", title=class_field)
        ax2.set_title(f"After: {class_field}")
    else:
        after_layer.objects.plot(ax=ax2)
        ax2.set_title("After")

    if "_class_id" in before_layer.objects.columns:
        before_layer.objects = before_layer.objects.drop(columns=["_class_id"])
    if "_class_id" in after_layer.objects.columns:
        after_layer.objects = after_layer.objects.drop(columns=["_class_id"])

    return fig


def plot_sample(
    layer,
    image_data=None,
    transform=None,
    rgb_bands=None,
    class_field="classification",
    figsize=(8, 6),
    class_color=None,
    legend=True,
):
    """Plot classified segments on top of RGB or grayscale image data.

    Parameters:
    - layer: Layer object with .objects (GeoDataFrame)
    - image_data: 3D numpy array (bands, height, width)
    - transform: Affine transform for the image (needed to compute extent)
    - red_band, green_band, blue_band: indices for RGB bands
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ---- Plot RGB or Grayscale image from array ----
    if image_data is not None:
        num_bands = image_data.shape[0]
        if rgb_bands and num_bands >= 3:
            r = image_data[rgb_bands[0]].astype(float)
            g = image_data[rgb_bands[1]].astype(float)
            b = image_data[rgb_bands[2]].astype(float)

            r_norm = np.clip((r - r.min()) / (r.max() - r.min() + 1e-10), 0, 1)
            g_norm = np.clip((g - g.min()) / (g.max() - g.min() + 1e-10), 0, 1)
            b_norm = np.clip((b - b.min()) / (b.max() - b.min() + 1e-10), 0, 1)

            rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
            if transform:
                from rasterio.plot import plotting_extent

                extent = plotting_extent(image_data[0], transform=transform)
                ax.imshow(rgb, extent=extent)
            else:
                ax.imshow(rgb)
        else:
            gray = image_data[0]
            gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
            if transform:
                from rasterio.plot import plotting_extent

                extent = plotting_extent(gray, transform=transform)
                ax.imshow(gray_norm, cmap="gray", extent=extent)
            else:
                ax.imshow(gray_norm, cmap="gray")

    # ---- Plot classification overlay ----
    gdf = layer.objects.copy()
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS")

    if not class_color:
        class_color = {}
    if class_field not in gdf.columns:
        raise ValueError(f"Class field '{class_field}' not found")

    class_values = [v for v in gdf[class_field].unique() if v is not None]
    base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(class_values), 1)))
    class_map = {}

    for idx, class_value in enumerate(class_values):
        if class_value in class_color:
            color_hex = class_color[class_value]
        else:
            rgb_val = base_colors[idx % len(base_colors)][:3]
            color_hex = "#{:02x}{:02x}{:02x}".format(int(rgb_val[0] * 255), int(rgb_val[1] * 255), int(rgb_val[2] * 255))
            class_color[class_value] = color_hex
        class_map[class_value] = color_hex

    for class_value in class_values:
        gdf[gdf[class_field] == class_value].plot(ax=ax, facecolor=class_map[class_value], edgecolor="black", linewidth=0.5)

    if legend:
        handles = [mpatches.Patch(color=class_map[val], label=val) for val in class_values]
        ax.legend(handles=handles, loc="upper right", title=class_field)

    ax.set_title("Sample Data Visualization")
    ax.set_axis_off()
    return fig


def plot_layer_interactive_plotly(layer, image_data, rgb_bands=(0, 1, 2), show_boundaries=True, figsize=(800, 400)):
    """Display an interactive RGB image with segment boundaries and hoverable segment IDs using Plotly.

    Run in google collab as well.

    Parameters:
    ----------
    layer : object
        An object with a `.raster` attribute representing the labeled segmentation layer
        (e.g., output from a segmentation algorithm, such as SLIC).
    image_data : image data to be visualized.
    rgb_bands : tuple of int, optional
        Tuple of three integers specifying which bands to use for the RGB composite (default is (0, 1, 2)).
    show_boundaries : bool, optional
        Whether to overlay the segment boundaries on the RGB image (default is True).
    figsize : tuple of int, optional
        Tuple specifying the width and height of the interactive Plotly figure in pixels (default is (800, 400)).

    Returns:
    -------
    None
        The function displays the interactive plot directly in the output cell in a Jupyter Notebook.

    Notes:
    -----
    - Segment boundaries are drawn using `skimage.segmentation.mark_boundaries`.
    - Hovering over the image displays the segment ID from `layer.raster`.

    """

    def get_rgb_image(r, g, b):
        r_norm = np.clip((r - r.min()) / (r.max() - r.min() + 1e-10), 0, 1)
        g_norm = np.clip((g - g.min()) / (g.max() - g.min() + 1e-10), 0, 1)
        b_norm = np.clip((b - b.min()) / (b.max() - b.min() + 1e-10), 0, 1)
        return np.stack([r_norm, g_norm, b_norm], axis=2)

    def update_plot(rgb_bands, show_boundaries=True):
        rgb_image = get_rgb_image(image_data[rgb_bands[0]], image_data[rgb_bands[1]], image_data[rgb_bands[2]])

        if show_boundaries:
            rgb_image = mark_boundaries(rgb_image, layer.raster, color=(1, 1, 0), mode="thick")

        fig = go.Figure(data=go.Image(z=(rgb_image * 255).astype(np.uint8)))

        fig.add_trace(
            go.Heatmap(
                z=layer.raster,
                opacity=0,
                hoverinfo="z",
                showscale=False,
                hovertemplate="Segment ID: %{z}<extra></extra>",
                colorscale="gray",
            )
        )

        fig.update_layout(
            title="Hover to see Segment ID", dragmode="pan", margin=dict(l=0, r=0, t=30, b=0), height=figsize[1], width=figsize[0]
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, scaleanchor="x")

        fig.show()

    update_plot(rgb_bands=rgb_bands, show_boundaries=show_boundaries)
