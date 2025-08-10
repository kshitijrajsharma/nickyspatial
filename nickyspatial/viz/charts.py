# -*- coding: utf-8 -*-
"""Visualization functions for plotting histograms, statistics, and scatter plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_histogram(layer, attribute, bins=20, figsize=(10, 6), by_class=None):
    """Plot a histogram of attribute values.

    Parameters:
    -----------
    layer : Layer
        Layer containing data
    attribute : str
        Attribute to plot
    bins : int
        Number of bins
    figsize : tuple
        Figure size
    by_class : str, optional
        Column to group by (e.g., 'classification')

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if layer.objects is None or attribute not in layer.objects.columns:
        raise ValueError(f"Attribute '{attribute}' not found in layer objects")

    fig, ax = plt.subplots(figsize=figsize)

    if by_class and by_class in layer.objects.columns:
        data = layer.objects[[attribute, by_class]].copy()

        for class_value, group in data.groupby(by_class):
            if class_value is None:
                continue

            sns.histplot(group[attribute], bins=bins, alpha=0.6, label=str(class_value), ax=ax)

        ax.legend(title=by_class)
    else:
        sns.histplot(layer.objects[attribute], bins=bins, ax=ax)

    ax.set_title(f"Histogram of {attribute}")
    ax.set_xlabel(attribute)
    ax.set_ylabel("Count")

    return fig


def plot_statistics(layer, stats_dict, figsize=(12, 8), kind="bar", y_log=False):
    """Plot statistics from a statistics dictionary.

    Parameters:
    -----------
    layer : Layer
        Layer the statistics are calculated for
    stats_dict : dict
        Dictionary with statistics (from attach_* functions)
    figsize : tuple
        Figure size
    kind : str
        Plot type: 'bar', 'line', or 'pie'
    y_log : bool
        Whether to use logarithmic scale for y-axis

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    flat_stats = {}

    def _flatten_dict(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                _flatten_dict(value, f"{prefix}{key}_")
            else:
                flat_stats[f"{prefix}{key}"] = value

    _flatten_dict(stats_dict)

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "pie" and "class_percentages" in stats_dict:
        percentages = stats_dict["class_percentages"]
        values = list(percentages.values())
        labels = list(percentages.keys())

        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
        ax.axis("equal")
        ax.set_title("Class Distribution")

    elif kind == "pie" and "percentages" in flat_stats:
        percentages = pd.Series(flat_stats).filter(like="percentage")
        values = percentages.values
        labels = [label.replace("_percentage", "") for label in percentages.index]

        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
        ax.axis("equal")
        ax.set_title("Distribution")

    else:
        stats_df = pd.DataFrame({"Metric": list(flat_stats.keys()), "Value": list(flat_stats.values())})

        if kind != "line":
            stats_df = stats_df.sort_values("Value", ascending=False)

        if kind == "bar":
            sns.barplot(x="Metric", y="Value", data=stats_df, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        elif kind == "line":
            sns.lineplot(x="Metric", y="Value", data=stats_df, ax=ax, marker="o")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        if y_log:
            ax.set_yscale("log")

        ax.set_title("Statistics Summary")

    plt.tight_layout()
    return fig


def plot_scatter(layer, x_attribute, y_attribute, color_by=None, figsize=(10, 8)):
    """Create a scatter plot of two attributes.

    Parameters:
    -----------
    layer : Layer
        Layer containing data
    x_attribute : str
        Attribute for x-axis
    y_attribute : str
        Attribute for y-axis
    color_by : str, optional
        Attribute to color points by
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if layer.objects is None or x_attribute not in layer.objects.columns or y_attribute not in layer.objects.columns:
        raise ValueError("Attributes not found in layer objects")

    fig, ax = plt.subplots(figsize=figsize)

    if color_by and color_by in layer.objects.columns:
        scatter = ax.scatter(
            layer.objects[x_attribute],
            layer.objects[y_attribute],
            c=layer.objects[color_by],
            cmap="viridis",
            alpha=0.7,
            s=50,
            edgecolor="k",
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        ax.scatter(
            layer.objects[x_attribute],
            layer.objects[y_attribute],
            alpha=0.7,
            s=50,
            edgecolor="k",
        )

    ax.set_title(f"{y_attribute} vs {x_attribute}")
    ax.set_xlabel(x_attribute)
    ax.set_ylabel(y_attribute)
    ax.grid(alpha=0.3)

    return fig


def plot_training_history(history):
    """Plot (training and validation) loss and accuracy curves from a Keras(CNN) training history.

    This function visualizes the model's performance over epochs by plotting:
    - Training and validation loss
    - Training and validation accuracy

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned by `model.fit()`, containing loss and accuracy values
        for each epoch.

    Returns:
    -------
    matplotlib.pyplot
        The pyplot module with the generated figure, allowing further modification or saving.
    """
    epoches = np.arange(1, len(history.history.get("loss")) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15, 7))  # Removed sharey=True

    # Extract Loss Values
    train_loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")

    # Extract Accuracy Values (Scaled to Percentage)
    train_acc = [x * 100 for x in history.history.get("accuracy")]
    val_acc = [x * 100 for x in history.history.get("val_accuracy")]

    # --- Plot Training & Validation Loss ---
    ax1.plot(epoches, train_loss, "b", marker="o", label="Training Loss")
    ax1.plot(epoches, val_loss, "r", marker="o", label="Validation Loss")
    ax1.set_title("Training and Validation Loss", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.legend(fontsize=13)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)

    # Dynamically adjust the Y-Axis range for loss
    ax1.set_ylim(min(train_loss + val_loss) * 0.9, max(train_loss + val_loss) * 1.1)
    ax1.set_yticks(np.linspace(min(train_loss + val_loss), max(train_loss + val_loss), num=5))  # 5 evenly spaced ticks

    # --- Plot Training & Validation Accuracy ---
    ax2.plot(epoches, train_acc, "b", marker="o", label="Training Accuracy")
    ax2.plot(epoches, val_acc, "r", marker="o", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy", fontsize=16)
    ax2.set_ylabel("Accuracy (%)", fontsize=14)  # Accuracy in Percentage
    ax2.legend(fontsize=13)
    ax2.tick_params(axis="x", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    # Dynamically adjust the Y-Axis range for accuracy
    ax2.set_ylim(min(train_acc + val_acc) * 0.9, max(train_acc + val_acc) * 1.1)
    ax2.set_yticks(np.linspace(min(train_acc + val_acc), max(train_acc + val_acc), num=5))  # 5 evenly spaced ticks
    # plt.show()
    return plt
