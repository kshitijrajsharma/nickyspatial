# -*- coding: utf-8 -*-
"""Defines the Layer class and related functionality for organizing geospatial data.

A layer can represent a conceptual container for a vector object which is tightly coupled with underlying raster data, allowing
additional metadata or processing logic to be attached making sure heirarchical relationships are maintained.
This module provides the Layer and LayerManager classes, which manage layers of geospatial data, including segmentation results,
classifications,and filters. Layers can be created, copied, and manipulated, and they support attaching functions to calculate
additional properties.
"""

import uuid

import pandas as pd


class Layer:
    """A Layer represents a set of objects (segments or classification results) with associated properties.

    Layers can be derived from segmentation, rule application, or filters.
    Each layer can have functions attached to calculate additional properties.
    """

    def __init__(self, name=None, parent=None, type="generic"):
        """Initialize a Layer.

        Parameters:
        -----------
        name : str, optional
            Name of the layer. If None, a unique name will be generated.
        parent : Layer, optional
            Parent layer that this layer is derived from.
        type : str
            Type of layer: "segmentation", "classification", "filter", "generic" or "merged".
        """
        self.id = str(uuid.uuid4())
        self.name = name if name else f"Layer_{self.id[:8]}"
        self.parent = parent
        self.type = type
        self.created_at = pd.Timestamp.now()

        self.raster = None
        self.objects = None
        self.metadata = {}
        self.transform = None
        self.crs = None

        self.attached_functions = {}

    def attach_function(self, function, name=None, **kwargs):
        """Attach a function to this layer and execute it.

        Parameters:
        -----------
        function : callable
            Function to attach and execute
        name : str, optional
            Name for this function. If None, uses function.__name__
        **kwargs : dict
            Arguments to pass to the function

        Returns:
        --------
        self : Layer
            Returns self for chaining
        """
        func_name = name if name else function.__name__

        result = function(self, **kwargs)

        self.attached_functions[func_name] = {
            "function": function,
            "args": kwargs,
            "result": result,
        }

        return self

    def get_function_result(self, function_name):
        """Get the result of an attached function.

        Parameters:
        -----------
        function_name : str
            Name of the attached function

        Returns:
        --------
        result : any
            Result of the function
        """
        if function_name not in self.attached_functions:
            raise ValueError(f"Function '{function_name}' not attached to this layer")

        return self.attached_functions[function_name]["result"]

    def copy(self):
        """Create a copy of this layer.

        Returns:
        --------
        layer_copy : Layer
            Copy of this layer
        """
        new_layer = Layer(name=f"{self.name}_copy", parent=self.parent, type=self.type)

        if self.raster is not None:
            new_layer.raster = self.raster.copy()

        if self.objects is not None:
            new_layer.objects = self.objects.copy()

        new_layer.metadata = self.metadata.copy()
        new_layer.transform = self.transform
        new_layer.crs = self.crs

        return new_layer

    def __str__(self):
        """String representation of the layer."""
        if self.objects is not None:
            num_objects = len(self.objects)
        else:
            num_objects = 0

        parent_name = self.parent.name if self.parent else "None"

        return f"Layer '{self.name}' (type: {self.type}, parent: {parent_name}, objects: {num_objects})"


class LayerManager:
    """Manages a collection of layers and their relationships."""

    def __init__(self):
        """Initialize the layer manager."""
        self.layers = {}
        self.active_layer = None

    def add_layer(self, layer, set_active=True):
        """Add a layer to the manager.

        Parameters:
        -----------
        layer : Layer
            Layer to add
        set_active : bool
            Whether to set this layer as the active layer

        Returns:
        --------
        layer : Layer
            The added layer
        """
        self.layers[layer.id] = layer

        if set_active:
            self.active_layer = layer

        return layer

    def get_layer(self, layer_id_or_name):
        """Get a layer by ID or name.

        Parameters:
        -----------
        layer_id_or_name : str
            Layer ID or name

        Returns:
        --------
        layer : Layer
            The requested layer
        """
        if layer_id_or_name in self.layers:
            return self.layers[layer_id_or_name]

        for layer in self.layers.values():
            if layer.name == layer_id_or_name:
                return layer

        raise ValueError(f"Layer '{layer_id_or_name}' not found")

    def get_layer_names(self):
        """Get a list of all layer names.

        Returns:
        --------
        names : list
            List of layer names
        """
        return [layer.name for layer in self.layers.values()]

    def remove_layer(self, layer_id_or_name):
        """Remove a layer from the manager.

        Parameters:
        -----------
        layer_id_or_name : str
            Layer ID or name
        """
        layer = self.get_layer(layer_id_or_name)

        if layer.id in self.layers:
            del self.layers[layer.id]

        if self.active_layer and self.active_layer.id == layer.id:
            if self.layers:
                self.active_layer = list(self.layers.values())[-1]
            else:
                self.active_layer = None
