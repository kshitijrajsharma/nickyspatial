# -*- coding: utf-8 -*-
"""Layer class and related functionality for organizing geospatial data."""

import uuid

import pandas as pd


class Layer:
    """A Layer represents a set of objects with associated properties."""

    def __init__(self, name=None, parent=None, layer_type="generic"):
        """Initialize Layer with unique ID and optional name/parent."""
        self.id = str(uuid.uuid4())
        self.name = name if name else f"Layer_{self.id[:8]}"
        self.parent = parent
        self.type = layer_type
        self.created_at = pd.Timestamp.now()

        self.raster = None
        self.objects = None
        self.metadata = {}
        self.transform = None
        self.crs = None

        self.attached_functions = {}

    def attach_function(self, function, name=None, **kwargs):
        """Attach and execute function, store result for later retrieval."""
        func_name = name if name else function.__name__
        result = function(self, **kwargs)
        self.attached_functions[func_name] = {
            "function": function,
            "args": kwargs,
            "result": result,
        }
        return self

    def get_function_result(self, function_name):
        """Retrieve stored result from previously attached function."""
        if function_name not in self.attached_functions:
            raise ValueError(f"Function '{function_name}' not attached to this layer")
        return self.attached_functions[function_name]["result"]

    def copy(self):
        """Create independent copy with deep-copied raster/objects data."""
        new_layer = Layer(name=f"{self.name}_copy", parent=self.parent, layer_type=self.type)
        if self.raster is not None:
            new_layer.raster = self.raster.copy()
        if self.objects is not None:
            new_layer.objects = self.objects.copy()
        new_layer.metadata = self.metadata.copy()
        new_layer.transform = self.transform
        new_layer.crs = self.crs
        return new_layer

    def __str__(self):
        """Return layer info: name, type, and data availability."""
        num_objects = len(self.objects) if self.objects is not None else 0
        parent_name = self.parent.name if self.parent else "None"
        return f"Layer '{self.name}' (type: {self.type}, parent: {parent_name}, objects: {num_objects})"


class LayerManager:
    """Manages a collection of layers and their relationships."""

    def __init__(self):
        """Initialize empty layer manager with no active layer."""
        self.layers = {}
        self.active_layer = None

    def add_layer(self, layer, set_active=True):
        """Add layer to manager, optionally set as active layer."""
        self.layers[layer.id] = layer
        if set_active:
            self.active_layer = layer
        return layer

    def get_layer(self, layer_id_or_name):
        """Find layer by ID first, then by name if not found."""
        if layer_id_or_name in self.layers:
            return self.layers[layer_id_or_name]
        for layer in self.layers.values():
            if layer.name == layer_id_or_name:
                return layer
        raise ValueError(f"Layer '{layer_id_or_name}' not found")

    def get_layer_names(self):
        """Get a list of all layer names."""
        return [layer.name for layer in self.layers.values()]

    def remove_layer(self, layer_id_or_name):
        """Remove a layer from the manager."""
        layer = self.get_layer(layer_id_or_name)

        if layer.id in self.layers:
            del self.layers[layer.id]

        if self.active_layer and self.active_layer.id == layer.id:
            if self.layers:
                self.active_layer = list(self.layers.values())[-1]
            else:
                self.active_layer = None
