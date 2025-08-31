# -*- coding: utf-8 -*-
"""Provides a rule engine for object-based analysis, where segments or layers are processed according to custom logic.

Main idea here is to allow encode expert rules that can be applied to object segments which are layers in a nickyspatial context.
So rules are tied up to the layers , they can be attached or revoked or executed multiple items

Developers can define domain-specific rules to classify or merge features based on attributes.
This module includes the Rule and RuleSet classes, which allow users to create, manage, and apply rules to layers.
The RuleSet class can be used to group multiple rules together, and the execute method applies these rules to a given layer.
The rules can be defined using string expressions that can be evaluated using the numexpr library for performance.
"""

import numexpr as ne
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union

from .layer import Layer


class Rule:
    """A rule defines a condition to classify segments."""

    def __init__(self, name, condition, class_value=None):
        """Initialize a rule.

        Parameters:
        -----------
        name : str
            Name of the rule
        condition : str
            Condition as a string expression that can be evaluated using numexpr
        class_value : str, optional
            Value to assign when the condition is met.
            If None, uses the rule name.
        """
        self.name = name
        self.condition = condition
        self.class_value = class_value if class_value is not None else name

    def __str__(self):
        """String representation of the rule."""
        return f"Rule '{self.name}': {self.condition} -> {self.class_value}"


class RuleSet:
    """A collection of rules to apply to a layer."""

    def __init__(self, name=None):
        """Initialize a rule set.

        Parameters:
        -----------
        name : str, optional
            Name of the rule set
        """
        self.name = name if name else "RuleSet"
        self.rules = []

    @staticmethod
    def wrap_condition_parts_simple(self, condition):
        """Wrap condition parts with parentheses for evaluation."""
        parts = condition.split("&")
        parts = [f"({part.strip()})" for part in parts]
        return " & ".join(parts)

    def add_rule(self, name, condition, class_value=None):
        """Add a rule to the rule set.

        Parameters:
        -----------
        name : str
            Name of the rule
        condition : str
            Condition as a string expression that can be evaluated using numexpr
        class_value : str, optional
            Value to assign when the condition is met

        Returns:
        --------
        rule : Rule
            The added rule
        """
        rule = Rule(name, condition, class_value)
        self.rules.append(rule)
        return rule

    def get_rules(self):
        """Get the list of rules in the rule set.

        Returns:
        --------
        list of tuples
            List of (name, condition) tuples for each rule
        """
        return [(rule.name, rule.condition) for rule in self.rules]

    def execute(
        self,
        source_layer,
        layer_manager=None,
        layer_name=None,
        result_field="classification",
    ):
        """Apply rules to classify segments in a layer.

        Parameters:
        -----------
        source_layer : Layer
            Source layer with segments to classify
        layer_manager : LayerManager, optional
            Layer manager to add the result layer to
        layer_name : str, optional
            Name for the result layer
        result_field : str
            Field name to store classification results

        Returns:
        --------
        result_layer : Layer
            Layer with classification results
        """
        if not layer_name:
            layer_name = f"{source_layer.name}_{self.name}"

        result_layer = Layer(name=layer_name, parent=source_layer, layer_type="classification")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

        result_layer.objects = source_layer.objects.copy()

        if result_field not in result_layer.objects.columns:
            result_layer.objects[result_field] = None

        result_layer.metadata = {
            "ruleset_name": self.name,
            "rules": [
                {
                    "name": rule.name,
                    "condition": rule.condition,
                    "class_value": rule.class_value,
                }
                for rule in self.rules
            ],
            "result_field": result_field,
        }

        for rule in self.rules:
            try:
                if result_field in result_layer.objects.columns and (
                    f"{result_field} ==" in rule.condition
                    or f"{result_field}==" in rule.condition
                    or f"{result_field} !=" in rule.condition
                    or f"{result_field}!=" in rule.condition
                ):
                    eval_condition = rule.condition.replace("&", " and ").replace("|", " or ")

                    mask = result_layer.objects.apply(
                        lambda row, cond=eval_condition: eval(
                            cond,
                            {"__builtins__": {}},
                            {col: row[col] for col in result_layer.objects.columns if col != "geometry"},
                        ),
                        axis=1,
                    )

                else:
                    try:
                        local_dict = {
                            col: result_layer.objects[col].values for col in result_layer.objects.columns if col != "geometry"
                        }

                        mask = ne.evaluate(rule.condition, local_dict=local_dict)
                        mask = pd.Series(mask, index=result_layer.objects.index).fillna(False)
                    except Exception:
                        mask = result_layer.objects.eval(rule.condition, engine="python")

                result_layer.objects.loc[mask, result_field] = rule.class_value

            except Exception as e:
                print(f"Error applying rule '{rule.name}': {str(e)}")
                continue

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer


class CommonBase:
    """A shared utility base class for spatial rule sets.

    This class provides common methods used by multiple rule sets
    to preprocess layer data and determine spatial relationships
    between segments.
    """

    @staticmethod
    def _preprocess_layer(layer, class_column_name):
        """Prepare geometry and class maps from a spatial layer.

        Parameters:
        -----------
        layer : Layer
            The spatial layer containing objects with segment geometry and class labels.
        class_column_name : str
            The column name that stores class values (e.g., "veg_class", "land_use").

        Returns:
        --------
        geom_map : dict
            A dictionary mapping segment IDs to shapely geometry objects.
        class_map : dict
            A dictionary mapping segment IDs to their respective class values.
        """
        df = layer.objects
        geom_map = {sid: shape(geom) for sid, geom in zip(df["segment_id"], df["geometry"], strict=False)}
        class_map = dict(zip(df["segment_id"], df[class_column_name], strict=False))
        return geom_map, class_map

    @staticmethod
    def _find_neighbors(segment_id, geom_map):
        """Find neighboring segments based on spatial intersection.

        Parameters:
        -----------
        segment_id : int or str
            The ID of the segment whose neighbors are to be found.
        geom_map : dict
            A dictionary mapping segment IDs to shapely geometry objects.

        Returns:
        --------
        neighbors : list
            A list of segment IDs that intersect with the given segment.
        """
        segment_geom = geom_map[segment_id]
        neighbors = []
        for other_id, other_geom in geom_map.items():
            if other_id != segment_id and segment_geom.intersects(other_geom):
                neighbors.append(other_id)
        return neighbors


class MergeRuleSet(CommonBase):
    """A rule set for merging segments of the same class based on specified class values."""

    def __init__(self, name=None):
        """Initialize the merge rule set.

        Parameters:
        -----------
        name : str, optional
            Name of the merge rule set
        """
        self.name = name if name else "MergeRuleSet"

    def execute(self, source_layer, class_column_name, class_value, layer_manager=None, layer_name=None):
        """Merge segments of the same class in a layer.

        Parameters:
        -----------
        source_layer : Layer
            Source layer with segments to merge
        class_value : str or list of str
            One or more attribute field names to group and merge segments
        layer_manager : LayerManager, optional
            Layer manager to add the result layer to
        layer_name : str, optional
            Name for the result layer

        Returns:
        --------
        result_layer : Layer
            Layer with merged geometries
        """
        if not layer_name:
            layer_name = f"{source_layer.name}_{self.name}"

        result_layer = Layer(name=layer_name, parent=source_layer, layer_type="merged")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

        df = source_layer.objects.copy()

        if isinstance(class_value, str):
            class_values = [class_value]
        else:
            class_values = class_value

        new_rows = []
        to_drop = set()
        geom_map, class_map = self._preprocess_layer(source_layer, class_column_name)

        for class_value in class_values:
            visited = set()

            for sid in df["segment_id"].unique():
                if sid in visited or class_map[sid] != class_value:
                    continue

                group_geom = [geom_map[sid]]
                group_ids = [sid]
                queue = [sid]
                visited.add(sid)

                while queue:
                    current_id = queue.pop()
                    neighbors = self._find_neighbors(current_id, geom_map)
                    for n_id in neighbors:
                        if n_id not in visited and class_map.get(n_id) == class_value:
                            visited.add(n_id)
                            group_geom.append(geom_map[n_id])
                            group_ids.append(n_id)
                            queue.append(n_id)

                merged_geom = unary_union(group_geom)
                row_data = {"segment_id": min(group_ids), class_column_name: class_value, "geometry": merged_geom}

                new_rows.append(row_data)
                to_drop.update(group_ids)

        df = df[~df["segment_id"].isin(to_drop)]
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        result_layer.objects = df

        result_layer.metadata = {
            "mergeruleset_name": self.name,
            "merged_fields": class_values,
        }

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer


class EnclosedByRuleSet(CommonBase):
    """A rule set to reclassify segments based on spatial enclosure.

    This rule set identifies segments of one class (A) that are entirely surrounded
    by segments of another class (B), and reclassifies them into a new class.
    """

    def __init__(self, name=None):
        """Initialize the merge rule set.

        Parameters:
        -----------
        name : str, optional
            Name of the merge rule set
        """
        self.name = name if name else "EnclosedByRuleSet"

    def execute(
        self, source_layer, class_column_name, class_value_a, class_value_b, new_class_name, layer_manager=None, layer_name=None
    ):
        """Apply enclosed-by logic to identify and reclassify segments.

        Parameters:
        -----------
        source_layer : Layer
            The source spatial layer containing segments.
        class_column_name : str
            The name of the column containing class labels (e.g., "veg_class").
        class_value_a : str
            The class value to check for enclosure (target to reclassify).
        class_value_b : str
            The class value expected to surround class A segments.
        new_class_name : str
            The new class name to assign to enclosed segments.
        layer_manager : LayerManager, optional
            Optional manager to register the resulting layer.
        layer_name : str, optional
            Optional name for the result layer.

        Returns:
        --------
        result_layer : Layer
            A new layer with updated class values for enclosed segments.
        """
        if not layer_name:
            layer_name = f"{source_layer.name}_{self.name}"

        result_layer = Layer(name=layer_name, parent=source_layer, layer_type="merged")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

        df = source_layer.objects.copy()
        surrounded_segments = []
        geom_map, class_map = self._preprocess_layer(source_layer, class_column_name)

        for sid in df["segment_id"].unique():
            if class_map.get(sid) != class_value_a:
                continue

            neighbors = self._find_neighbors(sid, geom_map)
            if neighbors and all(class_map.get(n_id) == class_value_b for n_id in neighbors):
                surrounded_segments.append(sid)

        df.loc[(df["segment_id"].isin(surrounded_segments)), class_column_name] = new_class_name

        result_layer.objects = df

        result_layer.metadata = {
            "enclosed_by_ruleset_name": self.name,
        }

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer


class TouchedByRuleSet(CommonBase):
    """A rule set to reclassify segments based on spatial enclosure.

    This rule set identifies segments of one class (A) that are entirely surrounded
    by segments of another class (B), and reclassifies them into a new class.
    """

    def __init__(self, name=None):
        """Initialize the merge rule set.

        Parameters:
        -----------
        name : str, optional
            Name of the merge rule set
        """
        self.name = name if name else "TouchedByRuleSet"

    def execute(
        self, source_layer, class_column_name, class_value_a, class_value_b, new_class_name, layer_manager=None, layer_name=None
    ):
        """Executes the merge rule set by identifying and updating segments of a given class that are adjacent to another class!

        Parameters:
        - source_layer: Layer
            The input layer containing segment geometries and attributes.
        - class_column_name: str
            The name of the column containing class labels.
        - class_value_a: str or int
            The class value of segments to be checked for touching neighbors.
        - class_value_b: str or int
            The class value of neighboring segments that would trigger a merge.
        - new_class_name: str
            The new class value to assign to segments of class_value_a that touch class_value_b.
        - layer_manager: optional
            An optional manager for adding the resulting layer to a collection or interface.
        - layer_name: optional
            Optional custom name for the resulting layer. Defaults to "<source_layer_name>_<ruleset_name>".

        Returns:
        - result_layer: Layer
            A new Layer object with updated segment classifications where applicable.

        Logic:
        - Copies the source layer and initializes a new result layer.
        - Preprocesses the source layer to build geometry and class lookup maps.
        - Iterates through each segment of class_value_a, checking if any of its neighbors belong to class_value_b.
        - If so, updates the segment's class to new_class_name.
        - Stores the modified DataFrame in the result layer and optionally registers it via the layer_manager.

        """
        if not layer_name:
            layer_name = f"{source_layer.name}_{self.name}"

        result_layer = Layer(name=layer_name, parent=source_layer, layer_type="merged")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

        df = source_layer.objects.copy()
        touched_segments = []
        geom_map, class_map = self._preprocess_layer(source_layer, class_column_name)

        for sid in df["segment_id"].unique():
            if class_map.get(sid) != class_value_a:
                continue

            neighbors = self._find_neighbors(sid, geom_map)
            if neighbors and any(class_map.get(n_id) == class_value_b for n_id in neighbors):
                touched_segments.append(sid)

        df.loc[(df["segment_id"].isin(touched_segments)), class_column_name] = new_class_name

        result_layer.objects = df

        result_layer.metadata = {
            "enclosed_by_ruleset_name": self.name,
        }

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer
