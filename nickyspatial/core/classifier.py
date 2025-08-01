# -*- coding: utf-8 -*-
"""Implements supervised classification algorithms to classify the segments."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .layer import Layer


class SupervisedClassifier:
    """Implementation of Supervised Classification algorithm."""

    # TODO: name vs layer_name

    def __init__(self, name=None, classifier_type="Random Forests", classifier_params=None):
        """Initialize the segmentation algorithm.

        Parameters:
        -----------
        scale : str
            classifier type name eg: RF for Random Forest, SVC for Support Vector Classifier
        classifier_params : dict
           additional parameters relayed to classifier
        """
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params
        self.training_layer = None
        self.classifier = None
        self.name = name if name else "Supervised_Classification"
        self.features = None

    def _training_sample(self, layer, samples):
        """Create vector objects from segments.

        Parameters:
        -----------
        samples : dict
            key: class_name
            values: list of segment_ids
            eg: {"cropland":[1,2,3],"built-up":[4,5,6]}

        Returns:
        --------
        segment_objects : geopandas.GeoDataFrame
            GeoDataFrame with segment polygons
        """
        layer["classification"] = None

        for class_name in samples.keys():
            layer.loc[layer["segment_id"].isin(samples[class_name]), "classification"] = class_name

        layer = layer[layer["classification"].notna()]
        self.training_layer = layer
        return layer

    def _train(self, features):
        """Train the classifier using the training samples and compute accuracy and feature importances.

        Parameters
        ----------
        features : list of str or None
            List of feature column names to use. If None, all columns except segment_id, geometry, and classification are used.

        Returns:
        -------
        classifier : sklearn classifier object
            The trained classifier.
        test_accuracy : float
            Accuracy score on training data.
        feature_importances : pd.Series or None
            Feature importances (only for Random Forest), else None.
        """
        self.features = features
        if not self.features:
            self.features = self.training_layer.columns
        self.features = [col for col in self.features if col not in ["segment_id", "classification", "geometry"]]

        x = self.training_layer[self.features]
        y = self.training_layer["classification"]

        # Random Forest
        if self.classifier_type == "Random Forest":
            self.classifier = RandomForestClassifier(**self.classifier_params)
            self.classifier.fit(x, y)
            test_accuracy = self.classifier.oob_score_
            feature_importances = pd.Series(self.classifier.feature_importances_, index=self.features) * 100
            feature_importances = feature_importances.sort_values(ascending=False)

        # Support Vector Machine (SVC)
        elif self.classifier_type == "SVC":
            self.classifier = SVC(**self.classifier_params)
            self.classifier.fit(x, y)
            predictions = self.classifier.predict(x)
            test_accuracy = accuracy_score(y, predictions)
            feature_importances = None  # SVM does not support feature importances

        # K-Nearest Neighbors (KNN)
        elif self.classifier_type == "KNN":
            self.classifier = KNeighborsClassifier(**self.classifier_params)
            self.classifier.fit(x, y)
            predictions = self.classifier.predict(x)
            test_accuracy = accuracy_score(y, predictions)
            feature_importances = None  # KNN does not support feature importances

        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

        return self.classifier, test_accuracy, feature_importances

    def _prediction(self, layer):
        """Perform classification prediction on input layer features.

        Parameters
        ----------
        layer : geopandas.GeoDataFrame
            Input data containing at least a 'segment_id' and 'geometry' column, along with
            feature columns required by the classifier. If a 'classification' column does not
            exist, it will be created.

        Returns:
        -------
        The input layer with an updated 'classification' column containing predicted labels.

        """
        layer["classification"] = ""
        # if not features:
        #     x = layer.drop(columns=["segment_id", "classification", "geometry"], errors="ignore")
        # else:
        x = layer[self.features]

        # print(layer.columns)
        # x = layer.drop(columns=["segment_id", "classification", "geometry"], errors="ignore")

        predictions = self.classifier.predict(x)
        layer.loc[layer["classification"] == "", "classification"] = predictions
        return layer

    def execute(self, source_layer, samples, layer_manager=None, layer_name=None, features=None):
        """Execute the supervised classification workflow on the source layer.

        This method creates a new layer by copying the input source layer, training a classifier
        using provided samples, predicting classifications, and storing the results in a new layer.
        Optionally, the resulting layer can be added to a layer manager.

        Parameters
        ----------
        source_layer : Layer
            The input layer containing spatial objects and metadata (transform, CRS, raster).
        samples : dict
            A dictionary of training samples where keys are class labels and values are lists
            of segment IDs or features used for training. Default is an empty dictionary.
        layer_manager : LayerManager, optional
            An optional layer manager object used to manage and store the resulting layer.
        layer_name : str, optional
            The name to assign to the resulting classified layer.

        Returns:
        -------
        Layer
            A new Layer object containing the predicted classifications, copied metadata from
            the source layer, and updated attributes.
        """
        result_layer = Layer(name=layer_name, parent=source_layer, type="merged")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

        layer = source_layer.objects.copy()
        self._training_sample(layer, samples)
        _, accuracy, feature_importances = self._train(features)

        layer = self._prediction(layer)

        result_layer.objects = layer

        result_layer.metadata = {
            "supervised classification": self.name,
        }

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer, accuracy, feature_importances
