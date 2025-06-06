# -*- coding: utf-8 -*-
"""Implements supervised classification algorithms to classify the segments."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .layer import Layer


class SupervisedClassifier:
    """Implementation of Supervised Classification algorithm."""

    # TODO: najime vs layer_name

    def __init__(self, name=None, classifier_type="Random Forest", classifier_params=None):
        """Initialize the segmentation algorithm.

        Parameters:
        -----------
        name : str
            name of the algorithm, used for metadata
        classifier_type : str
            "Random Forest", "SVC"
        classifier_params : dict
           additional parameters related to classifier
        """
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params
        self.training_layer = None
        self.classifier = None
        self.name = name if name else "Supervised_Classification"
        self.features = None
        self.scaler = None

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
        """Train  the classifier using the training layer.

        Parameters:
        -----------
        layer : Layer
            Layer containing segments
        image_data : numpy.ndarray
            Array with raster data values (bands, height, width)
        bands : list of str
            Names of the bands
        """
        self.features = features
        if not self.features:
            self.features = self.training_layer.columns
        self.features = [col for col in self.features if col not in ["segment_id", "classification", "geometry"]]

        x = self.training_layer[self.features]

        y = self.training_layer["classification"]

        metrics = {}

        ## validate
        supported_classifiers = ["Random Forest", "SVC"]
        if self.classifier_type not in ["Random Forest", "SVC"]:
            raise ValueError(f"Unsupported classifier type: '{self.classifier_type}'. Supported types are: {supported_classifiers}")

        if self.classifier_type == "Random Forest":
            self.classifier = RandomForestClassifier(**self.classifier_params)
            self.classifier.fit(x, y)

            ## calculate feature importances
            feature_importances = pd.Series(self.classifier.feature_importances_, index=self.features) * 100
            feature_importances = feature_importances.sort_values(ascending=False)

            ## add metrics
            metrics["feature_importances"] = feature_importances.to_dict()
            metrics["test_accuracy"] = self.classifier.oob_score_

        elif self.classifier_type == "SVC":
            ## scale
            self.scaler = StandardScaler()
            x_scaled = self.scaler.fit_transform(x)

            self.classifier = SVC(**self.classifier_params)
            self.classifier.fit(x_scaled, y)

            ## There is no native feature importance on SVC, so permutation is  used to achieve feature importance
            result = permutation_importance(self.classifier, x_scaled, y, n_repeats=10, random_state=420)
            metrics["feature_importances"] = (
                pd.Series(result.importances_mean, index=self.features).sort_values(ascending=False).to_dict()
            )

            ## add metrics
            cv_scores = cross_val_score(self.classifier, x_scaled, y, cv=5)
            test_accuracy = cv_scores.mean()

            metrics["test_accuracy"] = test_accuracy

        return self.classifier, metrics

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

        ## Apply scaling if using SVC
        if self.classifier_type == "SVC" and self.scaler:
            x = self.scaler.transform(x)

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
        _, metrics = self._train(features)

        layer = self._prediction(layer)

        result_layer.objects = layer

        result_layer.metadata = {"supervised classification": self.name, "metrics": metrics}

        result_layer.metadata["classifier"] = self.classifier_type
        result_layer.metadata["classifier_params"] = self.classifier_params

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer, metrics
