# -*- coding: utf-8 -*-
"""Implements supervised classification algorithms to classify the segments."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .layer import Layer
from collections import Counter

class SupervisedClassifier:
    """Implementation of Supervised Classification algorithm."""

    # TODO: name vs layer_name

    def __init__(self, name=None, classifier_type="Random Forest", classifier_params=None):
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
        """Calculate statistics for segments based on image data.

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

        if self.classifier_type == "Random Forest":
            self.classifier = RandomForestClassifier(**self.classifier_params)
            self.classifier.fit(x, y)
            feature_importances = pd.Series(self.classifier.feature_importances_, index=self.features) * 100
            feature_importances = feature_importances.sort_values(ascending=False)

        test_accuracy = self.classifier.oob_score_
        # print("OOB Score:", self.classifier.oob_score_)

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


class SupervisedClassifierDL:
    """Implementation of deep learning based supervised classification"""

    def __init__(self, name="CNN_Classification", classifier_type="Concolution Neural Network (CNN)", classifier_params=None):
        """
        Initialize a Convolutional Neural Network (CNN) classifier.

        Parameters
        ----------
        name : str, optional
            Custom name for the classification layer. Defaults to "CNN_Classification" if None.
        classifier_type : str, optional
            Type of classifier. Defaults to "Convolution Neural Network (CNN)".
        classifier_params : dict, optional
            Dictionary of training parameters for the CNN. 
            If None, the following defaults are used:
                - epochs (int): 50
                - batch_size (int): 32
                - patch_size (tuple of int): (5, 5)
                - early_stopping_patience (int): 5
        """
        

        # print(classifier_params,"classifier_params")
        self.name = name
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params if classifier_params else {"epochs": 50, "batch_size": 32, "patch_size": (5, 5), "early_stopping_patience":5}
        self.model = None
        self.le = LabelEncoder()

    def _extract_training_patches(self, image, segments, samples):
        """
        Extract fixed-size training patches from an image based on segment IDs and labeled samples.

        This method iterates over the provided labeled segments, extract all the possible image patches
        of the specified patch size.

        Parameters
        ----------
        image : np.ndarray
            Input image as a NumPy array of shape (C, H, W) where C is the number of channels.
        segments : object
            Segmentation object containing a `raster` attribute (2D array of segment IDs).
        samples : dict
            Dictionary mapping class labels to lists of segment IDs, e.g.,
            {
                "class_1": [1, 5, 9],
                "class_2": [2, 6, 10]
            }.

        Returns
        -------
        patches : np.ndarray
            Array of extracted patches of shape (N, patch_height, patch_width, C),
            where N is the number of valid patches.
        labels : list
            List of class labels corresponding to each patch.
        counts_dict : dict
             Dictionary mapping each unique class label to the number of occurrences in `labels`.

        Notes
        -----
        - Patch size is taken from `self.classifier_params["patch_size"]` if provided;
          otherwise defaults to (5, 5).
        - Only patches where all pixels belong to the same segment ID are included.
        """
        
        image = np.moveaxis(image, 0, -1)
        patches = []
        labels = []
        patch_size = self.classifier_params["patch_size"] if "patch_size" in self.classifier_params.keys() else (5,5)

        # Extract region properties
        props = regionprops(segments.raster)
        segment_id_to_region = {prop.label: prop for prop in props}

        for key in samples.keys():
            segment_ids = samples[key]
            for seg_id in segment_ids:
                incount = 0
                outcount = 0
                region = segment_id_to_region.get(seg_id)
                if region is None:
                    print(f"Segment id {seg_id} not found, skipping.")
                    continue

                bbox = region.bbox  # min_row, min_col, max_row, max_col
                min_row, min_col, max_row, max_col = bbox[0], bbox[1], bbox[2], bbox[3]

                n_row_patches = (max_row - min_row) // patch_size[0]
                n_col_patches = (max_col - min_col) // patch_size[1]

                for i in range(n_row_patches):
                    for j in range(n_col_patches):
                        row_start = min_row + i * patch_size[0]
                        row_end = row_start + patch_size[0]

                        col_start = min_col + j * patch_size[1]
                        col_end = col_start + patch_size[1]

                        mask = segments.raster[row_start:row_end, col_start:col_end] == seg_id
                        if np.all(mask):
                            incount += 1
                            patch = image[row_start:row_end, col_start:col_end]
                            patches.append(patch)
                            labels.append(key)
                        else:
                            outcount += 1
        patches = np.array(patches)
        print(f"** Extracted {len(patches)} training patches of shape {patches.shape[1:]} **")
        counts = Counter(labels)
        counts_dict=dict(counts)
        print("** Class distribution: **")
        for cls, count in counts_dict.items():
            print(f"  - {cls}: {count} patches")
        return patches, labels, counts_dict

    def _extract_patches_for_prediction(self, image, segments):
        """
        Extract fixed-size patches from an image for prediction.

        This method iterates over each segment in the segmentation raster, identifies
        rectangular regions that match the specified patch size, and extracts those
        patches where all pixels belong to the same segment. 

        Parameters
        ----------
        image : np.ndarray
            Input image as a NumPy array of shape (C, H, W), where C is the number of channels.
        segments : object
            Segmentation object containing a `raster` attribute (2D array of segment IDs).

        Returns
        -------
        patches : np.ndarray
            List of extracted patches, each of shape (patch_height, patch_width, Channels).
        segment_ids : list of ids(int)
            List of segment IDs corresponding to each extracted patch.
        invalid_patches_segments_ids : list of int
            List of segment IDs for which no valid patches were found.

        Notes
        -----
        - Patch size is taken from `self.classifier_params["patch_size"]`.
        - Only patches where all pixels belong to the same segment ID are included.
        - `segment_ids` and `patches` maintain the same ordering so that each patch
        can be mapped back to its original segment.
        """
       
        image = np.moveaxis(image, 0, -1)
        patches = []
        segment_ids = []
        invalid_patches_segments_ids=[]
        patch_size = self.classifier_params["patch_size"]

        # Extract region properties
        props = regionprops(segments.raster)

        for prop in props:
            incount = 0
            outcount = 0
            bbox = prop.bbox  # min_row, min_col, max_row, max_col
            min_row, min_col, max_row, max_col = bbox[0], bbox[1], bbox[2], bbox[3]

            n_row_patches = (max_row - min_row) // patch_size[0]
            n_col_patches = (max_col - min_col) // patch_size[1]

            for i in range(n_row_patches):
                for j in range(n_col_patches):
                    row_start = min_row + i * patch_size[0]
                    row_end = row_start + patch_size[0]

                    col_start = min_col + j * patch_size[1]
                    col_end = col_start + patch_size[1]

                    mask = segments.raster[row_start:row_end, col_start:col_end] == prop.label
                    if np.all(mask):
                        incount += 1
                        patch = image[row_start:row_end, col_start:col_end]
                        patches.append(patch)
                        segment_ids.append(prop.label)
                    else:
                        outcount += 1
            if incount==0:
                invalid_patches_segments_ids.append(prop.label)
        patches = np.array(patches)
        if invalid_patches_segments_ids:
            print(f"Error: Could not create patch for the following segments: {invalid_patches_segments_ids} \n Possible reasons: lare patch_size or small segements")
        return patches, segment_ids, invalid_patches_segments_ids

    def _create_cnn_model(self, input_shape, num_classes):
        """Define a CNN model."""
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),  # softmax for multi-class classification
            ]
        )

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def _train_model(self, patches_train, labels_train, patches_val, labels_val):
        """
        Train the CNN model using the provided training and validation datasets.

        This method fits the CNN model with early stopping and learning rate reduction 
        callbacks to prevent overfitting and improve convergence. The best model weights 
        are restored based on the lowest validation loss.

        Parameters
        ----------
        patches_train : np.ndarray
            Training image patches of shape (N, H, W, C).
        labels_train : np.ndarray
            Training labels corresponding to `patches_train`.
        patches_val : np.ndarray
            Validation image patches of shape (N, H, W, C).
        labels_val : np.ndarray
            Validation labels corresponding to `patches_val`.

        Returns
        -------
        history : tensorflow.python.keras.callbacks.History
            Keras History object containing training and validation loss/accuracy metrics 
            for each epoch.

        Notes
        -----
        - Early stopping patience is taken from `self.classifier_params["early_stopping_patience"]` 
          if available; otherwise defaults to 5.
        - The learning rate is reduced by a factor of 0.5 after `early_stopping_patience` 
          epochs without improvement, with a minimum learning rate of 1e-7.
        - The method assumes that `self.model` has already been compiled.
        """
        
        early_stopping_patience= self.classifier_params["early_stopping_patience"] if "early_stopping_patience" in self.classifier_params.keys() else 5

        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",  
            patience=early_stopping_patience,  # Stop training after early_stopping_patience epochs of no improvement
            restore_best_weights=True,  # Restore the best weights from the epoch with the lowest validation loss
            verbose=1,  # Print messages when early stopping is triggered
        )

        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=early_stopping_patience, min_lr=1e-7)

        history = self.model.fit(
            patches_train,
            labels_train,
            epochs=self.classifier_params["epochs"],
            batch_size=self.classifier_params["batch_size"],
            validation_data=(patches_val, labels_val),
            callbacks=[early_stopping, reduce_lr],
        )
        return history

    def _prediction(self, patches, segment_ids):
        """
        Predict class labels for image patches and aggregate results by segment ID.

        This method uses the trained CNN model to predict class probabilities for each 
        patch, determines the most probable class, and then assigns a final label to 
        each segment based on the majority vote of its corresponding patches.

        Parameters
        ----------
        patches : np.ndarray
            Image patches to classify, of shape (N, patch_height, patch_width, channels).
        segment_ids : list of int
            List of segment IDs corresponding to each patch in `patches`.

        Returns
        -------
        final_segment_ids : list of int
            Unique segment IDs that received predictions.
        final_labels : list
            Predicted class labels for each segment, determined by majority voting.

        Notes
        -----
        - For each segment, the label assigned is the one with the highest occurrence 
        among its patches.
        """
        
        predictions = self.model.predict(patches)
        predicted_classes = predictions.argmax(axis=1)
        predicted_labels = self.le.inverse_transform(predicted_classes)

        # Step 1: collect predicted labels per segment
        segment_label_map = defaultdict(list)

        for seg_id, label in zip(segment_ids, predicted_labels, strict=False):
            segment_label_map[seg_id].append(label)

        # Step 2: for each unique segment_id, choose the label with highest occurrence
        final_segment_ids = []
        final_labels = []

        for seg_id, labels in segment_label_map.items():
            most_common_label = Counter(labels).most_common(1)[0][0]
            final_segment_ids.append(seg_id)
            final_labels.append(most_common_label)
        return final_segment_ids, final_labels

    def _evaluate(self, patches_test, labels_test):
        """
        Evaluate the trained CNN model on test data.

        This method predicts class labels for test patches, calculates accuracy,
        confusion matrix, and generates a detailed classification report. Results
        are printed and also returned as a dictionary.

        Parameters
        ----------
        patches_test : np.ndarray
            Test image patches of shape (N_test, patch_height, patch_width, channels).
        labels_test : np.ndarray
            Test labels corresponding to `patches_test`.

        Returns
        -------
        results : dict
            Dictionary containing evaluation metrics:
            - "accuracy" : float, classification accuracy score.
            - "confusion_matrix" : np.ndarray, confusion matrix array.
            - "report" : str, text summary of precision, recall, f1-score for each class.

        """
        predictions = self.model.predict(patches_test)
        predicted_classes = predictions.argmax(axis=1)

        # Compute accuracy
        accuracy = accuracy_score(labels_test, predicted_classes)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels_test, predicted_classes)

        # Classification report
        report = classification_report(labels_test, predicted_classes, target_names=self.le.classes_)

        print("Evaluation Results:")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(report)
        return {"accuracy": accuracy, "confusion_matrix": conf_matrix, "report": report}
        
    def execute(self, source_layer, samples, image_data, layer_manager=None, layer_name=None):
        """
        Perform CNN-based classification on image segments using labeled training samples.

        This method extracts training patches from the input image based on provided 
        samples, trains a CNN model, evaluates it on test data, predicts labels for all 
        segments, and stores the classification results in a new output layer.

        Parameters
        ----------
        source_layer : Layer
            Input spatial layer containing segments/objects to classify.
        samples : dict
            Dictionary mapping class names to lists of segment IDs used for training.
        image_data : np.ndarray
            Raster image data array from which patches are extracted for classification.
        layer_manager : LayerManager, optional
            manager object to register the output classification layer.
        layer_name : str, optional
            The name to assign to the resulting classified layer.
        
        Returns
        -------
        result_layer : Layer
            New layer containing the original segments with a "classification" attribute 
            representing predicted class labels.
        history : keras.callbacks.History
            Training history object containing loss and accuracy metrics per epoch.
        eval_result : dict
            Dictionary containing evaluation metrics such as accuracy, confusion matrix,
            and classification report on the test dataset.
        count_dict : dict
            Dictionary mapping class labels to the count of training patches extracted.
        invalid_patches_segments_ids : list
            List of segment IDs for which no valid patches could be extracted for prediction.
        """

        result_layer = Layer(name=layer_name, parent=source_layer, type="merged")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs

        layer = source_layer.objects.copy()
        patches, labels, count_dict = self._extract_training_patches(image=image_data, segments=source_layer, samples=samples)

        # Shuffle patches and labels
        indices = np.arange(len(patches))
        np.random.shuffle(indices)
        patches = patches[indices]
        labels = np.array(labels)[indices]

        labels_encoded = self.le.fit_transform(labels)
        num_classes = len(self.le.classes_)
        patches = patches.astype("float32") / 255.0

        patches_temp, patches_test, labels_temp, labels_test = train_test_split(
            patches, labels_encoded, test_size=0.3, random_state=42
        )

        patches_train, patches_val, labels_train, labels_val = train_test_split(
            patches_temp, labels_temp, test_size=0.2, random_state=42
        )

        input_shape = patches.shape[1:]
        num_classes = len(np.unique(labels))

        if self.classifier_type == "Convolution Neural Network (CNN)":
            self.model = self._create_cnn_model(input_shape, num_classes)

        history = self._train_model(patches_train, labels_train, patches_val, labels_val)

        patches_all, segment_ids, invalid_patches_segments_ids = self._extract_patches_for_prediction(image=image_data, segments=source_layer)

        eval_result = self._evaluate(patches_test, labels_test)
        
        patches_all = patches_all.astype("float32") / 255.0
        final_segment_ids, final_labels = self._prediction(patches_all, segment_ids)

        segment_to_label = dict(zip(final_segment_ids, final_labels, strict=False))
        layer["classification"] = ""
        layer["classification"] = layer["segment_id"].map(segment_to_label)

        result_layer.objects = layer
        result_layer.metadata = {"cnn_classification": self.name}

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer, history, eval_result, count_dict, invalid_patches_segments_ids
