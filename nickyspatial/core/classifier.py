# -*- coding: utf-8 -*-
"""Implements supervised classification algorithms to classify the segments."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .layer import Layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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



"""Implements CNN-based classification for segmented image patches."""

import numpy as np
import geopandas as gpd
from skimage.measure import regionprops
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import layers, models
from .layer import Layer

class SupervisedClassifierDL:
# CNNClassifier:
    """Implementation of CNN-based classification for image patches from segments."""

    def __init__(self, name=None, classifier_type="Concolution Neural Network (CNN)", classifier_params=None):
        """Initialize the CNN classifier.

        Parameters:
        -----------
        name : str, optional
            Name of the classifier.
        patch_size : tuple
            Size of image patches (height, width).
        classifier_params : dict
            Parameters for CNN training (e.g., epochs, batch_size).
        """
        self.name = name if name else "CNN_Classification"
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params if classifier_params else {"epochs": 50, "batch_size": 32,"patch_size": (5,5)}
        self.model = None
        self.le = LabelEncoder()

    def _extract_training_patches(self, image, segments, samples):
        """
        Extract fixed-size patches centered on object centroids.
        Patches near image edges are padded using reflection.
        
        :param image: np.ndarray, shape (H, W, C)
        :param segments: segmentation raster (2D array of integer labels)
        :param samples: dict mapping class names to list of segment IDs
        :param patch_size: tuple, e.g., (5,5)
        :return: np.ndarray of patches, shape (N, patch_height, patch_width, channels)
        task_type="train" or "prediction"
        """
        image = np.moveaxis(image, 0, -1)
        patches = []
        labels=[]
        patch_size=self.classifier_params['patch_size']
        
        # Extract region properties
        props = regionprops(segments.raster)
        segment_id_to_region = {prop.label: prop for prop in props}
        
        for key in samples.keys():
            segment_ids = samples[key]
            for seg_id in segment_ids:
                # print("segment_id", seg_id)
                incount=0
                outcount=0
                region = segment_id_to_region.get(seg_id)
                if region is None:
                    print(f"Segment id {seg_id} not found, skipping.")
                    continue
                    
                bbox=region.bbox #min_row, min_col, max_row, max_col
                min_row, min_col, max_row, max_col= bbox[0],bbox[1],bbox[2],bbox[3]

                n_row_patches= (max_row-min_row) // patch_size[0]
                n_col_patches= (max_col-min_col) // patch_size[1]

                for i in range(n_row_patches):
                    for j in range(n_col_patches):
                        row_start = min_row + i * patch_size[0]
                        row_end = row_start + patch_size[0]
                        
                        col_start = min_col + j * patch_size[1]
                        col_end = col_start + patch_size[1]
                        
                        mask= (segments.raster[row_start:row_end, col_start:col_end] == seg_id)
                        if np.all(mask):
                            incount+=1
                            patch = image[row_start:row_end, col_start:col_end]
                            patches.append(patch)
                            labels.append(key)
                        else:
                            outcount+=1
                # print("incount", incount)
                # print("outcount", outcount)
                

        patches = np.array(patches)
        print(f"Extracted {len(patches)} training patches of shape {patches.shape[1:]}")
        return patches, labels
    

    def _extract_patches_for_prediction(self,image, segments):
        """
        Extract fixed-size patches centered on object centroids.
        Patches near image edges are padded using reflection.
        
        :param image: np.ndarray, shape (H, W, C)
        :param segments: segmentation raster (2D array of integer labels)
        :param samples: dict mapping class names to list of segment IDs
        :param patch_size: tuple, e.g., (5,5)
        :return: np.ndarray of patches, shape (N, patch_height, patch_width, channels)
        task_type="train" or "prediction"
        """
        image = np.moveaxis(image, 0, -1)
        patches = []
        segment_ids=[]
        patch_size=self.classifier_params['patch_size']
        
        # Extract region properties
        props = regionprops(segments.raster)
        segment_id_to_region = {prop.label: prop for prop in props}

        for prop in props:
            incount=0
            outcount=0
            # print(prop.label)
            # centroid=prop.centroid
            # print(centroid)
            bbox=prop.bbox #min_row, min_col, max_row, max_col
            min_row, min_col, max_row, max_col= bbox[0],bbox[1],bbox[2],bbox[3]
            # print(min_row, min_col, max_row, max_col)

            n_row_patches= (max_row-min_row) // patch_size[0]
            n_col_patches= (max_col-min_col) // patch_size[1]

            for i in range(n_row_patches):
                for j in range(n_col_patches):
                    row_start = min_row + i * patch_size[0]
                    row_end = row_start + patch_size[0]
                    
                    col_start = min_col + j * patch_size[1]
                    col_end = col_start + patch_size[1]
                    
                    mask= (segments.raster[row_start:row_end, col_start:col_end] == prop.label)
                    # print(mask)
                    if np.all(mask):
                        incount+=1
                        patch = image[row_start:row_end, col_start:col_end]
                        patches.append(patch)
                        # labels.append(key)
                        segment_ids.append(prop.label)
                    else:
                        outcount+=1
            # print("incount", incount)
            # print("outcount", outcount)
        patches = np.array(patches)
        # print(f"Extracted {len(patches)} patches of shape {patches.shape[1:]}")
        # return patches, labels
        return patches,  segment_ids

    def _create_cnn_model(self, input_shape, num_classes):
        """Define a CNN model."""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')   # softmax for multi-class classification
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # self.model=model
        return model
    
    def _train_model(self,patches_train,labels_train,patches_val,labels_val):
        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',    # Can use 'val_accuracy' if preferred
            patience=5,            # Stop training after 5 epochs of no improvement
            restore_best_weights=True,  # Restore the best weights from the epoch with the lowest validation loss
            verbose=1              # Print messages when early stopping is triggered
        )

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
                
        history = self.model.fit(
            patches_train, labels_train,
            epochs=self.classifier_params['epochs'],
            batch_size=self.classifier_params['batch_size'],
            validation_data=(patches_val, labels_val),
            callbacks=[early_stopping,reduce_lr]
        )
        return history
  
    def _prediction(self, patches,segment_ids):
        predictions = self.model.predict(patches)
        predicted_classes = predictions.argmax(axis=1)
        predicted_labels = self.le.inverse_transform(predicted_classes)

        # Step 1: collect predicted labels per segment
        segment_label_map = defaultdict(list)

        for seg_id, label in zip(segment_ids, predicted_labels):
            segment_label_map[seg_id].append(label)

        # Step 2: for each unique segment_id, choose the label with highest occurrence
        final_segment_ids = []
        final_labels = []

        for seg_id, labels in segment_label_map.items():
            most_common_label = Counter(labels).most_common(1)[0][0]
            final_segment_ids.append(seg_id)
            final_labels.append(most_common_label)
        # print(len(final_segment_ids))
        # print(len(final_labels))
        return final_segment_ids, final_labels
    
    def _evaluate(self, patches_test, labels_test):
        predictions = self.model.predict(patches_test)
        predicted_classes = predictions.argmax(axis=1)

        # # Convert ground truth labels to class indices if necessary
        # if hasattr(self, 'le'):
        #     # If labels_test are strings, encode them to indices
        #     true_classes = self.le.transform(labels_test)
        #     # Also get back predicted labels (strings)
        #     predicted_labels = self.le.inverse_transform(predicted_classes)
        # else:
        #     # Otherwise assume labels_test are already numeric
        #     true_classes = labels_test
        #     predicted_labels = predicted_classes

        # Compute accuracy
        accuracy = accuracy_score(labels_test, predicted_classes)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(labels_test, predicted_classes)

        # Classification report
        report = classification_report(labels_test, predicted_classes, target_names=self.le.classes_)
        
        return {"accuracy":accuracy,"confusion_matrix":conf_matrix,"report":report}
        # print("Evaluation Results:")
        # print(f"Accuracy: {acc:.4f}")
        # print("Confusion Matrix:")
        # print(cm)
        # print("Classification Report:")
        # print(report)

    def execute(self, source_layer, samples, image_data, layer_manager=None, layer_name=None):
        """Execute CNN-based classification.

        Parameters:
        ----------
        source_layer : Layer
            Input layer with spatial objects and segments.
        samples : dict
            Key: class_name, Values: list of segment_ids.
        image_data : numpy.ndarray
            Raster image data for patch extraction.
        layer_manager : LayerManager, optional
            Manager to store the resulting layer.
        layer_name : str, optional
            Name for the resulting layer.

        Returns:
        -------
        result_layer : Layer
            Layer with predicted classifications.
        accuracy : float
            Validation accuracy.
        """
        result_layer = Layer(name=layer_name, parent=source_layer, type="merged")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs

        layer = source_layer.objects.copy()
        # image_data=source_layer.raster.copy()
        # print(image_data,"image_data")
        patches, labels = self._extract_training_patches(image=image_data, segments=source_layer, samples=samples)
        labels_encoded = self.le.fit_transform(labels)
        num_classes = len(self.le.classes_)
        # print("Classes:", self.le.classes_)
        patches = patches.astype('float32') / 255.0
        
        patches_temp, patches_test, labels_temp, labels_test = train_test_split(
            patches, labels_encoded, test_size=0.3, random_state=42)
        
        patches_train, patches_val, labels_train, labels_val = train_test_split(
            patches_temp, labels_temp, test_size=0.2, random_state=42)

        # input_shape = (self.patch_size[0], self.patch_size[1], 3)
        input_shape = patches.shape[1:]
        num_classes = len(np.unique(labels))

        if self.classifier_type == "Convolution Neural Network (CNN)":
            self.model = self._create_cnn_model(input_shape, num_classes)

        history=self._train_model(patches_train,labels_train,patches_val,labels_val)
        
        # history = self.classifier.fit(patches, labels, **self.classifier_params, validation_split=0.2, verbose=0)
        # accuracy = history.history['val_accuracy'][-1]

        patches_all, segment_ids = self._extract_patches_for_prediction(image=image_data, segments=source_layer )

        eval_result= self._evaluate( patches_test, labels_test)
        print("Evaluation Results:")
        print(f"Accuracy: {eval_result["accuracy"]}")
        print("Confusion Matrix:")
        print(eval_result["confusion_matrix"])
        print("Classification Report:")
        print(eval_result["report"])

        final_segment_ids, final_labels =self._prediction(patches_all, segment_ids)

        segment_to_label = dict(zip(final_segment_ids, final_labels))
        layer["classification"]=""
        layer["classification"] = layer["segment_id"].map(segment_to_label)

        result_layer.objects = layer
        result_layer.metadata = {"cnn_classification": self.name}

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer, history, eval_result
