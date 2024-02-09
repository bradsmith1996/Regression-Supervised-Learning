import argparse
import ast
import os
import yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import ExtraTreesClassifier
import utils.general_utils as utils
from sklearn import preprocessing
from scipy.stats import kurtosis
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 1
TRAINING_FILE_NAME = "training.csv"
VALIDATION_FILE_NAME = "validation.csv"
TEST_FILE_NAME = "test.csv"

if __name__ == '__main__':
    # Load arguments
    parser = argparse.ArgumentParser(
        description="Feature Reduction Algorithm Script"
    )
    # [1] Provide the data to run clustering algorithm on
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to run the optimization problem over",
        required=True,
    )
    # [2] State the clustering algorithm to use
    parser.add_argument(
        "--reduction-algorithm",
        type=str,
        help="Name of the algorithm to use for clustering",
        required=True,
        choices=[
            "PCA",  # Principal Component Analysis (PCA)
            "ICA",  # Independent Component Analysis (ICA)
            "RP",  # Randomized Projections
            "DT",  # Decision tree forrest
        ],
    )
    # [3] Algorithm Inputs
    parser.add_argument(
        "--algorithm-inputs",
        help="kargs for algorithm to be passed during initialization",
        type=ast.literal_eval,
        default=None
    )
    # [4] Max Features (for decision tree only, not an input to the other algs)
    parser.add_argument(
        "--max-features",
        help="Max number of features to use for decision tree forest, automatically set to num features if not added",
        type=int,
        default=None
    )
    # [5] Mean scale factor for treshold (for decision tree only, not an input to the other algs)
    parser.add_argument(
        "--mean-scale-factor",
        help="Mean scale factor for trimming off mean values from decision tree classifier important features",
        type=float,
        default=1.0
    )

    # Parse arguments into the inputs
    inputs = parser.parse_args()
    dataset_dir_name = utils.get_feature_reduction_dir_name(inputs)
    os.makedirs(dataset_dir_name, exist_ok=True)

    # Dump Inputs
    with open(os.path.join(dataset_dir_name, 'inputs.yml'), 'w') as input_file:
        yaml.dump(vars(inputs), input_file, default_flow_style=False)

    # Always overwrite the random state, don't want to mess with that
    if "random_state" in inputs.algorithm_inputs.keys():
        inputs.algorithm_inputs['random_state'] = RANDOM_STATE

    # Grab the files for training data
    training_data_file = os.path.join(inputs.dataset, TRAINING_FILE_NAME)
    validation_data_file = os.path.join(inputs.dataset, VALIDATION_FILE_NAME)
    test_data_file = os.path.join(inputs.dataset, TEST_FILE_NAME)

    # Load training, validation, and test data
    training_data = np.genfromtxt(training_data_file, delimiter=',')
    training_data_size = training_data.shape[0]
    validation_data = np.genfromtxt(validation_data_file, delimiter=',')
    validation_data_size = validation_data.shape[0]
    test_data = np.genfromtxt(test_data_file, delimiter=',')
    test_data_size = test_data.shape[0]

    # Combine for clustering, index only the X matrix (exclude last col. as y values)
    training_data_combined = np.concatenate((training_data, validation_data), axis=0)
    X_training = training_data_combined[:, :-1]
    num_features = X_training.shape[1]
    if inputs.max_features != None:
        num_features = inputs.max_features
    y_training = training_data_combined[:, -1]
    X_test = test_data[:, :-1]

    # Create a scale using the training data
    scaler = preprocessing.StandardScaler()
    X_training_scaled = scaler.fit_transform(X_training)
    X_test_scaled = scaler.transform(X_test)

    # Fit the data to the selected algorithm
    if inputs.reduction_algorithm == "PCA":
        # Create instance of PCA class
        pca = PCA(random_state=RANDOM_STATE, **inputs.algorithm_inputs)

        # Fit PCA to the scaled data and transform training / test data
        X_training_transformed = pca.fit_transform(X_training_scaled)
        X_test_transformed = pca.transform(X_test_scaled)

        # Compute the inverse transform to compute projection loss
        X_projected = pca.inverse_transform(X_training_transformed)  # Project into signal space
        projection_loss = np.sum((X_training_scaled - X_projected) ** 2, axis=1).mean()

        # Write out the projection loss for PCA:
        output_dictionary = {'n_components': pca.n_components, 'projection_loss': float(projection_loss)}
        with open(os.path.join(dataset_dir_name, 'outputs.yml'), 'w') as output_file:
            yaml.dump(output_dictionary, output_file, default_flow_style=False)
    elif inputs.reduction_algorithm == "ICA":
        # Create instance of ICA class
        ica = FastICA(random_state=RANDOM_STATE, **inputs.algorithm_inputs)

        # Fit ICA to the scaled data and transform training / test data
        X_training_transformed = ica.fit_transform(X_training_scaled)
        mean_kurtosis = np.abs(np.mean(kurtosis(X_training_transformed)))
        X_test_transformed = ica.transform(X_test_scaled)

        # Write out the mean kurtosis for ICA:
        output_dictionary = {'n_components': ica.n_components, 'mean_kurtosis': float(mean_kurtosis)}
        with open(os.path.join(dataset_dir_name, 'outputs.yml'), 'w') as output_file:
            yaml.dump(output_dictionary, output_file, default_flow_style=False)
    elif inputs.reduction_algorithm == "RP":
        # Create instance of RandomProjectionGaussian class
        rp = GaussianRandomProjection(random_state=RANDOM_STATE, **inputs.algorithm_inputs)

        # Fit ICA to the scaled data and transform training / test data
        X_training_transformed = rp.fit_transform(X_training_scaled)
        X_test_transformed = rp.transform(X_test_scaled)

        # Compute the projection loss
        transformation_matrix = rp.components_
        X_original = X_training_transformed @ transformation_matrix
        projection_loss = np.sum((X_training_scaled - X_original) ** 2, axis=1).mean()

        # Write out the projection loss for RP:
        output_dictionary = {'n_components': rp.n_components, 'projection_loss': float(projection_loss)}
        with open(os.path.join(dataset_dir_name, 'outputs.yml'), 'w') as output_file:
            yaml.dump(output_dictionary, output_file, default_flow_style=False)

    elif inputs.reduction_algorithm == "DT":
        # Create a random forest of decision trees
        clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
        clf = clf.fit(X_training_scaled, y_training)
        mean_factor = str(inputs.mean_scale_factor)+'*mean'
        model = SelectFromModel(estimator=clf, prefit=True, max_features=num_features, threshold=mean_factor)
        X_training_transformed = model.transform(X_training_scaled)
        X_test_transformed = model.transform(X_test_scaled)

        dummy_classifier = tree.DecisionTreeClassifier(random_state=RANDOM_STATE)
        score = cross_val_score(dummy_classifier, X_training_transformed, y_training, cv=10)
        mean_cv_score = score.mean()

        # Compute the inverse transform to compute projection loss
        X_projected = model.inverse_transform(X_training_transformed)  # Project into signal space
        projection_loss = np.sum((X_training_scaled - X_projected) ** 2, axis=1).mean()

        # Write out the projection loss for DT:
        output_dictionary = {'mean_scale_factor': inputs.mean_scale_factor, 'mean_cv_score': float(mean_cv_score)}
        with open(os.path.join(dataset_dir_name, 'outputs.yml'), 'w') as output_file:
            yaml.dump(output_dictionary, output_file, default_flow_style=False)

    # Separate out the different datasets to save them
    X_training_out = X_training_transformed[0:training_data_size, :]
    X_valid_out = X_training_transformed[training_data_size:training_data_size + validation_data_size, :]
    X_test_out = X_test_transformed

    # Humpty dumpty clustered labels back with feature vectors to analyze how similar to labels of original data
    training_data_clustered = utils.combine_data(X_training_out, training_data[:, -1])
    validation_data_clustered = utils.combine_data(X_valid_out, validation_data[:, -1])
    test_data_clustered = utils.combine_data(X_test_out, test_data[:, -1])

    # Write new dataset to datasets folder
    print("Writing feature reduced data to folder: {}".format(dataset_dir_name))
    np.savetxt(os.path.join(dataset_dir_name, TRAINING_FILE_NAME), training_data_clustered, delimiter=',')
    np.savetxt(os.path.join(dataset_dir_name, VALIDATION_FILE_NAME), validation_data_clustered, delimiter=',')
    np.savetxt(os.path.join(dataset_dir_name, TEST_FILE_NAME), test_data_clustered, delimiter=',')