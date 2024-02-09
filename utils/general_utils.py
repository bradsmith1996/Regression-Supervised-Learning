import numpy as np
import os
from sklearn.metrics.cluster import adjusted_rand_score
import argparse
import ast


# Create cluster directory names to put into the datasets folder
def get_cluster_dir_name(inputs):
    dataset_name = inputs.dataset
    algorithm = inputs.clustering_algorithm
    algorithm_inputs = ""
    for key in inputs.algorithm_inputs:
        algorithm_inputs += "{}_{}".format(key, inputs.algorithm_inputs[key])
    return dataset_name + "_" + algorithm + "_" + algorithm_inputs


# Create feature reduction directory names to put into the datasets folder
def get_feature_reduction_dir_name(inputs):
    dataset_name = inputs.dataset
    algorithm = inputs.reduction_algorithm
    algorithm_inputs = ""
    addition_for_dt = ""
    if inputs.reduction_algorithm == "DT":
        addition_for_dt = "_"+str(inputs.max_features)+"_"+str(inputs.mean_scale_factor)

    for key in inputs.algorithm_inputs:
        algorithm_inputs += "{}_{}".format(key, inputs.algorithm_inputs[key])
    return dataset_name + "_" + algorithm + "_" + algorithm_inputs + addition_for_dt


# Combine new features with original y data or vice versa
def combine_data(X_data, y_data):
    return np.concatenate((X_data, y_data.reshape(y_data.shape[0], 1)), axis=1)


# Adjusted rand score
def get_adjusted_rand_score(truth_data_path,
                            clustered_data_path,
                            ):
    dataset_truth_training = np.genfromtxt(os.path.join(truth_data_path, 'training.csv'), delimiter=',')
    dataset_truth_valid = np.genfromtxt(os.path.join(truth_data_path, 'validation.csv'), delimiter=',')
    dataset_clustered_training = np.genfromtxt(os.path.join(clustered_data_path, 'training_clustered.csv'),
                                               delimiter=',')
    dataset_clustered_valid = np.genfromtxt(os.path.join(clustered_data_path, 'validation_clustered.csv'),
                                            delimiter=',')

    dataset_truth = np.concatenate((dataset_truth_training, dataset_truth_valid), axis=0)
    dataset_clustered = np.concatenate((dataset_clustered_training, dataset_clustered_valid), axis=0)

    y_clustered = dataset_clustered[:, -1].astype(int)
    y_truth = dataset_truth[:, -1].astype(int)

    return adjusted_rand_score(y_clustered, y_truth)

def get_arg_parser():
    # Load arguments
    parser = argparse.ArgumentParser(
        description="Neural Network Training Script"
    )
    # [1] Need to know where the data is located
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset to train on",
        required=True,
    )
    # [1a] Need to know the model architecture
    parser.add_argument(
        "--architecture",
        type=str,
        help="Define which type of classification learner to be used",
        choices=[
            "neural_network",
            "random_forest",
            "polynomial"
        ],
        default="neural_network"
    )
    # [2] Need to know how many outputs there are. Can't assume. Must be provided
    parser.add_argument(
        "--num-outputs",
        type=int,
        help="Number of outputs from the provided dataset. Assumes ouptuts are in right most col. of dataset csvs!",
        required=True,
    )
    # [3] Need to know where to put the output information (to compare results later/together)
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to output training artifacts. There must be 3 files present: training.csv, validation,csv, and test.csv",
        required=True,
    )
    # [6] Validation Plot Arguments
    parser.add_argument(
        "--validation-curve",
        type=str,
        help="Flag / instruction of which parameter to run a validation curve over",
        default=None,
    )
    parser.add_argument(
        "--validation-parameter-range",
        help="Range of values to run the validation curve over for the architecture",
        nargs='+',
        default=None,
    )
    # [7] Grid-Cross Validation
    parser.add_argument(
        "--grid-cv-search-dict",
        help="For use in grid searching over a range of parameters, must provide argument as a python dictionary",
        type=ast.literal_eval,
        default=None
    )
    # [8] Flag to command the script to run over test data
    parser.add_argument(
        "--test",
        dest='test',
        help="Flag to run test. Should only do this when you think you have a good model",
        action='store_true'
    )
    parser.add_argument(
        "--algorithm-inputs",
        help="kargs for algorithm to be passed during initialization",
        type=ast.literal_eval,
        required=True
    )

    return parser


def get_cluster_output_dir_name(inputs):
    algorithm_inputs = ""
    # If no inputs given bail
    if inputs['algorithm_inputs'] is None:
        return os.path.join(inputs['architecture'], "defaults")

    # Make a unique algorithm string
    for key in inputs['algorithm_inputs']:
        if key == 'architecture':
            # Skip, we want the architecture to be a subdirectory
            continue
        algorithm_inputs += "{}_{}".format(key, inputs['algorithm_inputs'][key])
    return os.path.join(inputs['architecture'], algorithm_inputs)