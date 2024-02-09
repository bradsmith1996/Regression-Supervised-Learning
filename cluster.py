import argparse
import ast
import os
import yaml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import utils.general_utils as utils
import utils.plot as plot
from sklearn import preprocessing

RANDOM_STATE = 1
TRAINING_FILE_NAME = "training.csv"
VALIDATION_FILE_NAME = "validation.csv"
TEST_FILE_NAME = "test.csv"
TRAINING_FILE_NAME_CLUSTERED = "training_clustered.csv"
VALIDATION_FILE_NAME_CLUSTERED = "validation_clustered.csv"
TEST_FILE_NAME_CLUSTERED = "test_clustered.csv"

if __name__ == '__main__':
    # Load arguments
    parser = argparse.ArgumentParser(
        description="Clustering Algorithm Script"
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
        "--clustering-algorithm",
        type=str,
        help="Name of the algorithm to use for clustering",
        required=True,
        choices=[
            "KMeans",  # K means algorithm
            "GaussianMixture",  # Expectation Maximization
        ],
    )
    # [3] Algorithm Inputs
    parser.add_argument(
        "--algorithm-inputs",
        help="kargs for algorithm to be passed during initialization",
        type=ast.literal_eval,
        default=None
    )

    # Parse arguments into the inputs
    inputs = parser.parse_args()
    dataset_dir_name = utils.get_cluster_dir_name(inputs)
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
    X_test = test_data[:, :-1]

    # Create a scale using the training data
    scaler = preprocessing.StandardScaler()
    X_training_scaled = scaler.fit_transform(X_training)
    X_test_scaled = scaler.transform(X_test)

    # Fit the data to the selected algorithm
    if inputs.clustering_algorithm == "KMeans":
        # Instantiate kmeans given the user inputs
        kmeans = KMeans(random_state=RANDOM_STATE, **inputs.algorithm_inputs)

        # Fit kmeans to the training data
        clusters = kmeans.fit(X_training_scaled)
        clustered_training_labels = clusters.labels_

        # Transform test data into the clusters and into cluster distance space (new features are cluster distances)
        clustered_test_labels = kmeans.predict(X_test_scaled)
        X_training_cluster_space = kmeans.transform(X_training_scaled)
        X_test_cluster_space = kmeans.transform(X_test_scaled)
    if inputs.clustering_algorithm == "GaussianMixture":
        # Instantiate gaussian mixture model given the user inputs
        em = GaussianMixture(random_state=RANDOM_STATE, **inputs.algorithm_inputs)

        # Fit mixture of gaussian distributions to the training data
        clusters = em.fit(X_training_scaled)

        clustered_training_labels = clusters.predict(X_training_scaled)
        clustered_test_labels = clusters.predict(X_test_scaled)
        X_training_cluster_space = clusters.predict_proba(X_training_scaled)
        X_test_cluster_space = clusters.predict_proba(X_test_scaled)

    # Create clustered labels for the different dataset types
    y_training = clustered_training_labels[0:training_data_size]
    y_valid = clustered_training_labels[training_data_size:training_data_size + validation_data_size]
    y_test = clustered_test_labels

    # Create new features according to the cluster distance space (distance between each sample and each cluster)
    X_training_cs = X_training_cluster_space[0:training_data_size, :]
    X_valid_out_cs = X_training_cluster_space[training_data_size:training_data_size + validation_data_size, :]
    X_test_out_cs = X_test_cluster_space

    # Humpty dumpty clustered labels back with feature vectors to analyze how similar to labels of original data
    training_data_clustered = utils.combine_data(training_data[:, :-1], y_training)
    validation_data_clustered = utils.combine_data(validation_data[:, :-1], y_valid)
    test_data_clustered = utils.combine_data(test_data[:, :-1], y_test)
    # Humpty dumpty cluster distance features for new features to train a neural network model on
    training_data_out = utils.combine_data(X_training_cs, training_data[:, -1])
    validation_data_out = utils.combine_data(X_valid_out_cs, validation_data[:, -1])
    test_data_out = utils.combine_data(X_test_out_cs, test_data[:, -1])

    # Write new dataset to datasets folder
    print("Writing clustered data to folder: {}".format(dataset_dir_name))
    np.savetxt(os.path.join(dataset_dir_name, TRAINING_FILE_NAME_CLUSTERED), training_data_clustered, delimiter=',')
    np.savetxt(os.path.join(dataset_dir_name, VALIDATION_FILE_NAME_CLUSTERED), validation_data_clustered, delimiter=',')
    np.savetxt(os.path.join(dataset_dir_name, TEST_FILE_NAME_CLUSTERED), test_data_clustered, delimiter=',')
    np.savetxt(os.path.join(dataset_dir_name, TRAINING_FILE_NAME), training_data_out, delimiter=',')
    np.savetxt(os.path.join(dataset_dir_name, VALIDATION_FILE_NAME), validation_data_out, delimiter=',')
    np.savetxt(os.path.join(dataset_dir_name, TEST_FILE_NAME), test_data_out, delimiter=',')

    # Visualize the clusters
    if inputs.clustering_algorithm == "KMeans":
        # Examine KMeans using Silhouette Analysis
        plot.plot_silhoutte(X_training=X_training_scaled,
                            cluster_labels=clustered_training_labels,
                            n_clusters=kmeans.n_clusters,
                            output_dir=dataset_dir_name,
                            data_set_name=inputs.dataset,
                            )
    elif inputs.clustering_algorithm == "GaussianMixture":
        output_string = "For n_components ={} The average BIC is : {}".format(em.n_components,
                                                                              em.bic(X_training_scaled))
        print(output_string)
        with open(os.path.join(dataset_dir_name, "output.txt"), "w") as the_file:
            the_file.write(output_string)

        output_dictionary = {'n_components': em.n_components, 'bic': float(em.bic(X_training_scaled))}
        with open(os.path.join(dataset_dir_name, 'outputs.yml'), 'w') as output_file:
            yaml.dump(output_dictionary, output_file, default_flow_style=False)
