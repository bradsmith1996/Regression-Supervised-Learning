import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.ticker as mticker
import yaml

nicer_label_dictionary = {"n_clusters": "Cluster Count",
                          "n_components": "Component Count",
                          "avg_silhouette": "Average Silhouette Score",
                          "bic": "Bayesian Information-Theoretic Criteria",
                          "projection_loss": "Projection Loss",
                          "mean_scale_factor": "Mean Scale Factor Threshold",
                          "mean_cv_score": "Mean Cross-Validation Score (Decision Tree Classifier)",
                          }

semilogx_parameters = ["learning_curve",
                       "gamma",
                       ]

nicer_label = {"neural_network": "Neural Network",
               "random_forest": "Random Forest Regressor",
               "polynomial": "Polynomial Regression"
               }

font_dictionary = {'fontname':'Times New Roman'}


def plot_silhoutte(X_training,
                   cluster_labels,
                   n_clusters,
                   output_dir,
                   data_set_name,
                   ):
    # Create a subplot with 1 row and 2 columns
    fig1, ax1 = plt.subplots(1, 1)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_training) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_training, cluster_labels)
    output_string = "For n_clusters ={} The average silhouette_score is : {}".format(n_clusters, silhouette_avg)
    print(output_string)
    with open(os.path.join(output_dir, "output.txt"), "w") as the_file:
        the_file.write(output_string)

    output_dictionary = {'n_clusters': n_clusters, 'avg_silhouette': float(silhouette_avg)}
    with open(os.path.join(output_dir, 'outputs.yml'), 'w') as output_file:
        yaml.dump(output_dictionary, output_file, default_flow_style=False)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_training, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    just_data_set = os.path.split(data_set_name)[-1]
    ax1.set_title("Silhouette Plot (Data={}, Clusters={})".format(just_data_set, n_clusters))
    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Save off the silhouette plot
    temp = output_dir.split("/")
    file_name = "{}_silhouette.png".format(temp[-1])
    fig1.savefig(os.path.join(output_dir, file_name), dpi=1000, facecolor="white")
    fig1.show()


def generate_learning_curve_plot_nn(history,
                                    save_path,
                                    ):
    # Key names:
    train_accuracy_key = 'accuracy'

    # For the x axis have epoch indexing starting from 1
    epoch = list(range(0, len(history.history['loss'])))

    # Save off plots of metrics during training (loss)
    plt.grid()
    plt.plot(epoch, history.history['loss'], "-", color="b", label='Training Loss')
    plt.plot(epoch, history.history['val_loss'], "-", color="g", label='Validation Loss')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.legend()
    plt.title("Learning Curve: Neural Network", **font_dictionary)
    plt.xlabel("Epoch", **font_dictionary)
    plt.ylabel("Cross Entropy Loss", **font_dictionary)
    plt.savefig(os.path.join(save_path, "loss.png"), dpi=1000, facecolor="white")
    plt.show()
    plt.close()

    # Save off plots of metrics during training (accuracy)
    plt.grid()
    plt.ylim(0.25, 1.01)
    plt.plot(epoch, history.history[train_accuracy_key], "-", color="b", label='Training Accuracy')
    plt.plot(epoch, history.history[valid_accuracy_key], "-", color="g", label='Validation Accuracy')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.legend()
    plt.title("Learning Curve: Neural Network", **font_dictionary)
    plt.xlabel("Epoch", **font_dictionary)
    plt.ylabel("Accuracy (%)", **font_dictionary)
    plt.savefig(os.path.join(save_path, "accuracy.png"), dpi=1000, facecolor="white")
    plt.show()
    plt.close()


def generate_learning_curve_plot(architecture,
                                 train_sizes,
                                 train_scores,
                                 valid_scores,
                                 output_file_path,
                                 ):
    # Gather statistics from the scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    output_string1 = "Maximum Validation Accuracy Samples = {} (Accuracy={})\n".format(
        train_sizes[np.nanargmax(valid_scores_mean)],
        np.nanmax(valid_scores_mean)
    )
    output_string2 = "Validation Accuracy Over All Samples = {} (Accuracy={})".format(
        train_sizes[-1],
        valid_scores_mean[-1]
    )
    print(output_string1)
    print(output_string2)
    # Write some simple information to a file to persist
    with open(os.path.join(output_file_path, "final_validation_accuracy.txt"), "w") as the_file:
        the_file.write(output_string1)
        the_file.write(output_string2)

    plt.grid()
    plt.ylim(0.25, 1.01)
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
    )
    plt.fill_between(
        train_sizes,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(
        train_sizes, train_scores_mean, "-", color="b", label="Training score"
    )
    plt.plot(
        train_sizes, valid_scores_mean, "-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")
    plt.title('Learning Curve (Neural Network)', **font_dictionary)
    plt.xlabel("Training Examples", **font_dictionary)
    plt.ylabel("Accuracy (%)", **font_dictionary)
    plt.savefig(os.path.join(output_file_path, "learning_curve.png"), dpi=1000, facecolor="white")
    plt.show()
    plt.close()


def generate_learning_curve_plot_nn(history,
                                    save_path,
                                    ):
    # Key names:
    train_accuracy_key = 'mean_squared_error'
    valid_accuracy_key = 'val_mse'

    # For the x axis have epoch indexing starting from 1
    epoch = list(range(0, len(history['loss'])))

    # Save off plots of metrics during training (loss)
    plt.grid()
    plt.plot(epoch, history['loss'], "-", color="b", label='Training Loss')
    plt.plot(epoch, history['val_loss'], "-", color="g", label='Validation Loss')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.legend()
    plt.title("Learning Curve: Neural Network", **font_dictionary)
    plt.xlabel("Epoch", **font_dictionary)
    plt.ylabel("Cross Entropy Loss", **font_dictionary)
    plt.savefig(os.path.join(save_path, "loss.png"), dpi=1000, facecolor="white")
    plt.show()
    plt.close()

    # Save off plots of metrics during training (accuracy)
    plt.grid()
    plt.plot(epoch, history[train_accuracy_key], "-", color="b", label='Training Accuracy')
    plt.plot(epoch, history[valid_accuracy_key], "-", color="g", label='Validation Accuracy')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.legend()
    plt.title("Learning Curve: Neural Network", **font_dictionary)
    plt.xlabel("Epoch", **font_dictionary)
    plt.ylabel("Accuracy (%)", **font_dictionary)
    plt.savefig(os.path.join(save_path, "accuracy.png"), dpi=1000, facecolor="white")
    plt.show()
    plt.close()


def generate_validation_curve_plot(architecture,
                                   parameter,
                                   param_range,
                                   train_scores,
                                   valid_scores,
                                   output_file_path,
                                   ):
    if (architecture == "knn") and (parameter == "n_neighbors"):
        x_lower_bound = 0.0
    else:
        x_lower_bound = 0.5

    # Some parameters are best plotted plt.semilogx:
    plot_data_fn = None
    if parameter in semilogx_parameters:
        plot_data_fn = plt.semilogx
    elif (parameter == "learning_rate") and (architecture == "neural_network"):
        plot_data_fn = plt.semilogx
    elif (parameter == "learning_rate") and (architecture == "boosting"):
        plot_data_fn = plt.semilogx
    else:
        plot_data_fn = plt.plot

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    # This is bad, extremely stupid, but we have to do it, it's going to be okay
    if parameter == "base_estimator__max_depth":
        parameter = "max_depth"

    output_string = "Maximum Validation Accuracy with {} = {} (Accuracy={})".format(parameter,
                                                                                    param_range[np.nanargmax(
                                                                                        valid_scores_mean)],
                                                                                    np.nanmax(valid_scores_mean)
                                                                                    )
    print(output_string)
    # Write some simple information to a file to persist
    with open(os.path.join(output_file_path, "best_parameter.txt"), "w") as the_file:
        the_file.write(output_string)

    plt.title("Validation Curve: {}".format(nicer_label_dictionary[architecture]), **font_dictionary)
    plt.xlabel(convert_case(parameter), **font_dictionary)
    plt.ylabel("Accuracy (%)", **font_dictionary)
    plt.grid()
    plt.ylim(x_lower_bound, 1.01)
    lw = 2
    plot_data_fn(param_range, train_scores_mean, "-", color="b", label="Training score", lw=lw)
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
        lw=lw,
    )
    plot_data_fn(param_range, valid_scores_mean, "-", color="g", label="Cross-Validation Score", lw=lw)
    plt.fill_between(
        param_range,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="g",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_file_path, "validation_curve.png"), dpi=1000, facecolor="white")
    plt.show()


def generate_validation_curve_plot_regression(architecture,
                                              parameter,
                                              param_range,
                                              train_scores,
                                              valid_scores,
                                              output_file_path,
                                              ):

    # Some parameters are best plotted plt.semilogx:
    plot_data_fn = None
    if parameter in semilogx_parameters:
        plot_data_fn = plt.semilogx
    elif (parameter == "learning_rate") and (architecture == "neural_network"):
        plot_data_fn = plt.semilogx
    else:
        plot_data_fn = plt.plot

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    # This is bad, extremely stupid, but we have to do it, it's going to be okay
    if parameter == "base_estimator__max_depth":
        parameter = "max_depth"

    output_string = "Maximum Validation Accuracy with {} = {} (Accuracy={})".format(parameter,
                                                                                    param_range[np.nanargmax(
                                                                                        valid_scores_mean)],
                                                                                    np.nanmax(valid_scores_mean)
                                                                                    )
    print(output_string)
    # Write some simple information to a file to persist
    with open(os.path.join(output_file_path, "best_parameter.txt"), "w") as the_file:
        the_file.write(output_string)

    plt.title("Validation Curve: {}".format(nicer_label_dictionary[architecture]), **font_dictionary)
    plt.xlabel(convert_case(parameter), **font_dictionary)
    plt.ylabel("Accuracy (%)", **font_dictionary)
    plt.grid()
    plt.ylim(x_lower_bound, 1.01)
    lw = 2
    plot_data_fn(param_range, train_scores_mean, "-", color="b", label="Training score", lw=lw)
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
        lw=lw,
    )
    plot_data_fn(param_range, valid_scores_mean, "-", color="g", label="Cross-Validation Score", lw=lw)
    plt.fill_between(
        param_range,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="g",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_file_path, "validation_curve.png"), dpi=1000, facecolor="white")
    plt.show()


def generate_validation_curve_plot_regression(architecture,
                                              parameter,
                                              param_range,
                                              train_scores,
                                              valid_scores,
                                              output_file_path,
                                              scoring
                                              ):
    # Some parameters are best plotted plt.semilogx:
    plot_data_fn = None
    if parameter in semilogx_parameters:
        plot_data_fn = plt.semilogx
    elif (parameter == "learning_rate") and (architecture == "neural_network"):
        plot_data_fn = plt.semilogx
    elif (parameter == "learning_rate") and (architecture == "boosting"):
        plot_data_fn = plt.semilogx
    else:
        plot_data_fn = plt.plot

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    # This is bad, extremely stupid, but we have to do it, it's going to be okay
    if parameter == "base_estimator__max_depth":
        parameter = "max_depth"

    output_string = "Maximum Validation Accuracy with {} = {} (Accuracy={})".format(parameter,
                                                                                    param_range[np.nanargmax(
                                                                                    valid_scores_mean)],
                                                                                    np.nanmax(valid_scores_mean)
                                                                                    )
    print(output_string)
    # Write some simple information to a file to persist
    with open(os.path.join(output_file_path, "best_parameter.txt"), "w") as the_file:
        the_file.write(output_string)
    plt.title("Validation Curve: {}".format(nicer_label[architecture]), **font_dictionary)
    plt.xlabel(parameter, **font_dictionary)
    plt.ylabel(scoring, **font_dictionary)
    plt.grid()
    lw = 2
    plot_data_fn(param_range, train_scores_mean, "-", color="b", label="Training score", lw=lw)
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
        lw=lw,
    )
    plot_data_fn(param_range, valid_scores_mean, "-", color="g", label="Cross-Validation Score", lw=lw)
    plt.fill_between(
        param_range,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="g",
        lw=lw,
    )
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_file_path, "validation_curve.png"), dpi=1000, facecolor="white")
    plt.show()


def plot_cluster_comparisons(output_dir_list=None,
                             input_param_name=None,
                             output_param_name=None,
                             save_name="no_name_given",
                             title="No Title Given!"
                             ):
    inputs_files = [os.path.join(folder, "inputs.yml") for folder in output_dir_list]
    output_files = [os.path.join(folder, "outputs.yml") for folder in output_dir_list]
    input_data = []
    output_data = []
    for input, output in zip(inputs_files, output_files):
        with open(input, "r") as stream:
            input_data.append(yaml.safe_load(stream))
        with open(output, "r") as stream:
            output_data.append(yaml.safe_load(stream))

    # Assume if using this function it's the same problem and algorithm (just get first index)
    # problem_name = input_data[0]['dataset']
    # cluster_algorithm = input_data[0]['clustering_algorithm']

    # Plot results together:
    plt.grid()
    plot_data = []
    for input_dat, output_dat in zip(input_data, output_data):
        plot_data.append((input_dat['algorithm_inputs'][input_param_name], output_dat[output_param_name]))

    # Sort the data by the cluster size
    to_plot = np.array(plot_data)

    plt.title(title, **font_dictionary)
    plt.plot(to_plot[:, 0].astype(int), to_plot[:, 1], 'o-')
    plt.ylabel(nicer_label_dictionary[output_param_name], **font_dictionary)
    plt.xlabel(nicer_label_dictionary[input_param_name], **font_dictionary)
    plt.xticks(to_plot[:, 0].astype(int))
    # plt.legend(loc="best")
    plt.savefig(save_name, dpi=1000, facecolor="white")
    plt.show()
    return to_plot[:, 0], to_plot[:, 1]


def plot_cluster_comparisons_outputs(output_dir_list=None,
                                     input_param_name=None,
                                     output_param_name=None,
                                     save_name="no_name_given",
                                     title="No Title Given!"
                                     ):
    inputs_files = [os.path.join(folder, "inputs.yml") for folder in output_dir_list]
    output_files = [os.path.join(folder, "outputs.yml") for folder in output_dir_list]
    input_data = []
    output_data = []
    for input, output in zip(inputs_files, output_files):
        with open(input, "r") as stream:
            input_data.append(yaml.safe_load(stream))
        with open(output, "r") as stream:
            output_data.append(yaml.safe_load(stream))

    # Assume if using this function it's the same problem and algorithm (just get first index)
    # problem_name = input_data[0]['dataset']
    # cluster_algorithm = input_data[0]['clustering_algorithm']

    # Plot results together:
    plt.grid()
    plot_data = []
    for input_dat, output_dat in zip(input_data, output_data):
        plot_data.append((output_dat[input_param_name], output_dat[output_param_name]))

    # Sort the data by the cluster size
    to_plot = np.array(plot_data)

    plt.title(title, **font_dictionary)
    plt.plot(to_plot[:, 0], to_plot[:, 1], 'o-')
    plt.ylabel(nicer_label_dictionary[output_param_name], **font_dictionary)
    plt.xlabel(nicer_label_dictionary[input_param_name], **font_dictionary)
    # plt.legend(loc="best")
    plt.savefig(save_name, dpi=1000, facecolor="white")
    plt.show()
    return to_plot[:, 0], to_plot[:, 1]


def generate_learning_curve_plot_regression(architecture,
                                            train_sizes,
                                            train_scores,
                                            valid_scores,
                                            output_file_path,
                                            ):
    # Gather statistics from the scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    output_string1 = "Maximum Validation Accuracy Samples = {} (Negative Mean Absolute Error={})\n".format(
        train_sizes[np.nanargmax(valid_scores_mean)],
        np.nanmax(valid_scores_mean)
    )
    output_string2 = "Validation Accuracy Over All Samples = {} (Negative Mean Absolute Error={})".format(
        train_sizes[-1],
        valid_scores_mean[-1]
    )
    print(output_string1)
    print(output_string2)
    # Write some simple information to a file to persist
    with open(os.path.join(output_file_path, "final_validation_accuracy.txt"), "w") as the_file:
        the_file.write(output_string1)
        the_file.write(output_string2)

    plt.grid()
    plt.ylim(-1.00, 0.0)
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
    )
    plt.fill_between(
        train_sizes,
        valid_scores_mean - valid_scores_std,
        valid_scores_mean + valid_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(
        train_sizes, train_scores_mean, "-", color="b", label="Training"
    )
    plt.plot(
        train_sizes, valid_scores_mean, "-", color="g", label="Cross-validation"
    )
    plt.legend(loc="best")
    plt.title('Learning Curve ({})'.format(nicer_label[architecture]), **font_dictionary)
    plt.xlabel("Training Examples", **font_dictionary)
    plt.ylabel("Negative Mean Absolute Error", **font_dictionary)
    plt.savefig(os.path.join(output_file_path, "learning_curve.png"), dpi=1000, facecolor="white")
    plt.show()
    plt.close()


def generate_unity_plot(y_truth_training,
                        y_predict_training,
                        y_truth_test,
                        y_predict_test,
                        name,
                        architecture,
                        class_name,
                        class_units,
                        output_limits=None,
                        ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    if output_limits is not None:
        plt.xlim(output_limits)
        plt.ylim(output_limits)
    plt.plot(y_truth_training,
             y_predict_training,
             '^',
             alpha=0.4,
             color='b',
             markerfacecolor="none",
             markersize="5",
             label="Training")
    plt.plot(y_truth_test,
             y_predict_test,
             '.',
             color='tab:orange',
             markersize="7",
             label="Test")
    plt.plot(y_truth_training,
             y_truth_training,
             "-",
             linewidth=1,
             color='k',
             label="Truth Reference")
    plt.grid()
    plt.title(nicer_label[architecture], **font_dictionary)
    plt.xlabel(class_name + " Truth Measurement (" + class_units + ")", **font_dictionary)
    plt.ylabel(class_name + " Model Prediction (" + class_units + ")", **font_dictionary)
    plt.legend(loc='best')
    plt.savefig(name + ".png", dpi=1000, facecolor="white")
    plt.show()

