from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import pickle
from sklearn.model_selection import validation_curve, learning_curve
from utils.plot import generate_learning_curve_plot_regression, \
    generate_learning_curve_plot_nn, \
    generate_validation_curve_plot_regression, \
    generate_unity_plot
import time
from sklearn.model_selection import GridSearchCV, KFold
from utils.general_utils import get_arg_parser, \
    get_cluster_output_dir_name
import yaml
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

CROSS_VALIDATION_NUM_SPLITS = 10
NUM_THREADS = 8  # If running while not using computer, change this to -1
PRE_DISPATCH = 8  # If running while not using computer, change this to 8
SCALE_FACTOR = 100  # This is just for Joe's work,

dictionary_simple = {"neural_network": "kerasregressor",
                     "random_forest": "randomforestregressor",
                     "polynomial": "polynomialfeatures",
                     }
nicer_label = {"neural_network": "Feedforward Neural Network",
               "random_forest": "Random Forest Regressor",
               "polynomial": "Polynomial Regression"
               }

CLASS_NAMES = ["Thermal Resistance", "Pressure Drop"]
UNITS = ["$^\circ$C/W", "Pa"]


def create_model(n_features=10,
                 n_outputs=1,
                 hidden_count=3,
                 hidden_dim=64,
                 learning_rate=0.01,
                 ) -> tf.keras.Model:
    tf_model = tf.keras.Sequential()
    # Input layer:
    tf_model.add(tf.keras.Input(shape=(n_features,)))
    # Hidden layers:
    for _ in range(hidden_count):
        tf_model.add(tf.keras.layers.Dense(units=hidden_dim, activation='relu'))
    # Output layers:
    tf_model.add(tf.keras.layers.Dense(units=n_outputs))

    # Compile the model
    tf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss="mse",
                     metrics=["mse", "accuracy"]
                     )

    return tf_model


if __name__ == '__main__':
    # Set global random seed for tensorflow for consistent results
    tf.random.set_seed(1234)

    # Parse arguments
    inputs = vars(get_arg_parser().parse_args())

    # If the validation-curve flag is given and parameter range must also be given:
    validation_curve_not_exists = (inputs['validation_curve'] is None)
    validation_parameter_range_not_exists = (inputs['validation_parameter_range'] is None)
    if validation_curve_not_exists ^ validation_parameter_range_not_exists:
        raise RuntimeError("If running validation curves, there must be an accompanying range of the parameters to run")
    elif validation_curve_not_exists and validation_parameter_range_not_exists:
        # If not running validation curves, just run the program over the input parameters with defaults
        validation_curves_flag = False
    elif not (validation_curve_not_exists or validation_parameter_range_not_exists):
        validation_curves_flag = True

    # To keep it simple, don't allow cross validation and grid search at the same time
    grid_cv_search_dict = inputs['grid_cv_search_dict']
    if (grid_cv_search_dict is not None) and validation_curves_flag:
        raise RuntimeError("Don't run validation curves and grid search in the same run, do it separately")
    elif grid_cv_search_dict is not None:
        grid_cv_search_flag = True
    else:
        grid_cv_search_flag = False

    # Have the target output path and config string, write out (common method for all)
    folder_name = get_cluster_output_dir_name(inputs)
    full_path_output = os.path.join(os.path.join(inputs['output_path'],
                                                 os.path.split(inputs['dataset_path'])[-1]),
                                    folder_name)
    os.makedirs(full_path_output, exist_ok=True)  # It's on the user not to mess this up
    print("Created output folder: {0}".format(full_path_output))
    with open(os.path.join(full_path_output, 'inputs.yml'), 'w') as input_file:
        yaml.dump(inputs, input_file, default_flow_style=False)

    # Load data:
    data_path = inputs['dataset_path']
    training_data_file = os.path.join(data_path, "training.csv")
    validation_data_file = os.path.join(data_path, "validation.csv")

    # Check that the paths exist:
    if not os.path.exists(training_data_file):
        raise RuntimeError("Training Data File path does not exist: {0}".format(training_data_file))
    if not os.path.exists(validation_data_file):
        raise RuntimeError("Validation Data File path does not exist: {0}".format(validation_data_file))

    training_data = np.genfromtxt(training_data_file, delimiter=',')
    validation_data = np.genfromtxt(validation_data_file, delimiter=',')

    # Assumption: Outputs are justified to the right moving left
    x_train = training_data[:, :-inputs['num_outputs']]
    y_train = training_data[:, -inputs['num_outputs']:]
    x_valid = validation_data[:, :-inputs['num_outputs']]
    y_valid = validation_data[:, -inputs['num_outputs']:]

    y_train[:, 0] *= SCALE_FACTOR
    y_valid[:, 0] *= SCALE_FACTOR

    # Grab the number of features
    _, num_features = x_train.shape

    # Combine training and validation data for k-folds cross-validation
    x_train_dt = np.concatenate((x_train, x_valid), axis=0)
    y_train_dt = np.concatenate((y_train, y_valid), axis=0)

    model = None
    if inputs['architecture'] == "neural_network":
        model = make_pipeline(preprocessing.StandardScaler(),
                              KerasRegressor(model=create_model,
                                             n_features=num_features,
                                             n_outputs=inputs['num_outputs'],
                                             verbose=0,
                                             **inputs['algorithm_inputs']
                                             )
                              )
    elif inputs['architecture'] == "random_forest":
        model = make_pipeline(preprocessing.StandardScaler(),
                              RandomForestRegressor(**inputs['algorithm_inputs'])
                              )
    elif inputs['architecture'] == "polynomial":
        model = make_pipeline(preprocessing.StandardScaler(),
                              PolynomialFeatures(**inputs['algorithm_inputs']),
                              LinearRegression()
                              )
    else:
        raise RuntimeError("The following architecture does not exist! {}".format(inputs['architecture']))

    # Make sure that the model is no longer none
    assert (model is not None)

    # If generating a validation curve, do that here:
    if validation_curves_flag:
        # Run validation curve function
        param_type = type(getattr(model[dictionary_simple[inputs['architecture']]], inputs['validation_curve']))
        scoring_type = "neg_mean_absolute_error"
        train_scores, valid_scores = validation_curve(
            model,
            x_train_dt,
            y_train_dt,
            param_name=dictionary_simple[inputs['architecture']] + "__" + inputs['validation_curve'],
            param_range=np.array(inputs['validation_parameter_range']).astype(param_type),
            scoring=scoring_type,
            cv=KFold(n_splits=CROSS_VALIDATION_NUM_SPLITS,
                     random_state=0,
                     shuffle=True
                     ),
            verbose=1,
            n_jobs=NUM_THREADS,
            pre_dispatch=PRE_DISPATCH,
        )

        # Generate plot of the validation_curve
        generate_validation_curve_plot_regression(architecture=inputs['architecture'],
                                                  parameter=inputs['validation_curve'],
                                                  param_range=inputs['validation_parameter_range'],
                                                  train_scores=train_scores,
                                                  valid_scores=valid_scores,
                                                  output_file_path=full_path_output,
                                                  scoring=scoring_type
                                                  )
    elif grid_cv_search_flag:
        clf = GridSearchCV(estimator=model,
                           param_grid=grid_cv_search_dict,
                           cv=KFold(n_splits=CROSS_VALIDATION_NUM_SPLITS,
                                    random_state=0,
                                    shuffle=True
                                    ),
                           scoring='neg_mean_absolute_error',
                           verbose=5,
                           n_jobs=NUM_THREADS,
                           pre_dispatch=PRE_DISPATCH,
                           )
        clf.fit(x_train_dt, y_train_dt)
        with open(os.path.join(full_path_output, "best_params.txt"), "w") as the_file:
            best_params = "Best Params : {}\n".format(str(clf.best_params_))
            best_score = "Best Score  : {}".format(clf.best_score_)
            print(best_params)
            print(best_score)
            the_file.write(best_params)
            the_file.write(best_score)
    if not (grid_cv_search_flag or validation_curves_flag):
        # Generate a Learning Curve with k-Fold Cross Validation:
        train_sizes, train_scores, valid_scores = learning_curve(model,
                                                                 x_train_dt,
                                                                 y_train_dt,
                                                                 cv=KFold(
                                                                     n_splits=CROSS_VALIDATION_NUM_SPLITS,
                                                                     random_state=0,
                                                                     shuffle=True
                                                                 ),
                                                                 scoring="neg_mean_absolute_error",
                                                                 train_sizes=np.linspace(
                                                                     0.1,
                                                                     1.0,
                                                                     5),
                                                                 n_jobs=NUM_THREADS,
                                                                 pre_dispatch=PRE_DISPATCH,
                                                                 )
        # Save Learning Curve:
        generate_learning_curve_plot_regression(inputs['architecture'],
                                                train_sizes,
                                                train_scores,
                                                valid_scores,
                                                full_path_output,
                                                )

        # For neural network to get iterative validation data we have to do things weirdly
        start = time.time()
        if inputs['architecture'] == "neural_network":
            # Scale training data for validation scaling
            scaler = preprocessing.StandardScaler()
            scaler.fit(x_train)
            x_valid_scaled = scaler.transform(x_valid)
            # Fit the model to the data
            model.fit(X=x_train,
                      y=y_train,
                      kerasregressor__validation_data=(x_valid_scaled, y_valid)
                      )
            # Neural network we save the iterative history (other algorithms not iterative like this)
            history = model['kerasregressor'].history_

            generate_learning_curve_plot_nn(history, full_path_output)

            # Only save of scaler and history for nn
            # Save off training history for plots
            with open(os.path.join(full_path_output, "history"), 'wb') as file_pi:
                pickle.dump(history, file_pi)

            # Save the scaler as a pickle for testing to avoid testing pollution
            with open(os.path.join(full_path_output, 'scaler.pkl'), 'wb') as fid:
                pickle.dump(model['standardscaler'], fid)
        else:
            model.fit(X=x_train,
                      y=y_train
                      )
        end = time.time()

        # # Shapley Stuff
        # explainer = shap.Explainer(model.predict)
        # shap_values = explainer(x_train)
        # x = 1

        # Evaluate final model on validation data to get more meaningful metrics
        if inputs['num_outputs'] > 1:
            y_predict_valid = model.predict(x_valid)
            y_predict_train = model.predict(x_train)
            for element in range(inputs['num_outputs']):
                y_pred_valid = y_predict_valid[:, element]
                y_pred_train = y_predict_train[:, element]
                accuracy_iter_valid = np.mean(
                    np.abs((y_pred_valid - y_valid[:, element])))
                accuracy_iter_train = np.mean(
                    np.abs((y_pred_train - y_train[:, element])))
                print("Train Output {} Mean Absolute Error: {}".format(element,
                                                                       accuracy_iter_train,
                                                                       ))
                print("Valid Output {} Mean Absolute Error: {}".format(element,
                                                                       accuracy_iter_valid,
                                                                       ))
        else:
            y_predict = model.predict(x_valid)
            accuracy_iter = np.abs(np.mean((y_predict - y_valid)))
            print("Output Mean Abs Percent Error: {}".format(accuracy_iter))

        # Save the time that it took to train the model
        with open(os.path.join(full_path_output, 'training_time.txt'), 'w') as fid:
            delta_time = end - start
            print("Training time: {}".format(delta_time))
            fid.write(str(delta_time))

        # Save the model
        with open(os.path.join(full_path_output, 'model.pkl'), 'wb') as fid:
            pickle.dump(model, fid)

        # This should be in a separate script, but for quick and dirty lets do it this way
        if inputs['test']:
            # Test here for now, it's easier
            test_data_file = os.path.join(data_path, "test.csv")
            test_data = np.genfromtxt(test_data_file, delimiter=',')
            x_test = test_data[:, :-inputs['num_outputs']]
            y_test = test_data[:, -inputs['num_outputs']:]

            y_test[:, 0] *= SCALE_FACTOR
            if inputs['num_outputs'] > 1:
                y_predict = model.predict(x_test)
                for element in range(inputs['num_outputs']):
                    accuracy_iter = np.mean(
                        np.abs((y_predict[:, element] - y_test[:, element])))
                    print("Test Output {} Mean Absolute Error: {}".format(element,
                                                                          accuracy_iter,
                                                                          ))
            else:
                y_predict = model.predict(x_test)
                accuracy_iter = np.abs(np.mean((y_predict - y_test)))
                print("Test Mean Absolute Error: {}".format(accuracy_iter))

            # Make a plot of the data compared to labels
            y_predict_training = model.predict(x_train)
            y_truth_training = y_train
            y_predict_test = model.predict(x_test)
            y_truth_test = y_test

            # Undo the scaling stuff here
            y_predict_training[:, 0] /= SCALE_FACTOR
            y_predict_test[:, 0] /= SCALE_FACTOR
            y_truth_training[:, 0] /= SCALE_FACTOR
            y_truth_test[:, 0] /= SCALE_FACTOR

            if inputs['num_outputs'] > 1:
                for element in range(inputs['num_outputs']):
                    name = inputs['architecture'] + str(element)

                    generate_unity_plot(y_truth_training=y_truth_training[:, element],
                                        y_predict_training=y_predict_training[:, element],
                                        y_truth_test=y_truth_test[:, element],
                                        y_predict_test=y_predict_test[:, element],
                                        name=name,
                                        architecture=inputs['architecture'],
                                        class_name=CLASS_NAMES[element],
                                        class_units=UNITS[element]
                                        )

                    # Add another plot for a smaller scale, for lower pressure drop range
                    if element == 1:
                        generate_unity_plot(y_truth_training=y_truth_training[:, element],
                                            y_predict_training=y_predict_training[:, element],
                                            y_truth_test=y_truth_test[:, element],
                                            y_predict_test=y_predict_test[:, element],
                                            name=name + "_small",
                                            architecture=inputs['architecture'],
                                            class_name=CLASS_NAMES[element],
                                            class_units=UNITS[element],
                                            output_limits=(0.0, 2.25)
                                            )
