import numpy as np
from imblearn.over_sampling import RandomOverSampler


def perform_oversample(training_data, validation_data):
    # Check number of unique values
    oversample_train = RandomOverSampler(sampling_strategy='not majority', random_state=0)
    oversample_valid = RandomOverSampler(sampling_strategy='not majority', random_state=0)

    x_over_train, y_over_train = oversample_train.fit_resample(training_data[:, :-1], training_data[:, -1])
    x_over_valid, y_over_valid = oversample_valid.fit_resample(validation_data[:, :-1], validation_data[:, -1])

    train_data = np.concatenate((x_over_train, y_over_train.reshape(len(y_over_train), -1)), axis=1)
    valid_data = np.concatenate((x_over_valid, y_over_valid.reshape(len(y_over_valid), -1)), axis=1)

    return train_data, valid_data
