import numpy as np
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(y_truth, y_predict):
    return confusion_matrix(y_truth, y_predict)


def get_true_positive_rates(confusion_mat):
    return np.divide(np.diag(confusion_mat), np.sum(confusion_mat, axis=1))
