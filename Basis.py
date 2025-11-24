import numpy as np
import pandas
import matplotlib

def train_model(X_train, y_train, **hyperparams):
    """
    Trains your ML algorithm on the provided training data.

    Parameters:
        X_train (numpy.ndarray): Training features, shape (n_samples, n_features)
        y_train (numpy.ndarray): Training labels, shape (n_samples,)
        **hyperparams: Algorithm-specific hyperparameters
                       (e.g., learning_rate=0.01, max_iter=1000, k=5, etc.)

    Returns:
        model_params (dict): A dictionary containing all information needed
                             to make predictions later.
                             For example, it might include learned weights,
                             biases, thresholds, or training statistics.
    """
    pass

def predict(X_test, model_params):
    """
    Uses the trained parameters to make predictions on new (test) data.

    Parameters:
        X_test (numpy.ndarray): Test features, shape (n_samples, n_features)
        model_params (dict): Dictionary of parameters returned by train_model()

    Returns:
        y_pred (numpy.ndarray): Predicted labels, shape (n_samples,)
    """
    pass