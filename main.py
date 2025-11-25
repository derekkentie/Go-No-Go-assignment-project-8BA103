import numpy as np
import pandas as pd
import matplotlib as plt

# The purpose of the model is to accurately categorise the sleep quality of each sample in the categories: bad, medium and good,
# based on the variables: Age, sleep duration, study hours, screen time, caffeine intake and physical activity.
# This means that we 6 variables for the input layer (+1 bias node), and 3 categories for the output layer.

def data_extraction(raw_data):
    """ This function extracts and returns the X_train and y_train for the train_model function"""

    data = [line.strip().split(',') for line in open(raw_data, 'r')] # extracting all the data from the raw csv file and placing it in a list of lists
    data.pop(0) #removing the header

    # creating the lists from the input variables and labels
    X_train = []
    y_train = []
    for line in data:
        y_train.append(line.pop(-1))
        X_train.append(line)
    # converting the lists to numpy arrays for easier use further on
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train.shape)
    return X_train, y_train

data_extraction("data\\train.csv")

def train_model(X_train, y_train, n_hidden_layers = 6, n_nodes_per_layer = 1, max_iter = 1000):
    
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
